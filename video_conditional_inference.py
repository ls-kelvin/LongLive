# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES
#
# Licensed under the Apache License, Version 2.0 (the "License").
# You may not use this file except in compliance with the License.
# To view a copy of this license, visit http://www.apache.org/licenses/LICENSE-2.0
#
# No warranties are given. The work is provided "AS IS", without warranty of any kind, express or implied.
#
# SPDX-License-Identifier: Apache-2.0

"""
Video-conditional inference script.
Reads a JSONL file with entries containing:
  - video_path: path to conditioning video
  - prompt1: text prompt for the conditioning video
  - prompt2: text prompt for the video to generate

The model first caches the given video with prompt1, then generates video for prompt2.
"""

import argparse
import os
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tqdm import tqdm
from torch.utils.data import DataLoader, SequentialSampler
from torch.utils.data.distributed import DistributedSampler
from torchvision.io import write_video, read_video
from torchvision import transforms
from einops import rearrange

from utils.misc import set_seed
from utils.distributed import barrier
from utils.memory import gpu, get_cuda_free_memory_gb, DynamicSwapInstaller, move_model_to_device_with_memory_preservation

from pipeline import CausalInferencePipeline
from utils.dataset import VideoConditionalDataset


# ----------------------------- Argument parsing -----------------------------
parser = argparse.ArgumentParser("Video-conditional inference")
parser.add_argument("--config_path", type=str, help="Path to the config file")
args = parser.parse_args()

config = OmegaConf.load(args.config_path)

# ----------------------------- Distributed setup -----------------------------
if "LOCAL_RANK" in os.environ:
    os.environ["NCCL_CROSS_NIC"] = "1"
    os.environ["NCCL_DEBUG"] = os.environ.get("NCCL_DEBUG", "INFO")
    os.environ["NCCL_TIMEOUT"] = os.environ.get("NCCL_TIMEOUT", "1800")
    
    local_rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", str(local_rank)))
    
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            rank=rank,
            world_size=world_size,
            timeout=torch.distributed.constants.default_pg_timeout
        )
    
    set_seed(config.seed + local_rank)
    config.distributed = True
    if rank == 0:
        print(f"[Rank {rank}] Initialized distributed processing on device {device}")
else:
    local_rank = 0
    rank = 0
    device = torch.device("cuda")
    set_seed(config.seed)
    config.distributed = False
    print(f"Single GPU mode on device {device}")

print(f'Free VRAM {get_cuda_free_memory_gb(device)} GB')
low_memory = get_cuda_free_memory_gb(device) < 40

torch.set_grad_enabled(False)

# ----------------------------- Initialize pipeline -----------------------------
pipeline = CausalInferencePipeline(config, device=device)

# Load generator checkpoint
if config.generator_ckpt:
    state_dict = torch.load(config.generator_ckpt, map_location="cpu")
    if "generator" in state_dict or "generator_ema" in state_dict:
        raw_gen_state_dict = state_dict["generator_ema" if config.use_ema else "generator"]
    elif "model" in state_dict:
        raw_gen_state_dict = state_dict["model"]
    else:
        raise ValueError(f"Generator state dict not found in {config.generator_ckpt}")
    
    if config.use_ema:
        def _clean_key(name: str) -> str:
            """Remove FSDP / checkpoint wrapper prefixes from parameter names."""
            name = name.replace("_fsdp_wrapped_module.", "")
            return name

        cleaned_state_dict = {_clean_key(k): v for k, v in raw_gen_state_dict.items()}
        missing, unexpected = pipeline.generator.load_state_dict(cleaned_state_dict, strict=False)
        if local_rank == 0:
            if len(missing) > 0:
                print(f"[Warning] {len(missing)} parameters missing: {missing[:8]} ...")
            if len(unexpected) > 0:
                print(f"[Warning] {len(unexpected)} unexpected params: {unexpected[:8]} ...")
    else:
        pipeline.generator.load_state_dict(raw_gen_state_dict)

# --------------------------- LoRA support (optional) ---------------------------
from utils.lora_utils import configure_lora_for_model
import peft

pipeline.is_lora_enabled = False
if getattr(config, "adapter", None) and configure_lora_for_model is not None:
    if local_rank == 0:
        print(f"LoRA enabled with config: {config.adapter}")
        print("Applying LoRA to generator (inference)...")
    
    pipeline.generator.model = configure_lora_for_model(
        pipeline.generator.model,
        model_name="generator",
        lora_config=config.adapter,
        is_main_process=(local_rank == 0),
    )

    lora_ckpt_path = getattr(config, "lora_ckpt", None)
    if lora_ckpt_path:
        if local_rank == 0:
            print(f"Loading LoRA checkpoint from {lora_ckpt_path}")
        lora_checkpoint = torch.load(lora_ckpt_path, map_location="cpu")
        if isinstance(lora_checkpoint, dict) and "generator_lora" in lora_checkpoint:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint["generator_lora"])
        else:
            peft.set_peft_model_state_dict(pipeline.generator.model, lora_checkpoint)
        if local_rank == 0:
            print("LoRA weights loaded for generator")
    else:
        if local_rank == 0:
            print("No LoRA checkpoint specified; using base weights with LoRA adapters initialized")

    pipeline.is_lora_enabled = True

# Move pipeline to appropriate dtype and device
pipeline = pipeline.to(dtype=torch.bfloat16)
if low_memory:
    DynamicSwapInstaller.install_model(pipeline.text_encoder, device=device)
pipeline.generator.to(device=device)
pipeline.vae.to(device=device)

# ----------------------------- Build dataset -----------------------------
dataset = VideoConditionalDataset(config.data_path, num_conditioning_frames=config.num_conditioning_frames)
num_prompts = len(dataset)
print(f"Number of samples: {num_prompts}")

if dist.is_initialized():
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
else:
    sampler = SequentialSampler(dataset)

dataloader = DataLoader(dataset, batch_size=1, sampler=sampler, num_workers=0, drop_last=False)

# Create output directory
if local_rank == 0:
    os.makedirs(config.output_folder, exist_ok=True)

if dist.is_initialized():
    dist.barrier()

# ----------------------------- Inference loop -----------------------------
for i, batch_data in tqdm(enumerate(dataloader), disable=(local_rank != 0)):
    idx = batch_data["idx"].item()
    video_path = batch_data["video_path"][0]  # unbatch
    prompt1 = batch_data["prompt1"][0]
    prompt2 = batch_data["prompt2"][0]
    
    if local_rank == 0:
        print(f"\n[Sample {idx}]")
        print(f"  Video: {video_path}")
        print(f"  Prompt1 (conditioning): {prompt1}")
        print(f"  Prompt2 (generation): {prompt2}")
    
    # Load and encode conditioning video
    video_frames, _, info = read_video(video_path, pts_unit='sec')
    video_frames = video_frames[:config.num_conditioning_frames]  # Take first N frames
    
    # Normalize video to [-1, 1]
    video_frames = video_frames.float() / 255.0
    video_frames = video_frames * 2.0 - 1.0
    
    # Rearrange to (T, C, H, W) and resize if needed
    video_frames = rearrange(video_frames, 't h w c -> t c h w')
    
    # Resize to model resolution (480x832)
    target_h, target_w = 480, 832
    if video_frames.shape[-2] != target_h or video_frames.shape[-1] != target_w:
        video_frames = torch.nn.functional.interpolate(
            video_frames, size=(target_h, target_w), mode='bilinear', align_corners=False
        )
    
    # Add batch dimension and move to device
    video_frames = video_frames.unsqueeze(0).to(device=device, dtype=torch.bfloat16)
    
    # Encode video to latent space using VAE
    with torch.no_grad():
        conditioning_latents = pipeline.vae.encode(video_frames)
    
    num_cond_frames = conditioning_latents.shape[1]
    print(f"Conditioning frames (latent): {num_cond_frames}")
    
    # Encode both prompts
    cond_dict_prompt1 = pipeline.text_encoder(text_prompts=[prompt1] * config.num_samples)
    cond_dict_prompt2 = pipeline.text_encoder(text_prompts=[prompt2] * config.num_samples)
    
    if low_memory:
        gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
        move_model_to_device_with_memory_preservation(
            pipeline.text_encoder,
            target_device=gpu,
            preserved_memory_gb=gpu_memory_preservation,
        )
    
    # Generate noise for the new frames to generate
    num_new_frames = config.num_output_frames - num_cond_frames
    sampled_noise = torch.randn(
        [config.num_samples, num_new_frames, 16, 60, 104],
        device=device,
        dtype=torch.bfloat16
    )
    
    # Initialize output with conditioning latents
    batch_size = config.num_samples
    output = torch.zeros(
        [batch_size, config.num_output_frames, 16, 60, 104],
        device=torch.device('cpu') if low_memory else device,
        dtype=torch.bfloat16
    )
    
    # Replicate conditioning latents for all samples in batch
    conditioning_latents_batch = conditioning_latents.repeat(batch_size, 1, 1, 1, 1)
    output[:, :num_cond_frames] = conditioning_latents_batch.to(output.device)
    
    # Initialize KV cache
    local_attn_cfg = getattr(config.model_kwargs, "local_attn_size", -1)
    if local_attn_cfg != -1:
        kv_cache_size = local_attn_cfg * pipeline.frame_seq_length
    else:
        kv_cache_size = config.num_output_frames * pipeline.frame_seq_length
    
    pipeline._initialize_kv_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device,
        kv_cache_size_override=kv_cache_size
    )
    pipeline._initialize_crossattn_cache(
        batch_size=batch_size,
        dtype=torch.bfloat16,
        device=device
    )
    
    pipeline.generator.model.local_attn_size = pipeline.local_attn_size
    pipeline._set_all_modules_max_attention_size(pipeline.local_attn_size)
    
    # Step 1: Cache conditioning frames with prompt1
    if local_rank == 0:
        print(f"Caching {num_cond_frames} conditioning frames with prompt1...")
    
    context_timestep = torch.ones(
        [batch_size, num_cond_frames],
        device=device,
        dtype=torch.int64
    ) * config.context_noise
    
    # Prepare block mask for conditioning frames
    block_mask = pipeline.generator.model._prepare_blockwise_causal_attn_mask(
        device=device,
        num_frames=num_cond_frames,
        frame_seqlen=pipeline.frame_seq_length,
        num_frame_per_block=pipeline.num_frame_per_block,
        local_attn_size=pipeline.local_attn_size
    )
    pipeline.generator.model.block_mask = block_mask
    
    # Run conditioning frames through model to populate cache
    with torch.no_grad():
        _ = pipeline.generator(
            noisy_image_or_video=conditioning_latents_batch.to(device),
            conditional_dict=cond_dict_prompt1,
            timestep=context_timestep,
            kv_cache=pipeline.kv_cache1,
            crossattn_cache=pipeline.crossattn_cache,
            current_start=0,
        )
    
    # Step 2: Reset cross-attention cache for new prompt
    for blk in pipeline.crossattn_cache:
        blk["k"].zero_()
        blk["v"].zero_()
        blk["is_init"] = False
    
    # Step 3: Generate new frames with prompt2
    if local_rank == 0:
        print(f"Generating {num_new_frames} new frames with prompt2...")
    
    current_start_frame = num_cond_frames
    num_blocks = num_new_frames // pipeline.num_frame_per_block
    
    for block_idx in range(num_blocks):
        current_num_frames = pipeline.num_frame_per_block
        noisy_input = sampled_noise[
            :, block_idx * current_num_frames : (block_idx + 1) * current_num_frames
        ]
        
        # Spatial denoising loop
        for index, current_timestep in enumerate(pipeline.denoising_step_list):
            timestep = torch.ones(
                [batch_size, current_num_frames],
                device=device,
                dtype=torch.int64
            ) * current_timestep
            
            if index < len(pipeline.denoising_step_list) - 1:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_dict_prompt2,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
                next_timestep = pipeline.denoising_step_list[index + 1]
                noisy_input = pipeline.scheduler.add_noise(
                    denoised_pred.flatten(0, 1),
                    torch.randn_like(denoised_pred.flatten(0, 1)),
                    next_timestep * torch.ones(
                        [batch_size * current_num_frames], device=device, dtype=torch.long
                    ),
                ).unflatten(0, denoised_pred.shape[:2])
            else:
                _, denoised_pred = pipeline.generator(
                    noisy_image_or_video=noisy_input,
                    conditional_dict=cond_dict_prompt2,
                    timestep=timestep,
                    kv_cache=pipeline.kv_cache1,
                    crossattn_cache=pipeline.crossattn_cache,
                    current_start=current_start_frame * pipeline.frame_seq_length,
                )
        
        # Store denoised output
        output[:, current_start_frame : current_start_frame + current_num_frames] = \
            denoised_pred.to(output.device)
        
        # Update cache with clean frames
        context_timestep = torch.ones_like(timestep) * config.context_noise
        with torch.no_grad():
            _ = pipeline.generator(
                noisy_image_or_video=denoised_pred,
                conditional_dict=cond_dict_prompt2,
                timestep=context_timestep,
                kv_cache=pipeline.kv_cache1,
                crossattn_cache=pipeline.crossattn_cache,
                current_start=current_start_frame * pipeline.frame_seq_length,
            )
        
        current_start_frame += current_num_frames
    
    # Decode latents to video
    if local_rank == 0:
        print("Decoding latents to video...")
    
    video = pipeline.vae.decode(output.to(device))
    video = rearrange(video, 'b t c h w -> b t h w c').cpu() * 255.0
    
    # Clear VAE cache
    pipeline.vae.model.clear_cache()
    
    # Determine model type for filename
    if hasattr(pipeline, 'is_lora_enabled') and pipeline.is_lora_enabled:
        model_type = "lora"
    elif getattr(config, 'use_ema', False):
        model_type = "ema"
    else:
        model_type = "regular"
    
    # Save videos
    for seed_idx in range(config.num_samples):
        if config.save_with_index:
            output_path = os.path.join(
                config.output_folder,
                f"rank{rank}-{idx}-{seed_idx}_{model_type}.mp4"
            )
        else:
            short_name = prompt2[:100].replace("/", "_").replace(" ", "_")
            output_path = os.path.join(
                config.output_folder,
                f"rank{rank}-{short_name}-{seed_idx}_{model_type}.mp4"
            )
        write_video(output_path, video[seed_idx].to(torch.uint8), fps=16)
        if local_rank == 0:
            print(f"Saved: {output_path}")
    
    if config.inference_iter != -1 and i >= config.inference_iter:
        break

if dist.is_initialized():
    dist.destroy_process_group()

if local_rank == 0:
    print("\nInference complete!")
