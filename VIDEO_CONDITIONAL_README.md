# Video-Conditional Inference

这是一个视频条件生成推理代码，支持以下功能：

## 功能特点

1. **视频条件生成**: 使用输入视频和第一个prompt作为条件，生成符合第二个prompt的视频
2. **KV缓存机制**: 先将给定视频和prompt1缓存，然后基于prompt2生成新视频
3. **多卡推理**: 支持单卡和多卡分布式推理
4. **LoRA支持**: 可选的LoRA微调模型加载

## 文件说明

- `video_conditional_inference.py`: 主推理代码
- `configs/longlive_video_conditional_inference.yaml`: 配置文件
- `example/video_conditional_example.jsonl`: 示例数据格式
- `video_conditional_inference.sh`: 运行脚本
- `utils/dataset.py`: 新增了 `VideoConditionalDataset` 类

## 数据格式

JSONL文件中每行包含一个JSON对象：

```json
{
  "video_path": "path/to/video.mp4",
  "prompt1": "conditioning prompt describing the input video",
  "prompt2": "generation prompt for the new video to generate"
}
```

示例：
```json
{"video_path": "videos/beach.mp4", "prompt1": "A serene beach scene with gentle waves", "prompt2": "The waves become larger and more turbulent"}
```

## 配置参数

关键配置参数（在 `configs/longlive_video_conditional_inference.yaml`）：

- `num_conditioning_frames`: 从输入视频中使用的条件帧数（默认40帧）
- `num_output_frames`: 输出视频总帧数（包括条件帧，默认240帧）
- `context_noise`: 上下文帧的噪声级别（0=干净）
- `data_path`: JSONL数据文件路径
- `output_folder`: 输出视频保存目录

## 使用方法

### 1. 准备数据

创建JSONL文件，格式如上所示。确保 `video_path` 指向有效的视频文件。

### 2. 修改配置

编辑 `configs/longlive_video_conditional_inference.yaml`:

```yaml
data_path: longlive_models/prompts/video_conditional_example.jsonl  # 你的JSONL文件
output_folder: videos/video_conditional  # 输出目录
num_conditioning_frames: 40  # 条件帧数
num_output_frames: 240       # 总输出帧数
```

### 3. 运行推理

**单卡推理**:
```bash
bash video_conditional_inference.sh
```

或直接：
```bash
python video_conditional_inference.py --config_path configs/longlive_video_conditional_inference.yaml
```

**多卡推理** (例如4卡):
```bash
torchrun --nproc_per_node=4 video_conditional_inference.py --config_path configs/longlive_video_conditional_inference.yaml
```

或使用脚本：
```bash
WORLD_SIZE=4 torchrun --nproc_per_node=4 video_conditional_inference.sh
```

## 工作流程

1. **加载视频**: 从 `video_path` 加载前 `num_conditioning_frames` 帧
2. **视频编码**: 使用VAE将视频帧编码到潜在空间
3. **文本编码**: 分别编码 `prompt1` 和 `prompt2`
4. **缓存阶段**: 
   - 使用条件视频帧 + prompt1 初始化KV缓存
   - 这建立了视频内容的上下文记忆
5. **生成阶段**:
   - 重置cross-attention缓存
   - 使用prompt2生成剩余的 `num_output_frames - num_conditioning_frames` 帧
   - KV缓存保留了前面视频的时序信息
6. **解码输出**: 使用VAE解码潜在帧为最终视频

## 注意事项

1. **视频分辨率**: 输入视频会被自动调整到模型分辨率（480x832）
2. **帧数要求**: `num_output_frames` 必须是 `num_frame_per_block`（默认3）的倍数
3. **内存管理**: 低显存模式（<40GB）会自动启用，将部分数据移到CPU
4. **条件帧数**: `num_conditioning_frames` 应该足够表达视频内容，但不要太长（推荐40-80帧）

## 与其他推理模式的区别

- **inference.py**: 纯文本到视频生成（T2V）
- **interactive_inference.py**: 多段prompt切换生成
- **video_conditional_inference.py** (本脚本): 
  - 视频+文本条件生成
  - **没有switch_prompt机制**
  - 先缓存视频内容，再生成新视频

## 示例输出

生成的视频将保存为：
```
videos/video_conditional/rank0-0-0_lora.mp4
videos/video_conditional/rank0-1-0_lora.mp4
...
```

文件名格式：`rank{rank}-{idx}-{seed}_{model_type}.mp4`
- `rank`: GPU编号
- `idx`: 样本索引
- `seed`: 随机种子索引
- `model_type`: 模型类型（lora/ema/regular）

## 故障排查

1. **找不到视频文件**: 检查 `video_path` 是否正确
2. **显存不足**: 减少 `num_samples` 或 `num_output_frames`
3. **分布式初始化失败**: 确保使用 `torchrun` 启动多卡推理
4. **生成质量差**: 调整 `num_conditioning_frames` 和检查prompt质量
