# LLM-zero2hero🚀

LLM-zero2hero是一个高度解耦的大语言模型(LLM)微调项目，支持自定义训练、验证和推理过程，实现全量微调和LoRA微调。

## 主要特性

- 🔥支持SFT、DPO等多种训练流程
- 支持单卡和多卡训练
- 支持单轮和多轮对话微调
- 🔥训练过程中支持使用Perplexity、BLEU、AI等指标进行验证
- 支持多种精度：int4、int8、float16、bfloat16等
- 🔥高度可定制化的训练和评估流程

## 项目结构

```
LLM-zero2hero/
├── scripts/        # 运行脚本
├── configs/        # 配置文件（实验前配置）
├── data/           # 数据部分
└── src/            # 源代码目录
    ├── Main.py     # 入口文件
    ├── Args/       # 参数配置
    ├── Enviroment/ # 实验环境配置代码
    ├── Model/      # 模型相关代码
    ├── Train/      # 训练相关代码
    ├── Evaluation/ # 验证推理代码
    ├── Dataset/    # 数据相关代码
    ├── Utils/      # 工具包
    └── Others/     # 运行相关杂项
```

## 快速开始

1. 环境配置

```bash
conda create -n llm-zero2hero python=3.11
git clone https://github.com/wjmZZZ/LLM-zero2hero.git
cd LLM-zero2hero
pip install -r requirements.txt
```

2. 配置训练参数

修改 `configs/cfg.json` 文件以设置所需的训练参数。详细参数说明请参考 [配置说明文档](configs/README.md)。

3. 准备训练数据

支持ShareGPT格式的对话数据。数据格式示例:

### SFT训练数据格式
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "你好"
    },
    {
      "from": "assistant", 
      "value": "你好！很高兴见到你。"
    }
  ]
}
```

### DPO训练数据格式
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "你好"
    },
    {
      "from": "chosen_gpt",
      "value": "你好！很高兴见到你。"
    },
    {
      "from": "rejected_gpt", 
      "value": "你好。"
    }
  ]
}
```

可参考 `shibing624/sharegpt_gpt4` 数据仓库，使用 [huggingface镜像](https://hf-mirror.com/) 下载数据:

```bash
cd LLM-zero2hero
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download shibing624/sharegpt_gpt4 --local-dir data
```

4. 开始训练

```bash
# SFT训练
sh scripts/sft.sh              # 默认使用第一张显卡
sh scripts/sft.sh 0            # 使用GPU 0
sh scripts/sft.sh 0,1,2,3      # 使用GPU 0,1,2,3多卡训练
sh scripts/sft.sh -n 4         # 使用前4张显卡
sh scripts/sft.sh -g 0,2 -n 2  # 指定使用GPU 0和2两张卡
```

脚本参数说明:
- 直接跟数字: 指定单张GPU (例如: `sh scripts/sft.sh 0`)
- 直接跟GPU列表: 指定多张GPU (例如: `sh scripts/sft.sh 0,1,2,3`)
- `-g, --gpus`: 指定要使用的GPU列表 (例如: `-g 0,2,4`)
- `-n, --num_gpus`: 指定要使用的GPU数量 (例如: `-n 4`)

注意:
- 使用多卡训练时会自动开启分布式训练
- 使用DeepSpeed时建议使用多卡以发挥其优势
- 单卡训练时建议使用LoRA以减少显存占用

## 主要功能

### 1. 训练方法
- SFT(Supervised Fine-tuning): 标准的监督微调
- DPO(Direct Preference Optimization): 基于人类偏好的强化学习训练

### 2. 评估指标
- Perplexity: 困惑度评估
- BLEU: 文本相似度评估  
- AI: 使用大模型评估生成质量

### 3. 优化方法
- LoRA: 低秩适应微调
- DeepSpeed: 分布式训练优化
- Flash Attention 2: 高效注意力机制

### 4. 实验管理
- W&B: 实验跟踪与可视化
- 自定义验证策略
- 灵活的checkpoint保存

## 更新日志

[2024-10-28] 集成DPO训练方法

[2024-08-23] 集成AI评估指标，适配**硅基流动**，免费使用Qwen7B评估生成效果（**OpenAI**式接口）

[2024-08-18] 集成 Weights & Biases (**W&B**) 日志记录器。改进实验管理与实时监控功能

## 致谢

本项目受益于以下开源项目：
- [transformers](https://github.com/huggingface/transformers)
- [h2o-llmstudio](https://github.com/h2oai/h2o-llmstudio)

感谢这些项目作者的贡献。

## 许可证
[Apache License 2.0](LICENSE)

