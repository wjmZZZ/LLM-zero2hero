# LLM-zero2hero🚀

English | [简体中文](README.md)

[![GitHub license](https://img.shields.io/github/license/wjmZZZ/LLM-zero2hero)](https://github.com/wjmZZZ/LLM-zero2hero/blob/main/LICENSE)
[![Python Version](https://img.shields.io/badge/Python-3.11+-blue)](https://www.python.org/)
[![GitHub stars](https://img.shields.io/github/stars/wjmZZZ/LLM-zero2hero)](https://github.com/wjmZZZ/LLM-zero2hero/stargazers)
[![GitHub forks](https://img.shields.io/github/forks/wjmZZZ/LLM-zero2hero)](https://github.com/wjmZZZ/LLM-zero2hero/network/members)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/wjmZZZ/LLM-zero2hero/pulls)

LLM-zero2hero is a highly decoupled Large Language Model (LLM) fine-tuning project that supports customizable training, validation, and inference processes, enabling both full-parameter and LoRA fine-tuning.


## Key Features

- 🔥 Support for multiple training workflows including SFT, DPO
- Support for single and multi-GPU training
- Support for single-turn and multi-turn dialogue fine-tuning
- 🔥 Validation during training using metrics like Perplexity, BLEU, AI evaluation
- Support for various precisions: int4, int8, float16, bfloat16, etc.
- 🔥 Highly customizable training and evaluation workflows

## Project Structure

```
LLM-zero2hero/
├── scripts/        # Running scripts
├── configs/        # Configuration files
├── data/           # Data directory
└── src/            # Source code
    ├── Main.py     # Entry point
    ├── Args/       # Argument configurations
    ├── Enviroment/ # Environment setup
    ├── Model/      # Model-related code
    ├── Train/      # Training-related code
    ├── Evaluation/ # Validation and inference
    ├── Dataset/    # Dataset processing
    ├── Utils/      # Utilities
    └── Others/     # Miscellaneous
```

## Quick Start

1. Environment Setup

```bash
conda create -n llm-zero2hero python=3.11
git clone https://github.com/wjmZZZ/LLM-zero2hero.git
cd LLM-zero2hero
pip install -r requirements.txt
```

2. Configure Training Parameters

Modify `configs/cfg.json` to set the required training parameters. For detailed parameter descriptions, refer to [Configuration Documentation](configs/README.md).

3. Prepare Training Data

Supports ShareGPT format dialogue data. Data format examples:

### SFT Training Data Format
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Hello"
    },
    {
      "from": "assistant", 
      "value": "Hello! Nice to meet you."
    }
  ]
}
```

### DPO Training Data Format
```json
{
  "conversations": [
    {
      "from": "human",
      "value": "Hello"
    },
    {
      "from": "chosen_gpt",
      "value": "Hello! Nice to meet you."
    },
    {
      "from": "rejected_gpt", 
      "value": "Hi."
    }
  ]
}
```

You can reference the `shibing624/sharegpt_gpt4` data repository and download using [huggingface mirror](https://hf-mirror.com/):

```bash
cd LLM-zero2hero
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download shibing624/sharegpt_gpt4 --local-dir data
```

4. Start Training

```bash
# SFT training
sh scripts/sft.sh              # Use first GPU by default
sh scripts/sft.sh 0            # Use GPU 0
sh scripts/sft.sh 0,1,2,3      # Multi-GPU training with GPUs 0,1,2,3
sh scripts/sft.sh -n 4         # Use first 4 GPUs
sh scripts/sft.sh -g 0,2 -n 2  # Specify GPUs 0 and 2
```

Script parameters:
- Direct number: Specify single GPU (e.g., `sh scripts/sft.sh 0`)
- GPU list: Specify multiple GPUs (e.g., `sh scripts/sft.sh 0,1,2,3`)
- `-g, --gpus`: Specify GPU list (e.g., `-g 0,2,4`)
- `-n, --num_gpus`: Specify number of GPUs (e.g., `-n 4`)

Notes:
- Distributed training is automatically enabled for multi-GPU training
- DeepSpeed is recommended for multi-GPU setups
- LoRA is recommended for single-GPU training to reduce memory usage

## Main Features

### 1. Training Methods
- SFT (Supervised Fine-tuning): Standard supervised fine-tuning
- DPO (Direct Preference Optimization): Reinforcement learning based on human preferences

### 2. Evaluation Metrics
- Perplexity: Language model evaluation
- BLEU: Text similarity evaluation
- AI: Generation quality evaluation using LLMs

### 3. Optimization Methods
- LoRA: Low-Rank Adaptation
- DeepSpeed: Distributed training optimization
- Flash Attention 2: Efficient attention mechanism

### 4. Experiment Management
- W&B: Experiment tracking and visualization
- Custom validation strategies
- Flexible checkpoint saving

## Changelog

[2024-10-28] Integrated DPO training method

[2024-08-23] Integrated AI evaluation metrics, compatible with **Silicon Flow**, free Qwen7B evaluation (**OpenAI**-like API)

[2024-08-18] Integrated Weights & Biases (**W&B**) logger. Improved experiment management and real-time monitoring

## Acknowledgments

This project benefits from the following open-source projects:
- [transformers](https://github.com/huggingface/transformers)
- [h2o-llmstudio](https://github.com/h2oai/h2o-llmstudio)

Thanks to the authors of these projects.

## License
[Apache License 2.0](LICENSE) 