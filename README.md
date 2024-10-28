# LLM-zero2hero🚀

LLM-zero2hero是一个高度解耦的大语言模型(LLM)微调项目，支持自定义训练、验证和推理过程，实现全量微调和LoRA微调。

## 主要特性

- 🔥支持SFT、DPO 等多种训练流程
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
    ├── Args/      # 参数配置
    ├── Enviroment/ # 实验环境配置代码
    ├── Model/      # 模型相关代码
    ├── Train/      # 训练相关代码
    ├── Evaluation/ # 验证推理代码
    ├── Dataset/    # 数据相关代码
    ├── Utils/      # 工具包
    └── Others/     # 运行相关杂项
```

## 更新日志
[2024-10-28]  集成DPO训练方法

[2024-08-23]  集成AI评估指标，适配**硅基流动**，免费使用Qwen7B评估生成效果（**OpenAI**式接口）

[2024-08-18]  集成 Weights & Biases (**W&B**) 日志记录器。改进实验管理与实时监控功能


## 快速开始

1. 环境配置

```bash
conda create -n llm-zero2hero python=3.11
git clone https://github.com/wjmZZZ/LLM-zero2hero.git
cd LLM-zero2hero
pip install -r requirements.txt
```

2. 配置训练参数

修改 `configs/cfg.json` 文件以设置所需的训练参数。

注意，DatasetArguments下的配置应该与所选backbone模型的特定格式相匹配。

例如，Qwen模型的配置示例：

```json
{
  "DatasetArguments": {
    "system_prefix": "<|im_start|>system\n",
    "system_default": "You are a helpful assistant.",
    "system_suffix": "<|im_end|>\n",
    "prompt_prefix": "<|im_start|>user\n",
    "prompt_suffix": "<|im_end|>\n<|im_start|>assistant\n",
    "response_prefix": "",
    "response_suffix": "<|im_end|>\n"
  }
}
```

3. 修改GPU设置

在 `scripts/llm.sh` 中调整GPU数量。

4. 开始训练

```bash
sh scripts/llm.sh
```



## 数据格式

目前支持shareGPT格式的对话数据

```json
[
  {
    "conversations": [
      {
        "from": "human",
        "value": ""
      },
      {
        "from": "gpt",
        "value": ""
      }
    ],
  }
]
```

可参考 `shibing624/sharegpt_gpt4` 数据仓库，使用 [huggingface镜像](https://hf-mirror.com/) 下载数据

```sh
cd LLM-zero2hero
export HF_ENDPOINT=https://hf-mirror.com
huggingface-cli download --repo-type dataset --resume-download shibing624/sharegpt_gpt4 --local-dir data
```





## 致谢

本项目受益于以下开源项目：
- [transformers](https://github.com/huggingface/transformers)
- [h2o-llmstudio](https://github.com/h2oai/h2o-llmstudio)

感谢这些项目作者的贡献。

## 许可证
 [Apache License 2.0](https://github.com/wjmZZZ/LLM-zero2hero/blob/main/LICENSE) 

