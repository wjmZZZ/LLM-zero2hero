# 配置参数说明

本项目支持两种训练方式的配置:
- `sft_cfg.json`: 用于SFT(Supervised Fine-tuning)监督微调
- `dpo_cfg.json`: 用于DPO(Direct Preference Optimization)偏好优化训练

## 配置结构

配置文件包含以下主要部分:

### 1. ExperimentArguments - 实验基本配置
```json
{
  "experiment_name": "实验名称",
  "sub_experiment_name": "子实验名称",
  "task": "任务类型(SFT/DPO)", 
  "output_dir": "输出目录",
  "use_wandb": "是否使用W&B记录实验",
  "wandb_entity": "W&B账户名"
}
```

### 2. DatasetArguments/DPODatasetArguments - 数据集配置
```json
{
  "train_data_dir": "训练数据路径",
  "valid_data_dir": "验证数据路径", 
  "valid_strategy": "验证策略(auto/custom)",
  "valid_size": "验证集比例",

  "system_column": "系统提示列名",
  "prompt_column": "输入提示列名",
  "answer_column": "回答列名",
  "rejected_answer_column": "DPO专用-被拒绝答案列名",

  "system_prefix": "系统提示前缀",
  "system_suffix": "系统提示后缀",
  "prompt_prefix": "输入提示前缀", 
  "prompt_suffix": "输入提示后缀",
  "response_prefix": "回答前缀",
  "response_suffix": "回答后缀"
}
```

### 3. ModelArguments - 模型配置
```json
{
  "llm_backbone": "底座模型路径",
  "backbone_dtype": "模型精度(float16/bfloat16等)",
  "use_pretrained_model": "是否使用预训练模型",
  "intermediate_dropout": "中间层dropout率",
  "trust_remote_code": "是否信任远程代码"
}
```

### 4. TrainingArguments/DPOTrainingArguments - 训练配置
```json
{
  "lora": "是否使用LoRA",
  "use_dora": "是否使用DoRA",
  "learning_rate": "学习率",
  "batch_size": "批次大小",
  "max_seq_length": "最大序列长度",
  "num_train_epochs": "训练轮数",
  "save_checkpoint": "保存检查点策略(best/last)",
  "num_validations_per_epoch": "每轮验证次数",
  "use_flash_attention_2": "是否使用Flash Attention 2",
  "loss_function": "损失函数",
  "optimizer": "优化器",
  "schedule": "学习率调度策略",
  "warmup_epochs": "预热轮数",
  "gradient_checkpointing": "是否使用梯度检查点"
}
```

### 5. InferenceArguments - 推理配置
```json
{
  "metric": "评估指标(Perplexity/BLEU/AI)",
  "batch_size_inference": "推理批次大小",
  "distributed_inference": "是否使用分布式推理",
  
  "AI_eval_model": "AI评估使用的模型",
  "AI_eval_template_name": "AI评估模板名称",
  "openai_api_key": "API密钥",
  "openai_base_url": "调用的基础URL"
}
```

### 6. EnvironmentArguments - 环境配置
```json
{
  "seed": "随机种子",
  "use_deepspeed": "是否使用DeepSpeed",
  "deepspeed_method": "DeepSpeed方法(ZeRO2/ZeRO3)",
  "compile_model": "是否编译模型",
  "mixed_precision": "是否使用混合精度训练"
}
```

## 参数可选值说明

### 1. 训练相关参数

#### Loss Functions (损失函数)
- `TokenAveragedCrossEntropyLoss`: 标准交叉熵损失
- `DPOLoss`: DPO训练专用损失函数

#### Optimizers (优化器)
- `AdamW`: AdamW优化器
- `Adam`: Adam优化器
- `SGD`: 随机梯度下降
- `Adafactor`: Adafactor优化器

#### Learning Rate Schedules (学习率调度)
- `Cosine`: 余弦退火调度
- `Linear`: 线性调度
- `Constant`: 常数学习率
- `ConstantWithWarmup`: 带预热的常数学习率

#### Checkpoint Saving (检查点保存)
- `best`: 仅保存最佳模型
- `last`: 仅保存最后一个检查点
- `all`: 保存所有检查点

### 2. 评估相关参数

#### Metrics (评估指标)
- `Perplexity`: 困惑度评估，用于评估模型的预测准确性
- `BLEU`: BLEU分数，用于评估生成文本与参考文本的相似度
- `AI`: 使用AI模型评估生成质量

#### AI Evaluation Models (AI评估可用模型)
- 可参考硅基流动网站使用的模型

#### AI Evaluation Templates (AI评估模板)
- `default`: 默认评估模板
- 可通过自定义模板扩展

### 3. 模型相关参数

#### Model Precision (模型精度)
- `float16`: 半精度浮点
- `bfloat16`: Brain浮点
- `int8`: 8位整数量化
- `int4`: 4位整数量化

#### LoRA Parameters (LoRA参数)
当`lora=true`时可用：
- `lora_r`: LoRA秩 (推荐值: 8, 16, 32, 64)
- `lora_alpha`: LoRA alpha值 (推荐值: 16, 32)
- `lora_dropout`: LoRA dropout率 (推荐值: 0.05-0.1)

#### DeepSpeed Methods
- `ZeRO2`: ZeRO优化器第2阶段
- `ZeRO3`: ZeRO优化器第3阶段

### 4. 数据格式示例

#### SFT训练数据格式
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

#### DPO训练数据格式
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

### 5. 特殊标记说明

不同模型的特殊标记可能不同，以下是一些常见模型的标记示例：

#### Qwen系列
```json
{
  "system_prefix": "<|im_start|>system\n",
  "system_suffix": "<|im_end|>\n",
  "prompt_prefix": "<|im_start|>user\n",
  "prompt_suffix": "<|im_end|>\n<|im_start|>assistant\n",
  "response_prefix": "",
  "response_suffix": "<|im_end|>\n"
}
```

#### ChatGLM系列
```json
{
  "system_prefix": "[SYSTEM]",
  "system_suffix": "[/SYSTEM]",
  "prompt_prefix": "[HUMAN]",
  "prompt_suffix": "[/HUMAN][ASSISTANT]",
  "response_prefix": "",
  "response_suffix": "[/ASSISTANT]"
}
```

## 配置示例

### SFT配置示例
```json
{
  "ExperimentArguments": {
    "experiment_name": "qwen-sft",
    "task": "SFT",
    "output_dir": "./outputs"
  },
  "ModelArguments": {
    "llm_backbone": "Qwen/Qwen-7B-Chat",
    "backbone_dtype": "bfloat16"
  },
  "TrainingArguments": {
    "lora": true,
    "learning_rate": 1e-4,
    "num_train_epochs": 3,
    "loss_function": "TokenAveragedCrossEntropyLoss"
  }
}
```

### DPO配置示例
```json
{
  "ExperimentArguments": {
    "experiment_name": "qwen-dpo",
    "task": "DPO",
    "output_dir": "./outputs"
  },
  "ModelArguments": {
    "llm_backbone": "Qwen/Qwen-7B-Chat",
    "backbone_dtype": "bfloat16"
  },
  "DPOTrainingArguments": {
    "lora": true,
    "learning_rate": 1e-5,
    "num_train_epochs": 1,
    "loss_function": "DPOLoss"
  }
}
```

## 注意事项

1. 数据格式要求:

   - 支持ShareGPT格式的对话数据

   - SFT训练需要prompt和response对

   - DPO训练需要prompt、chosen_response和rejected_response

2. 模型支持:

   - 支持Hugging Face格式的模型

   - 支持多种精度训练(int4/int8/float16/bfloat16)

   - 支持LoRA和全量微调

3. 分布式训练:

   - 支持单卡和多卡训练

   - 支持DeepSpeed ZeRO优化

   - 支持混合精度训练

4. 评估指标:

   - Perplexity: 困惑度评估

   - BLEU: 文本相似度评估

   - AI: 使用大模型评估生成质量

5. 实验管理:

   - 支持W&B实验跟踪

   - 支持自定义验证策略

   - 支持checkpoint保存 