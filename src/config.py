"""
训练参数配置：统一维护训练默认超参数与 LoRA 默认开关。
"""

from typing import Any, Dict, Optional

from transformers import TrainingArguments

USE_LORA: bool = False

TRAINING_ARGS_DEFAULT: Dict[str, Any] = {
    "num_train_epochs": 5,
    "per_device_train_batch_size": 24,
    "per_device_eval_batch_size": 24,
    "learning_rate": 2e-4,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,
    "weight_decay": 0.05,
    "logging_dir": None,
    "logging_steps": 5,
    "eval_strategy": "epoch",
    "save_strategy": "epoch",
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "save_total_limit": 2,
    "report_to": [],
    "fp16": False,
    "bf16": True,
    "dataloader_drop_last": False,
    "eval_accumulation_steps": 1,
    "max_grad_norm": 1.0,
}


def get_training_args(
    output_dir: str,
    overrides: Optional[Dict[str, Any]] = None,
) -> TrainingArguments:
    kwargs = {**TRAINING_ARGS_DEFAULT}
    kwargs["output_dir"] = output_dir
    kwargs["logging_dir"] = f"{output_dir}/logs"
    if overrides:
        kwargs.update(overrides)
    return TrainingArguments(**kwargs)

