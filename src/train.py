#!/usr/bin/env python3
"""
训练入口：支持单阶段训练与二阶段意图训练。
推荐从项目根目录运行：python -m src.train
"""

import argparse
from datetime import datetime
import json
import logging
import os
import shutil
import sys
from typing import Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.config import USE_LORA, get_training_args
from src.dataset import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    STAGE1_ID_TO_LABEL,
    STAGE1_LABEL_TO_ID,
    STAGE2_ID_TO_LABEL,
    STAGE2_LABEL_TO_ID,
    load_and_split_data,
    load_stage1_and_split_data,
    load_stage2_and_split_data,
)
from src.model import (
    clear_gpu_memory,
    compute_security_metrics,
    create_full_security_model,
    create_lora_config,
    create_lora_security_model,
    get_all_gpu_info,
    log_memory_usage,
    resolve_model_path,
    set_gpu_device,
    setup_logging,
    tokenize_security_data,
)

logger = logging.getLogger(__name__)


def _archive_and_prune_output(output_dir: str, keep_last: int = 2) -> None:
    """
    将已有输出目录按时间戳归档，并只保留最近 keep_last 份归档。
    - 最新训练结果始终落在 output_dir，不改变 API 读取路径。
    """
    if not os.path.isdir(output_dir):
        return
    if keep_last <= 0:
        shutil.rmtree(output_dir)
        return

    parent = os.path.dirname(output_dir) or "."
    model_name = os.path.basename(output_dir.rstrip("\\/"))
    archive_root = os.path.join(parent, "_training_archives", model_name)
    os.makedirs(archive_root, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    archive_dir = os.path.join(archive_root, ts)
    if os.path.exists(archive_dir):
        archive_dir = f"{archive_dir}_{datetime.now().strftime('%f')}"

    shutil.move(output_dir, archive_dir)
    logger.info("Archived previous model to: %s", archive_dir)

    # 按目录名（时间戳）排序，清理超出保留数量的旧归档
    dirs = []
    for name in os.listdir(archive_root):
        p = os.path.join(archive_root, name)
        if os.path.isdir(p):
            dirs.append((name, p))
    dirs.sort(key=lambda x: x[0], reverse=True)
    for _, old_path in dirs[keep_last:]:
        shutil.rmtree(old_path, ignore_errors=True)
        logger.info("Pruned old archive: %s", old_path)


def _train_one_stage(
    *,
    stage_name: str,
    model_name: str = "modernbert-base",
    use_lora: Optional[bool] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    output_dir: str = "intent_intent_router",
    data_path: Optional[str] = None,
    training_overrides: Optional[Dict] = None,
    gpu_id: Optional[int] = None,
    overwrite_output: bool = True,
    keep_last_archives: int = 2,
    label_to_id: Optional[Dict[str, int]] = None,
    id_to_label: Optional[Dict[str, str]] = None,
    data_loader_fn=None,
) -> None:
    from transformers import Trainer

    global logger
    logger = setup_logging()

    # - 数据加载/预处理：`load_and_split_data`（封装在 `data/dataset_loader.py` 里）
    # - 加载 tokenizer + 加载预训练权重 + 构建模型结构：`create_*_security_model`（封装在 `src/model.py`）
    # - 训练循环/优化器/学习率调度/评估/保存 checkpoint：`transformers.Trainer` 内部完成
    #
    # 因此你会感觉“逻辑没看到”，但实际上被库/封装函数隐藏了细节。

    if use_lora is None:
        use_lora = USE_LORA

    logger.info("开始训练阶段 %s（%s）", stage_name, "LoRA 微调" if use_lora else "全参数微调")
    device_str, selected_gpu = set_gpu_device(gpu_id=gpu_id, auto_select=(gpu_id is None))
    if selected_gpu >= 0:
        all_gpus = get_all_gpu_info()
        if all_gpus:
            for gpu in all_gpus:
                if gpu["id"] == selected_gpu:
                    logger.info(
                        "Using GPU %s: %s (%.1fGB free)",
                        selected_gpu,
                        gpu["name"],
                        gpu["free_memory_gb"],
                    )
                    break
        clear_gpu_memory()
        log_memory_usage("Pre-training")
    else:
        logger.info("未检测到 CUDA，使用 CPU 进行训练（速度较慢，建议有 GPU 时使用）。")
        log_memory_usage("Pre-training")

    model_path = resolve_model_path(model_name)
    logger.info("Using model: %s -> %s", model_name, model_path)

    if label_to_id is None:
        label_to_id = LABEL_TO_ID
    if id_to_label is None:
        id_to_label = ID_TO_LABEL
    if data_loader_fn is None:
        data_loader_fn = load_and_split_data

    train_data, val_data = data_loader_fn(data_path=data_path)
    logger.info("Training samples: %d, Validation samples: %d", len(train_data), len(val_data))
    logger.info("Labels: %s", label_to_id)

    num_labels = len(label_to_id)
    lora_config = None
    if use_lora:
        # (4) 加载/生成超参数配置（LoRA 专有超参）。训练超参在下面 `get_training_args()`。
        lora_config = create_lora_config(model_name, lora_rank, lora_alpha, lora_dropout)
        # (2)(3)(6) 加载预训练权重 + 加载 tokenizer + 定义模型结构：
        # - 内部调用 `AutoTokenizer.from_pretrained(...)`
        # - 内部调用 `AutoModelForSequenceClassification.from_pretrained(...)` 来加载 base 权重
        # - 再通过 PEFT/LoRA “包一层”得到可训练的 LoRA 模型
        model, tokenizer = create_lora_security_model(model_path, num_labels, lora_config)
    else:
        # (2)(3)(6) 同上，但走全参数微调：直接加载序列分类模型并返回 tokenizer。
        model, tokenizer = create_full_security_model(model_path, num_labels)

    # (1) 数据集预处理的一部分：把 text/label 通过 tokenizer 转成模型需要的张量字段。
    train_dataset = tokenize_security_data(train_data, tokenizer)
    val_dataset = tokenize_security_data(val_data, tokenizer)

    # (5) 设置模型保存路径：默认覆盖旧结果，并按时间戳保留最近若干归档。
    if overwrite_output and os.path.isdir(output_dir):
        _archive_and_prune_output(output_dir, keep_last=keep_last_archives)
    os.makedirs(output_dir, exist_ok=True)

    effective_overrides = dict(training_overrides) if training_overrides else {}
    if selected_gpu < 0:
        effective_overrides.setdefault("fp16", False)
        effective_overrides.setdefault("bf16", False)
        effective_overrides.setdefault("per_device_train_batch_size", 8)
        effective_overrides.setdefault("per_device_eval_batch_size", 8)

    training_args = get_training_args(output_dir, overrides=effective_overrides)
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        compute_metrics=compute_security_metrics,
    )

    logger.info("Starting training on %s...", device_str)
    # (7) 开始训练：
    # `Trainer.train()` 内部会完成：
    # - 构建 DataLoader（批处理、shuffle、collate）
    # - 创建优化器/学习率调度器（由 TrainingArguments + 默认策略决定）
    # - 前向/反向传播、梯度裁剪、日志记录、按策略保存 checkpoint、按策略评估
    trainer.train()

    # 额外显式保存（与 Trainer 的 checkpoint 不冲突）：
    # - 模型权重 / tokenizer 配置会以 HuggingFace 标准格式落到 output_dir
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    if not use_lora:
        for name in ("adapter_config.json", "adapter_model.safetensors", "adapter_model.bin"):
            p = os.path.join(output_dir, name)
            if os.path.isfile(p):
                try:
                    os.remove(p)
                    logger.info("Removed old LoRA file so inference uses full model: %s", name)
                except OSError as exc:
                    logger.warning("Could not remove %s: %s", p, exc)

    label_mapping_data = {"label_to_id": label_to_id, "id_to_label": id_to_label}
    with open(os.path.join(output_dir, "label_mapping.json"), "w", encoding="utf-8") as f:
        json.dump(label_mapping_data, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, "lora_config.json"), "w", encoding="utf-8") as f:
        if use_lora and lora_config is not None:
            json.dump(lora_config, f, indent=2)
        else:
            json.dump({"use_lora": False}, f, indent=2)

    eval_results = trainer.evaluate()
    logger.info("Validation Results:")
    for key in ("eval_accuracy", "eval_f1", "eval_precision", "eval_recall"):
        if key in eval_results:
            logger.info("  %s: %.4f", key.replace("eval_", "").capitalize(), eval_results[key])
    logger.info("Model saved to: %s", output_dir)


def train(
    model_name: str = "modernbert-base",
    use_lora: Optional[bool] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    output_dir: str = "intent_binary_lora_modernbert",
    data_path: Optional[str] = None,
    training_overrides: Optional[Dict] = None,
    gpu_id: Optional[int] = None,
    overwrite_output: bool = True,
    keep_last_archives: int = 2,
) -> None:
    """
    兼容旧入口：单模型训练（使用最终标签映射）。
    """
    _train_one_stage(
        stage_name="single",
        model_name=model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        output_dir=output_dir,
        data_path=data_path,
        training_overrides=training_overrides,
        gpu_id=gpu_id,
        overwrite_output=overwrite_output,
        keep_last_archives=keep_last_archives,
        label_to_id=LABEL_TO_ID,
        id_to_label=ID_TO_LABEL,
        data_loader_fn=load_and_split_data,
    )


def train_two_stage(
    model_name: str = "modernbert-base",
    use_lora: Optional[bool] = None,
    lora_rank: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    output_dir: str = "intent_two_stage_modernbert",
    data_path: Optional[str] = None,
    training_overrides: Optional[Dict] = None,
    gpu_id: Optional[int] = None,
    overwrite_output: bool = True,
    keep_last_archives: int = 2,
) -> None:
    """
    二阶段训练：
    - stage1: in_scope vs general_response
    - stage2: project monitoring vs purchasing record summary
    """
    stage1_dir = os.path.join(output_dir, "stage1_model")
    stage2_dir = os.path.join(output_dir, "stage2_model")
    if overwrite_output and os.path.isdir(output_dir):
        _archive_and_prune_output(output_dir, keep_last=keep_last_archives)
    os.makedirs(output_dir, exist_ok=True)

    _train_one_stage(
        stage_name="stage1",
        model_name=model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        output_dir=stage1_dir,
        data_path=data_path,
        training_overrides=training_overrides,
        gpu_id=gpu_id,
        overwrite_output=False,
        keep_last_archives=keep_last_archives,
        label_to_id=STAGE1_LABEL_TO_ID,
        id_to_label=STAGE1_ID_TO_LABEL,
        data_loader_fn=load_stage1_and_split_data,
    )

    _train_one_stage(
        stage_name="stage2",
        model_name=model_name,
        use_lora=use_lora,
        lora_rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        output_dir=stage2_dir,
        data_path=data_path,
        training_overrides=training_overrides,
        gpu_id=gpu_id,
        overwrite_output=False,
        keep_last_archives=keep_last_archives,
        label_to_id=STAGE2_LABEL_TO_ID,
        id_to_label=STAGE2_ID_TO_LABEL,
        data_loader_fn=load_stage2_and_split_data,
    )

    pipeline_config = {
        "pipeline_type": "two_stage_intent",
        "stage1_model_dir": "stage1_model",
        "stage2_model_dir": "stage2_model",
        "stage1_labels": STAGE1_LABEL_TO_ID,
        "stage2_labels": STAGE2_LABEL_TO_ID,
        "final_labels": list(LABEL_TO_ID.keys()),
    }
    with open(os.path.join(output_dir, "pipeline_config.json"), "w", encoding="utf-8") as f:
        json.dump(pipeline_config, f, ensure_ascii=False, indent=2)
    logger.info("Two-stage pipeline saved to: %s", output_dir)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="训练意图分类模型（支持二阶段）")
    parser.add_argument("--data", type=str, default=None, help="训练数据 JSON 路径；不传则使用 data 模块默认数据")
    parser.add_argument("--model-name", type=str, default="modernbert-base", help="基座模型名或目录")
    parser.add_argument("--model-path", type=str, default="intent_two_stage_modernbert", help="模型保存目录")
    parser.add_argument("--epochs", type=int, default=None, help="训练轮数（覆盖 src.config 默认值）")
    parser.add_argument("--batch-size", type=int, default=None, help="batch 大小")
    parser.add_argument("--learning-rate", type=float, default=None, help="学习率")
    parser.add_argument("--lora", action="store_true", help="启用 LoRA 微调")
    parser.add_argument("--no-lora", action="store_true", help="全参数微调")
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.1)
    parser.add_argument("--gpu-id", type=int, default=None, help="指定 GPU 卡号")
    parser.add_argument("--single-stage", action="store_true", help="仅训练单模型（兼容旧流程）")
    parser.add_argument("--keep-old-output", action="store_true", help="保留既有输出目录，不覆盖旧训练结果")
    parser.add_argument("--keep-last-archives", type=int, default=2, help="覆盖训练时按时间戳保留最近 N 份归档（默认 2）")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    use_lora = None
    if args.lora:
        use_lora = True
    if args.no_lora:
        use_lora = False

    overrides: Dict[str, object] = {}
    if args.epochs is not None:
        overrides["num_train_epochs"] = args.epochs
    if args.batch_size is not None:
        overrides["per_device_train_batch_size"] = args.batch_size
        overrides["per_device_eval_batch_size"] = args.batch_size
    if args.learning_rate is not None:
        overrides["learning_rate"] = args.learning_rate

    if args.single_stage:
        train(
            model_name=args.model_name,
            output_dir=args.model_path,
            data_path=args.data,
            use_lora=use_lora,
            lora_rank=args.lora_rank,
            lora_alpha=args.lora_alpha,
            lora_dropout=args.lora_dropout,
            training_overrides=overrides if overrides else None,
            gpu_id=args.gpu_id,
            overwrite_output=not args.keep_old_output,
            keep_last_archives=args.keep_last_archives,
        )
        return

    train_two_stage(
        model_name=args.model_name,
        output_dir=args.model_path,
        data_path=args.data,
        use_lora=use_lora,
        lora_rank=args.lora_rank,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        training_overrides=overrides if overrides else None,
        gpu_id=args.gpu_id,
        overwrite_output=not args.keep_old_output,
        keep_last_archives=args.keep_last_archives,
    )


if __name__ == "__main__":
    main()

