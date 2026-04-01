#!/usr/bin/env python3
"""
评估入口：支持单模型评估与二阶段评估（stage1/stage2/端到端）。
推荐从项目根目录运行：python -m src.evaluate
"""

import argparse
import logging
import os
import sys
from typing import Dict, Optional

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from src.dataset import (
    LABEL_TO_ID,
    load_stage1_test_data,
    load_stage2_test_data,
    load_test_data,
)
from src.model import compute_security_metrics, tokenize_security_data
from src.predict import load_model_and_tokenizer, predict_with_loaded

logger = logging.getLogger(__name__)


def run_evaluation(
    model_path: str,
    test_data_path: str,
    *,
    batch_size: int = 32,
    max_length: int = 512,
    device: Optional[str] = None,
) -> Dict[str, float]:
    model, tokenizer, _ = load_model_and_tokenizer(model_path)
    if device is not None:
        model = model.to(device)
    model.eval()

    test_samples = load_test_data(test_data_path)
    return _evaluate_loaded_model(
        model=model,
        tokenizer=tokenizer,
        test_samples=test_samples,
        model_path=model_path,
        batch_size=batch_size,
        max_length=max_length,
    )


def _evaluate_loaded_model(
    *,
    model,
    tokenizer,
    test_samples,
    model_path: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    from transformers import Trainer, TrainingArguments

    logger.info("测试集样本数: %d", len(test_samples))
    if not test_samples:
        raise ValueError("测试集为空，请检查 test_data_path")

    test_dataset = tokenize_security_data(test_samples, tokenizer, max_length=max_length)

    eval_output_dir = os.path.join(model_path, "eval_output")
    os.makedirs(eval_output_dir, exist_ok=True)
    eval_args = TrainingArguments(
        output_dir=eval_output_dir,
        per_device_eval_batch_size=batch_size,
        report_to=[],
        fp16=False,
        bf16=False,
    )

    trainer = Trainer(
        model=model,
        args=eval_args,
        eval_dataset=test_dataset,
        compute_metrics=compute_security_metrics,
    )

    metrics = trainer.evaluate()
    result: Dict[str, float] = {}
    for key, value in metrics.items():
        out_key = key.replace("eval_", "") if key.startswith("eval_") else key
        result[out_key] = float(value)
    return result


def run_two_stage_evaluation(
    model_root_path: str,
    test_data_path: str,
    *,
    batch_size: int = 32,
    max_length: int = 512,
) -> Dict[str, float]:
    from sklearn.metrics import accuracy_score, precision_recall_fscore_support

    stage1_model_path = os.path.join(model_root_path, "stage1_model")
    stage2_model_path = os.path.join(model_root_path, "stage2_model")
    if not os.path.isdir(stage1_model_path) or not os.path.isdir(stage2_model_path):
        raise FileNotFoundError("未找到 stage1_model 或 stage2_model，请检查模型目录")

    stage1_test_samples = load_stage1_test_data(test_data_path)
    stage2_test_samples = load_stage2_test_data(test_data_path)
    final_test_samples = load_test_data(test_data_path)

    stage1_model, stage1_tokenizer, stage1_id_to_label = load_model_and_tokenizer(stage1_model_path)
    stage2_model, stage2_tokenizer, stage2_id_to_label = load_model_and_tokenizer(stage2_model_path)
    stage1_model.eval()
    stage2_model.eval()

    stage1_metrics = _evaluate_loaded_model(
        model=stage1_model,
        tokenizer=stage1_tokenizer,
        test_samples=stage1_test_samples,
        model_path=stage1_model_path,
        batch_size=batch_size,
        max_length=max_length,
    )
    stage2_metrics = _evaluate_loaded_model(
        model=stage2_model,
        tokenizer=stage2_tokenizer,
        test_samples=stage2_test_samples,
        model_path=stage2_model_path,
        batch_size=batch_size,
        max_length=max_length,
    )

    y_true = [item["label"] for item in final_test_samples]
    y_pred = []
    for item in final_test_samples:
        text = item["text"]
        s1 = predict_with_loaded(stage1_model, stage1_tokenizer, stage1_id_to_label, [text], max_length=max_length)[0]
        if s1["label"] == "general_response":
            y_pred.append(LABEL_TO_ID["general_response"])
        else:
            s2 = predict_with_loaded(stage2_model, stage2_tokenizer, stage2_id_to_label, [text], max_length=max_length)[0]
            y_pred.append(s2["label_id"])

    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="macro", zero_division=0)
    result = {
        "stage1_samples": float(len(stage1_test_samples)),
        "stage2_samples": float(len(stage2_test_samples)),
        "end2end_accuracy": float(acc),
        "end2end_precision_macro": float(precision),
        "end2end_recall_macro": float(recall),
        "end2end_f1_macro": float(f1),
    }
    for k, v in stage1_metrics.items():
        result[f"stage1_{k}"] = float(v)
    for k, v in stage2_metrics.items():
        result[f"stage2_{k}"] = float(v)
    return result


def main(
    model_path: str = "intent_two_stage_modernbert",
    test_data_path: Optional[str] = None,
    batch_size: int = 32,
    max_length: int = 512,
    two_stage: bool = True,
) -> Dict[str, float]:
    if not test_data_path:
        test_data_path = "data/test_dataset.json"
    if not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"请指定有效的测试集文件: {test_data_path}")

    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    if two_stage:
        metrics = run_two_stage_evaluation(
            model_root_path=model_path,
            test_data_path=test_data_path,
            batch_size=batch_size,
            max_length=max_length,
        )
        for name in (
            "stage1_accuracy",
            "stage1_f1",
            "stage2_accuracy",
            "stage2_f1",
            "end2end_accuracy",
            "end2end_f1_macro",
        ):
            if name in metrics:
                logger.info("%s: %.4f", name, metrics[name])
        return metrics

    metrics = run_evaluation(model_path=model_path, test_data_path=test_data_path, batch_size=batch_size, max_length=max_length)
    for name in ("accuracy", "f1", "precision", "recall"):
        if name in metrics:
            logger.info("%s: %.4f", name, metrics[name])
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="在独立测试集上评估已训练的意图模型（支持二阶段）")
    parser.add_argument("--model-path", type=str, default="intent_two_stage_modernbert", help="模型所在目录")
    parser.add_argument("--test-data", type=str, default="data/test_dataset.json", help="测试集 JSON 路径")
    parser.add_argument("--batch-size", type=int, default=32, help="评估 batch 大小")
    parser.add_argument("--max-length", type=int, default=512, help="文本最大长度")
    parser.add_argument("--single-stage", action="store_true", help="按单模型方式评估")
    args = parser.parse_args()

    main(
        model_path=args.model_path,
        test_data_path=args.test_data,
        batch_size=args.batch_size,
        max_length=args.max_length,
        two_stage=not args.single_stage,
    )

