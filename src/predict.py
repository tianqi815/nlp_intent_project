"""
模型推理：加载已训练模型并执行文本意图预测。
"""

import json
import logging
import os
from typing import List

import torch
from peft import PeftConfig, PeftModel
from transformers import AutoModelForSequenceClassification

from src.model import create_tokenizer_for_model

logger = logging.getLogger(__name__)


def load_model_and_tokenizer(model_path: str):
    label_mapping_path = os.path.join(model_path, "label_mapping.json")
    if not os.path.isfile(label_mapping_path):
        raise FileNotFoundError(f"未找到 {label_mapping_path}，请指定正确的模型目录。")

    with open(label_mapping_path, "r", encoding="utf-8") as f:
        mapping_data = json.load(f)
    id_to_label = mapping_data["id_to_label"]
    num_labels = len(id_to_label)

    adapter_config_path = os.path.join(model_path, "adapter_config.json")
    if os.path.exists(adapter_config_path):
        logger.info("检测到 LoRA 适配器，使用 PEFT 加载...")
        peft_config = PeftConfig.from_pretrained(model_path)
        report_loggers = [
            logging.getLogger("transformers.utils.loading_report"),
            logging.getLogger("transformers.modeling_utils"),
        ]
        old_levels = [lg.level for lg in report_loggers]
        for lg in report_loggers:
            lg.setLevel(logging.ERROR)
        try:
            try:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    peft_config.base_model_name_or_path,
                    num_labels=num_labels,
                    dtype=torch.float32,
                )
            except TypeError:
                base_model = AutoModelForSequenceClassification.from_pretrained(
                    peft_config.base_model_name_or_path,
                    num_labels=num_labels,
                    torch_dtype=torch.float32,
                )
            model = PeftModel.from_pretrained(base_model, model_path)
        finally:
            for lg, lvl in zip(report_loggers, old_levels):
                lg.setLevel(lvl)
        tokenizer = create_tokenizer_for_model(
            model_path, peft_config.base_model_name_or_path
        )
    else:
        logger.info("检测到合并/完整模型，直接加载...")
        model = AutoModelForSequenceClassification.from_pretrained(
            model_path, num_labels=num_labels, torch_dtype=torch.float32
        )
        tokenizer = create_tokenizer_for_model(model_path)

    return model, tokenizer, id_to_label


def predict_with_loaded(
    model,
    tokenizer,
    id_to_label: dict,
    texts: List[str],
    max_length: int = 512,
) -> List[dict]:
    if not texts:
        return []
    results: List[dict] = []
    device = next(model.parameters()).device
    for text in texts:
        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=max_length,
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
            probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
            pred_id = probs.argmax().item()
            confidence = probs[0][pred_id].item()
        label_name = id_to_label[str(pred_id)]
        results.append(
            {
                "text": text,
                "label": label_name,
                "label_id": pred_id,
                "confidence": confidence,
            }
        )
    return results


def predict(
    model_path: str,
    texts: List[str],
    max_length: int = 512,
) -> List[dict]:
    model, tokenizer, id_to_label = load_model_and_tokenizer(model_path)
    return predict_with_loaded(model, tokenizer, id_to_label, texts, max_length)

