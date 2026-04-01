"""
模型与训练通用能力：
- 模型路径解析
- LoRA 配置与模型构建
- tokenizer / 数据 tokenize
- 评估指标
- GPU 与日志工具
"""

import gc
import logging
import os
from typing import Dict, List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig, TaskType, get_peft_model
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> logging.Logger:
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger(__name__)


def get_model_mapping() -> Dict[str, str]:
    return {
        "modernbert-base": "answerdotai/ModernBERT-base",
        "modernbert-large": "answerdotai/ModernBERT-large",
        "mmbert-base": "jhu-clsp/mmBERT-base",
        "mmbert-32k": "llm-semantic-router/mmbert-32k-yarn",
        "mmbert-32k-yarn": "llm-semantic-router/mmbert-32k-yarn",
        "bert-base-uncased": "bert-base-uncased",
        "bert-large-uncased": "bert-large-uncased",
        "roberta-base": "roberta-base",
        "roberta-large": "roberta-large",
        "deberta-v3-base": "microsoft/deberta-v3-base",
        "deberta-v3-large": "microsoft/deberta-v3-large",
        "distilbert-base-uncased": "distilbert-base-uncased",
    }


def resolve_model_path(model_name: str) -> str:
    resolved_path = get_model_mapping().get(model_name, model_name)
    if resolved_path != model_name:
        logger.info("Resolved model: %s -> %s", model_name, resolved_path)
    return resolved_path


def get_target_modules_for_model(model_name: str) -> List[str]:
    modernbert_modules = ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"]
    bert_modules = [
        "attention.self.query",
        "attention.self.key",
        "attention.self.value",
        "attention.output.dense",
        "intermediate.dense",
        "output.dense",
    ]
    if model_name in ["modernbert-base", "answerdotai/ModernBERT-base"]:
        return modernbert_modules
    if model_name in ["mmbert-base", "jhu-clsp/mmBERT-base"]:
        return modernbert_modules
    if model_name in [
        "mmbert-32k",
        "mmbert-32k-yarn",
        "llm-semantic-router/mmbert-32k-yarn",
    ]:
        return modernbert_modules
    if model_name in ["bert-base-uncased", "roberta-base"]:
        return bert_modules
    supported = [
        "bert-base-uncased",
        "roberta-base",
        "modernbert-base",
        "answerdotai/ModernBERT-base",
        "mmbert-base",
        "jhu-clsp/mmBERT-base",
        "mmbert-32k",
        "mmbert-32k-yarn",
        "llm-semantic-router/mmbert-32k-yarn",
    ]
    raise ValueError(f"Unsupported model: {model_name}. Supported: {supported}")


def create_lora_config(
    model_name: str, rank: int = 8, alpha: int = 16, dropout: float = 0.1
) -> Dict:
    if not isinstance(rank, int) or rank <= 0:
        raise ValueError(f"LoRA rank must be a positive integer, got: {rank}")
    if not isinstance(alpha, (int, float)) or alpha <= 0:
        raise ValueError(f"LoRA alpha must be a positive number, got: {alpha}")
    if not isinstance(dropout, (int, float)) or not (0 <= dropout <= 1):
        raise ValueError(f"LoRA dropout must be between 0 and 1, got: {dropout}")
    return {
        "rank": rank,
        "alpha": alpha,
        "dropout": dropout,
        "target_modules": get_target_modules_for_model(model_name),
    }


def get_all_gpu_info() -> List[Dict]:
    if not torch.cuda.is_available():
        return []
    gpu_info: List[Dict] = []
    for gpu_id in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(gpu_id)
        total_memory = props.total_memory / 1024**3
        torch.cuda.set_device(gpu_id)
        allocated = torch.cuda.memory_allocated(gpu_id) / 1024**3
        reserved = torch.cuda.memory_reserved(gpu_id) / 1024**3
        gpu_info.append(
            {
                "id": gpu_id,
                "name": torch.cuda.get_device_name(gpu_id),
                "total_memory_gb": total_memory,
                "allocated_memory_gb": allocated,
                "reserved_memory_gb": reserved,
                "free_memory_gb": total_memory - reserved,
                "utilization_percent": (reserved / total_memory) * 100,
            }
        )
    return gpu_info


def find_free_gpu(min_free_memory_gb: float = 2.0) -> Optional[int]:
    gpu_info = get_all_gpu_info()
    if not gpu_info:
        return None
    gpu_info.sort(key=lambda x: x["free_memory_gb"], reverse=True)
    best = gpu_info[0]
    if best["free_memory_gb"] < min_free_memory_gb:
        return None
    return best["id"]


def set_gpu_device(
    gpu_id: Optional[int] = None, auto_select: bool = True
) -> Tuple[str, int]:
    if not torch.cuda.is_available():
        logger.warning("No CUDA available, using CPU")
        return "cpu", -1
    if gpu_id is not None:
        if gpu_id < 0 or gpu_id >= torch.cuda.device_count():
            raise ValueError(
                f"Invalid GPU ID {gpu_id}. Available GPUs: 0-{torch.cuda.device_count()-1}"
            )
        torch.cuda.set_device(gpu_id)
        return f"cuda:{gpu_id}", gpu_id
    if auto_select:
        best_gpu_id = find_free_gpu()
        if best_gpu_id is None:
            logger.warning("No suitable GPU found, using CPU")
            return "cpu", -1
        torch.cuda.set_device(best_gpu_id)
        os.environ["CUDA_VISIBLE_DEVICES"] = str(best_gpu_id)
        return f"cuda:{best_gpu_id}", best_gpu_id
    torch.cuda.set_device(0)
    return "cuda:0", 0


def clear_gpu_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def log_memory_usage(stage: str = "") -> None:
    stage_prefix = f"[{stage}] " if stage else ""
    if torch.cuda.is_available():
        logger.info(
            "%sGPU Memory - Allocated: %.2fGB, Reserved: %.2fGB",
            stage_prefix,
            torch.cuda.memory_allocated() / 1024**3,
            torch.cuda.memory_reserved() / 1024**3,
        )
        return
    try:
        import psutil

        vm = psutil.virtual_memory()
        logger.info(
            "%sSystem Memory - Used: %.2fGB, Available: %.2fGB",
            stage_prefix,
            vm.used / 1024**3,
            vm.available / 1024**3,
        )
    except Exception:
        logger.info("%sSystem Memory stats unavailable", stage_prefix)


def create_tokenizer_for_model(model_path: str, base_model_name: str = None):
    model_identifier = base_model_name or model_path
    if "roberta" in model_identifier.lower():
        return AutoTokenizer.from_pretrained(model_path, add_prefix_space=True)
    return AutoTokenizer.from_pretrained(model_path)


def create_lora_security_model(model_name: str, num_labels: int, lora_config: dict):
    # 这里同时完成你理解中的 (2)(3)(6)：
    # - (3) tokenizer：`AutoTokenizer.from_pretrained`
    # - (2) base 权重：`AutoModelForSequenceClassification.from_pretrained`
    # - (6) “模型结构”：先是 base seq-cls 结构，再用 PEFT 包装成 LoRA 结构
    tokenizer = create_tokenizer_for_model(model_name, model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )

    peft_config = LoraConfig(
        task_type=TaskType.SEQ_CLS,
        inference_mode=False,
        r=lora_config["rank"],
        lora_alpha=lora_config["alpha"],
        lora_dropout=lora_config["dropout"],
        target_modules=lora_config["target_modules"],
        bias="none",
        modules_to_save=["classifier"],
    )
    lora_model = get_peft_model(base_model, peft_config)
    for param in lora_model.parameters():
        if param.requires_grad:
            param.data = param.data.float()
    return lora_model, tokenizer


def create_full_security_model(model_path: str, num_labels: int):
    # 全参数微调：直接加载序列分类模型（含预训练权重）与 tokenizer。
    tokenizer = create_tokenizer_for_model(model_path, model_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(
        model_path,
        num_labels=num_labels,
        torch_dtype=torch.float32,
    )
    return model, tokenizer


def tokenize_security_data(data, tokenizer, max_length: int = 512):
    texts = [item["text"] for item in data]
    labels = [item["label"] for item in data]
    encodings = tokenizer(
        texts, truncation=True, padding=True, max_length=max_length, return_tensors="pt"
    )
    return Dataset.from_dict(
        {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": labels,
        }
    )


def compute_security_metrics(eval_pred):
    predictions, labels = eval_pred
    predictions = torch.argmax(torch.tensor(predictions), dim=1)
    accuracy = accuracy_score(labels, predictions)
    unique_labels = set(labels) if isinstance(labels, (list, tuple)) else set(list(labels))
    average = "binary" if len(unique_labels) <= 2 else "macro"
    precision, recall, f1, _ = precision_recall_fscore_support(
        labels, predictions, average=average, zero_division=0
    )
    return {
        "accuracy": accuracy,
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }

