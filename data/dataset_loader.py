"""
意图数据集加载与划分。
支持：
1) 直接三分类（project monitoring / purchasing record summary / general_response）
2) 二阶段分类：
   - 阶段1：in_scope vs general_response
   - 阶段2：project monitoring vs purchasing record summary
"""
import json
import os
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


def _get_hitl_collected_path() -> Path:
    return Path(__file__).resolve().parent / "hitl_collected.json"


FINAL_LABEL_ORDER = [
    "project monitoring",
    "purchasing record summary",
    "general_response",
]
FINAL_LABEL_TO_ID = {name: idx for idx, name in enumerate(FINAL_LABEL_ORDER)}
FINAL_ID_TO_LABEL = {str(v): k for k, v in FINAL_LABEL_TO_ID.items()}

STAGE1_LABEL_TO_ID = {"in_scope": 0, "general_response": 1}
STAGE1_ID_TO_LABEL = {"0": "in_scope", "1": "general_response"}

STAGE2_LABEL_TO_ID = {"project monitoring": 0, "purchasing record summary": 1}
STAGE2_ID_TO_LABEL = {"0": "project monitoring", "1": "purchasing record summary"}

# 对外默认映射保持为最终三分类标签
LABEL_TO_ID = FINAL_LABEL_TO_ID
ID_TO_LABEL = FINAL_ID_TO_LABEL

DEFAULT_EXCEL_PATH = Path(__file__).resolve().parent.parent / "Question-data.xlsx"
DEFAULT_TRAIN_JSON_PATH = Path(__file__).resolve().parent / "train_dataset.json"
EXCEL_SHEETS_IN_ORDER = ["project monitoring", "purchasing record summary"]


def _normalize_text(x: object) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in ("nan", "none"):
        return ""
    return s


def _extract_texts_from_sheet(df) -> List[str]:
    columns = list(df.columns)
    header_candidates: List[str] = []
    if len(columns) == 1:
        col_name = _normalize_text(columns[0])
        lower = col_name.lower()
        if col_name and lower not in ("text", "question", "query") and not col_name.startswith(("问题", "Unnamed")) and col_name != "回答":
            if ("?" in col_name) or ("？" in col_name) or (len(col_name) >= 12):
                header_candidates.append(col_name)

    candidate_cols = []
    for c in columns:
        name = _normalize_text(c)
        if not name:
            continue
        if name.startswith("Unnamed") or name == "回答":
            continue
        if name.startswith("问题"):
            candidate_cols.append(c)
    if not candidate_cols:
        for c in columns:
            name = _normalize_text(c)
            if not name or name.startswith("Unnamed") or name == "回答":
                continue
            candidate_cols.append(c)

    seen = set()
    out: List[str] = []
    for s in header_candidates:
        if s and s not in seen:
            seen.add(s)
            out.append(s)
    for c in candidate_cols:
        for v in df[c].tolist():
            s = _normalize_text(v)
            if not s or s in seen:
                continue
            seen.add(s)
            out.append(s)
    return out


def load_excel_dataset(excel_path: str | Path = None) -> dict:
    """
    从 Excel 读取数据。Excel 仍只包含两个业务类，未包含 general_response。
    """
    try:
        import pandas as pd
    except Exception as e:  # pragma: no cover
        raise RuntimeError("缺少 pandas，无法读取 Excel 数据集。请先安装 requirements.txt 依赖。") from e

    p = Path(excel_path) if excel_path else DEFAULT_EXCEL_PATH
    if not p.is_file():
        raise FileNotFoundError(f"Excel 数据集不存在: {p}")

    grouped = {name: [] for name in EXCEL_SHEETS_IN_ORDER}
    for sheet_name in EXCEL_SHEETS_IN_ORDER:
        df = pd.read_excel(p, sheet_name=sheet_name)
        grouped[sheet_name] = _extract_texts_from_sheet(df)
    return _grouped_json_to_raw(grouped, FINAL_LABEL_TO_ID)


def _infer_grouped_label_order(grouped: dict) -> List[str]:
    ordered: List[str] = []
    for name in FINAL_LABEL_ORDER:
        if name in grouped:
            ordered.append(name)
    for name in grouped.keys():
        if name not in ordered:
            ordered.append(name)
    return ordered


def _build_label_mapping(label_names: List[str]) -> Tuple[Dict[str, int], Dict[str, str]]:
    label_to_id = {name: idx for idx, name in enumerate(label_names)}
    id_to_label = {str(v): k for k, v in label_to_id.items()}
    return label_to_id, id_to_label


def _grouped_json_to_raw(grouped: dict, label_to_id: Dict[str, int]) -> dict:
    texts: List[str] = []
    labels: List[int] = []
    for label_name, label_id in label_to_id.items():
        values = grouped.get(label_name, [])
        if values is None:
            values = []
        if not isinstance(values, list):
            raise ValueError(f"训练集分组 JSON 中，{label_name} 的值必须是 list")
        for v in values:
            s = _normalize_text(v)
            if not s:
                continue
            texts.append(s)
            labels.append(label_id)
    return {"text": texts, "label": labels}


def _raw_to_sample_list(data_raw: dict) -> List[dict]:
    texts = data_raw["text"]
    labels = data_raw["label"]
    if len(texts) != len(labels):
        raise ValueError(
            f"text 与 label 数量不一致: text 有 {len(texts)} 条, label 有 {len(labels)} 条。"
            "请检查 JSON 文件，确保两者长度相同。"
        )
    return [{"text": texts[i], "label": labels[i]} for i in range(len(texts))]


def _load_json_or_default(data_path: str = None) -> tuple[dict, Dict[str, int], Dict[str, str]]:
    if data_path and os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    elif DEFAULT_TRAIN_JSON_PATH.is_file():
        with open(DEFAULT_TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    else:
        return load_excel_dataset(), dict(FINAL_LABEL_TO_ID), dict(FINAL_ID_TO_LABEL)

    if isinstance(loaded, dict) and "text" in loaded and "label" in loaded:
        return loaded, dict(FINAL_LABEL_TO_ID), dict(FINAL_ID_TO_LABEL)

    grouped = loaded if isinstance(loaded, dict) else {}
    label_order = _infer_grouped_label_order(grouped)
    label_to_id, id_to_label = _build_label_mapping(label_order)
    return _grouped_json_to_raw(grouped, label_to_id), label_to_id, id_to_label


def load_raw_data(data_path: str = None, merge_hitl: bool = True) -> dict:
    """
    加载最终标签数据（默认三分类），返回 {"text":[...], "label":[...]}。
    """
    data_raw, label_to_id, _ = _load_json_or_default(data_path)
    if not merge_hitl:
        return data_raw

    hitl_path = _get_hitl_collected_path()
    if hitl_path.is_file():
        with open(hitl_path, "r", encoding="utf-8") as f:
            loaded_hitl = json.load(f)
        if isinstance(loaded_hitl, dict) and loaded_hitl.get("text") and loaded_hitl.get("label") and len(loaded_hitl["text"]) == len(loaded_hitl["label"]):
            hitl = loaded_hitl
        else:
            hitl = _grouped_json_to_raw(loaded_hitl, label_to_id)
        if hitl.get("text") and hitl.get("label") and len(hitl["text"]) == len(hitl["label"]):
            data_raw = {"text": list(data_raw["text"]), "label": list(data_raw["label"])}
            data_raw["text"].extend(hitl["text"])
            data_raw["label"].extend(hitl["label"])
    return data_raw


def _safe_split(sample_data: List[dict], val_ratio: float, random_state: int) -> Tuple[List[dict], List[dict]]:
    labels = [x["label"] for x in sample_data]
    try:
        return train_test_split(
            sample_data,
            test_size=val_ratio,
            random_state=random_state,
            stratify=labels,
        )
    except ValueError:
        # 类别过少或分布不满足分层条件时，自动回退到非分层切分
        return train_test_split(
            sample_data,
            test_size=val_ratio,
            random_state=random_state,
            stratify=None,
        )


def load_test_data(test_data_path: str) -> List[dict]:
    if not test_data_path or not os.path.isfile(test_data_path):
        raise FileNotFoundError(f"测试集文件不存在: {test_data_path}")
    raw, _, _ = _load_json_or_default(test_data_path)
    return _raw_to_sample_list(raw)


def load_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    data_raw = load_raw_data(data_path)
    sample_data = _raw_to_sample_list(data_raw)
    return _safe_split(sample_data, val_ratio, random_state)


def _load_grouped_data(data_path: str = None) -> dict:
    if data_path and os.path.isfile(data_path):
        with open(data_path, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    elif DEFAULT_TRAIN_JSON_PATH.is_file():
        with open(DEFAULT_TRAIN_JSON_PATH, "r", encoding="utf-8") as f:
            loaded = json.load(f)
    else:
        raise FileNotFoundError("未找到 JSON 分组训练集，请提供 data_path")
    if not isinstance(loaded, dict):
        raise ValueError("分组数据必须是 dict")
    if "text" in loaded and "label" in loaded:
        raise ValueError("二阶段训练请使用分组 JSON 格式，而非 raw 格式")
    return loaded


def load_stage1_raw_data(data_path: str = None, merge_hitl: bool = True) -> dict:
    grouped = _load_grouped_data(data_path)
    in_scope_texts: List[str] = []
    for k in ("project monitoring", "purchasing record summary"):
        in_scope_texts.extend(grouped.get(k, []) or [])
    stage1_grouped = {
        "in_scope": in_scope_texts,
        "general_response": grouped.get("general_response", []) or [],
    }
    data_raw = _grouped_json_to_raw(stage1_grouped, STAGE1_LABEL_TO_ID)
    if not merge_hitl:
        return data_raw

    hitl_path = _get_hitl_collected_path()
    if hitl_path.is_file():
        with open(hitl_path, "r", encoding="utf-8") as f:
            hitl_grouped = json.load(f)
        hitl_stage1 = {
            "in_scope": (hitl_grouped.get("project monitoring", []) or []) + (hitl_grouped.get("purchasing record summary", []) or []),
            "general_response": hitl_grouped.get("general_response", []) or [],
        }
        hitl_raw = _grouped_json_to_raw(hitl_stage1, STAGE1_LABEL_TO_ID)
        data_raw = {"text": list(data_raw["text"]), "label": list(data_raw["label"])}
        data_raw["text"].extend(hitl_raw["text"])
        data_raw["label"].extend(hitl_raw["label"])
    return data_raw


def load_stage2_raw_data(data_path: str = None, merge_hitl: bool = True) -> dict:
    grouped = _load_grouped_data(data_path)
    stage2_grouped = {
        "project monitoring": grouped.get("project monitoring", []) or [],
        "purchasing record summary": grouped.get("purchasing record summary", []) or [],
    }
    data_raw = _grouped_json_to_raw(stage2_grouped, STAGE2_LABEL_TO_ID)
    if not merge_hitl:
        return data_raw

    hitl_path = _get_hitl_collected_path()
    if hitl_path.is_file():
        with open(hitl_path, "r", encoding="utf-8") as f:
            hitl_grouped = json.load(f)
        hitl_stage2 = {
            "project monitoring": hitl_grouped.get("project monitoring", []) or [],
            "purchasing record summary": hitl_grouped.get("purchasing record summary", []) or [],
        }
        hitl_raw = _grouped_json_to_raw(hitl_stage2, STAGE2_LABEL_TO_ID)
        data_raw = {"text": list(data_raw["text"]), "label": list(data_raw["label"])}
        data_raw["text"].extend(hitl_raw["text"])
        data_raw["label"].extend(hitl_raw["label"])
    return data_raw


def load_stage1_test_data(test_data_path: str) -> List[dict]:
    grouped = _load_grouped_data(test_data_path)
    stage1_grouped = {
        "in_scope": (grouped.get("project monitoring", []) or []) + (grouped.get("purchasing record summary", []) or []),
        "general_response": grouped.get("general_response", []) or [],
    }
    return _raw_to_sample_list(_grouped_json_to_raw(stage1_grouped, STAGE1_LABEL_TO_ID))


def load_stage2_test_data(test_data_path: str) -> List[dict]:
    grouped = _load_grouped_data(test_data_path)
    stage2_grouped = {
        "project monitoring": grouped.get("project monitoring", []) or [],
        "purchasing record summary": grouped.get("purchasing record summary", []) or [],
    }
    return _raw_to_sample_list(_grouped_json_to_raw(stage2_grouped, STAGE2_LABEL_TO_ID))


def load_stage1_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    raw = load_stage1_raw_data(data_path=data_path)
    sample_data = _raw_to_sample_list(raw)
    return _safe_split(sample_data, val_ratio, random_state)


def load_stage2_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    raw = load_stage2_raw_data(data_path=data_path)
    sample_data = _raw_to_sample_list(raw)
    return _safe_split(sample_data, val_ratio, random_state)
