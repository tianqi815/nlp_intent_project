"""
数据加载统一入口：复用 data/dataset_loader.py 的实现，减少重复逻辑。
"""

from typing import List, Tuple

from data.dataset_loader import (
    ID_TO_LABEL,
    LABEL_TO_ID,
    STAGE1_ID_TO_LABEL,
    STAGE1_LABEL_TO_ID,
    STAGE2_ID_TO_LABEL,
    STAGE2_LABEL_TO_ID,
    load_and_split_data as _load_and_split_data,
    load_raw_data as _load_raw_data,
    load_stage1_and_split_data as _load_stage1_and_split_data,
    load_stage1_test_data as _load_stage1_test_data,
    load_stage2_and_split_data as _load_stage2_and_split_data,
    load_stage2_test_data as _load_stage2_test_data,
    load_test_data as _load_test_data,
)


def load_raw_data(data_path: str = None, merge_hitl: bool = True) -> dict:
    return _load_raw_data(data_path=data_path, merge_hitl=merge_hitl)


def load_test_data(test_data_path: str) -> List[dict]:
    return _load_test_data(test_data_path)


def load_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    return _load_and_split_data(
        data_path=data_path,
        val_ratio=val_ratio,
        random_state=random_state,
    )


def load_stage1_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    return _load_stage1_and_split_data(
        data_path=data_path,
        val_ratio=val_ratio,
        random_state=random_state,
    )


def load_stage2_and_split_data(
    data_path: str = None,
    val_ratio: float = 0.2,
    random_state: int = 42,
) -> Tuple[List[dict], List[dict]]:
    return _load_stage2_and_split_data(
        data_path=data_path,
        val_ratio=val_ratio,
        random_state=random_state,
    )


def load_stage1_test_data(test_data_path: str) -> List[dict]:
    return _load_stage1_test_data(test_data_path)


def load_stage2_test_data(test_data_path: str) -> List[dict]:
    return _load_stage2_test_data(test_data_path)

