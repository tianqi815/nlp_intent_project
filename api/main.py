"""
意图识别模型 FastAPI 服务：加载训练好的模型，提供预测接口；Web 测试页由静态文件提供。
支持 HITL 反馈：用户标记错误预测并提交正确答案，追加保存为可训练数据。
"""
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from src.predict import load_model_and_tokenizer, predict_with_loaded

DEFAULT_MODEL_PATH = os.environ.get("MODEL_PATH", "intent_two_stage_modernbert")

_model = None  # 兼容单模型
_tokenizer = None
_id_to_label = None

_stage1_model = None
_stage1_tokenizer = None
_stage1_id_to_label = None
_stage2_model = None
_stage2_tokenizer = None
_stage2_id_to_label = None
_is_two_stage = False
_final_id_to_label = {"0": "project monitoring", "1": "purchasing record summary", "2": "general_response"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _tokenizer, _id_to_label
    global _stage1_model, _stage1_tokenizer, _stage1_id_to_label
    global _stage2_model, _stage2_tokenizer, _stage2_id_to_label
    global _is_two_stage, _final_id_to_label
    model_path = os.environ.get("MODEL_PATH", DEFAULT_MODEL_PATH)
    if not os.path.isdir(model_path):
        raise RuntimeError(
            f"模型目录不存在: {model_path}，请设置环境变量 MODEL_PATH 或先训练并保存模型到该目录。"
        )
    # 打印模型路径与权重更新时间，便于确认是否加载了最新训练结果（训练后需重启 API）
    config_path = Path(model_path) / "config.json"
    if config_path.is_file():
        mtime = config_path.stat().st_mtime
        from datetime import datetime
        ts = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[API] 加载模型: {os.path.abspath(model_path)} (权重时间: {ts})")
    stage1_dir = Path(model_path) / "stage1_model"
    stage2_dir = Path(model_path) / "stage2_model"
    if stage1_dir.is_dir() and stage2_dir.is_dir():
        _is_two_stage = True
        _stage1_model, _stage1_tokenizer, _stage1_id_to_label = load_model_and_tokenizer(str(stage1_dir))
        _stage2_model, _stage2_tokenizer, _stage2_id_to_label = load_model_and_tokenizer(str(stage2_dir))
        _stage1_model.eval()
        _stage2_model.eval()
        _final_id_to_label = {"0": "project monitoring", "1": "purchasing record summary", "2": "general_response"}
        print(f"[API] 以二阶段模式启动: {os.path.abspath(model_path)}")
    else:
        _is_two_stage = False
        _model, _tokenizer, _id_to_label = load_model_and_tokenizer(model_path)
        _model.eval()
        _final_id_to_label = dict(_id_to_label)
        print(f"[API] 以单模型模式启动: {os.path.abspath(model_path)}")
    yield
    _model = None
    _tokenizer = None
    _id_to_label = None
    _stage1_model = None
    _stage1_tokenizer = None
    _stage1_id_to_label = None
    _stage2_model = None
    _stage2_tokenizer = None
    _stage2_id_to_label = None
    _is_two_stage = False


app = FastAPI(
    title="意图识别模型测试平台",
    description="输入问题文本，获取模型预测的意图类别与置信度。供 Open WebUI / Sanfield Chatbot 等应用做智能路由。",
    lifespan=lifespan,
)

# 允许其他应用（如 Open WebUI、前端）跨域调用；生产环境建议用环境变量限制 origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=os.environ.get("CORS_ORIGINS", "*").split(",") if os.environ.get("CORS_ORIGINS") else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- 请求/响应模型 ----------
class PredictSingleRequest(BaseModel):
    text: str = Field(..., min_length=1, description="用户输入的问题文本")


class PredictBatchRequest(BaseModel):
    texts: List[str] = Field(..., min_length=1, description="多条问题文本")


class PredictItem(BaseModel):
    text: str
    label: str
    label_id: int
    confidence: float
    stage1_label: Optional[str] = None
    stage1_confidence: Optional[float] = None


class IntentResponse(BaseModel):
    """供路由层使用：仅返回意图与置信度，便于其他应用选择下游模型。"""
    intent: str = Field(..., description="意图标签，如 project monitoring / purchasing record summary / general_response")
    label_id: int = Field(..., description="标签 id（从 0 开始递增，对应 label_mapping.json）")
    confidence: float = Field(..., description="置信度 0~1")
    suggested_route: Optional[str] = Field(None, description="建议调用的模型/路由名，由 ROUTE_0/ROUTE_1/ROUTE_2 环境变量配置")
    stage1_label: Optional[str] = Field(None, description="调试字段：阶段1结果（in_scope/general_response）")
    stage1_confidence: Optional[float] = Field(None, description="调试字段：阶段1置信度")


class FeedbackRequest(BaseModel):
    """用户对预测结果的反馈；当 is_correct=False 时需提供 correct_label_id。"""
    text: str = Field(..., min_length=1, description="用户输入的问题文本")
    predicted_label: str = Field(..., description="模型预测的标签文本")
    predicted_label_id: int = Field(..., ge=0, description="模型预测的标签 id（从 0 开始）")
    is_correct: bool = Field(..., description="用户认为预测是否正确")
    correct_label_id: Optional[int] = Field(None, ge=0, description="用户给出的正确答案 id，仅当 is_correct=False 时必填")

    @model_validator(mode="after")
    def require_correct_label_when_wrong(self):
        if not self.is_correct and self.correct_label_id is None:
            raise ValueError("当 is_correct 为 false 时，必须提供 correct_label_id")
        return self


class FeedbackResponse(BaseModel):
    ok: bool = True
    message: str = "已记录，将用于后续训练"


def _get_hitl_data_path() -> Path:
    """HITL 收集数据文件路径，可由环境变量 HITL_DATA_PATH 覆盖。"""
    env_path = os.environ.get("HITL_DATA_PATH")
    if env_path:
        return Path(env_path)
    return Path(__file__).resolve().parent.parent / "data" / "hitl_collected.json"


def _append_hitl_sample(text: str, correct_label_id: int, id_to_label: dict) -> None:
    """将一条错误样本追加到 HITL JSON 文件（按标签分组格式）。写文件时加锁避免并发覆盖。"""
    path = _get_hitl_data_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        init_payload = {v: [] for _, v in sorted(((int(k), v) for k, v in id_to_label.items()), key=lambda x: x[0])}
        path.write_text(json.dumps(init_payload, ensure_ascii=False, indent=2), encoding="utf-8")
    try:
        import fcntl
        with open(path, "r+", encoding="utf-8") as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                raw = f.read()
                data = json.loads(raw) if raw.strip() else {}
                # 兼容旧格式 {"text":[...],"label":[...]}，读到后直接迁移成分组格式
                if isinstance(data, dict) and "text" in data and "label" in data and isinstance(data["text"], list) and isinstance(data["label"], list):
                    migrated = {v: [] for _, v in sorted(((int(k), v) for k, v in id_to_label.items()), key=lambda x: x[0])}
                    for t, lid in zip(data["text"], data["label"]):
                        name = id_to_label.get(str(int(lid)))
                        if name:
                            migrated.setdefault(name, []).append(t)
                    data = migrated
                label_name = id_to_label.get(str(correct_label_id))
                if not label_name:
                    raise ValueError(f"未知的 correct_label_id: {correct_label_id}")
                data.setdefault(label_name, [])
                data[label_name].append(text)
                payload = json.dumps(data, ensure_ascii=False, indent=2)
                f.seek(0)
                f.write(payload)
                f.truncate()
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)
    except ImportError:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "text" in data and "label" in data and isinstance(data["text"], list) and isinstance(data["label"], list):
            migrated = {v: [] for _, v in sorted(((int(k), v) for k, v in id_to_label.items()), key=lambda x: x[0])}
            for t, lid in zip(data["text"], data["label"]):
                name = id_to_label.get(str(int(lid)))
                if name:
                    migrated.setdefault(name, []).append(t)
            data = migrated
        label_name = id_to_label.get(str(correct_label_id))
        if not label_name:
            raise ValueError(f"未知的 correct_label_id: {correct_label_id}")
        data.setdefault(label_name, [])
        data[label_name].append(text)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)


# ---------- 接口 ----------
_STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/health")
async def health():
    """健康检查：供负载均衡/编排调用；模型已加载时返回 200。"""
    if _is_two_stage:
        if _stage1_model is None or _stage2_model is None:
            raise HTTPException(status_code=503, detail="模型未加载")
    elif _model is None or _tokenizer is None:
        raise HTTPException(status_code=503, detail="模型未加载")
    return {"status": "ok", "model_loaded": True, "mode": "two_stage" if _is_two_stage else "single"}


@app.get("/", response_class=HTMLResponse)
async def index():
    """返回 Web 测试页（静态 HTML）。"""
    index_file = _STATIC_DIR / "index.html"
    if index_file.is_file():
        return FileResponse(index_file)
    raise HTTPException(status_code=404, detail="未找到 index.html")


def _get_suggested_route(label_id: int) -> Optional[str]:
    """从环境变量 ROUTE_0、ROUTE_1、ROUTE_2... 读取建议路由名，未配置则返回 None。"""
    key = f"ROUTE_{label_id}"
    return os.environ.get(key)


@app.get("/labels")
async def get_labels():
    """返回最终可用标签映射，不暴露内部阶段标签。"""
    if not _final_id_to_label:
        raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
    return {"id_to_label": _final_id_to_label}


def _intent_response_from_text(text: str) -> IntentResponse:
    items = _do_predict([text])
    if not items:
        raise HTTPException(status_code=500, detail="预测结果为空")
    one = items[0]
    label_id = one["label_id"]
    return IntentResponse(
        intent=one["label"],
        label_id=label_id,
        confidence=round(one["confidence"], 6),
        suggested_route=_get_suggested_route(label_id),
        stage1_label=one.get("stage1_label"),
        stage1_confidence=round(one["stage1_confidence"], 6) if one.get("stage1_confidence") is not None else None,
    )


@app.get("/intent")
async def get_intent_query(text: Optional[str] = Query(None, description="用户问题；不传则返回本接口使用说明")):
    """GET /intent：不传 text 时返回使用说明，传 text 时返回意图预测结果。"""
    if not text or not text.strip():
        doc = {
            "description": "NLP 意图识别接口：根据用户输入文本预测意图类别（project monitoring / purchasing record summary / general_response），供路由到不同下游模型。",
            "usage": {
                "GET": "请求 GET /intent?text=你的问题，例如：GET /intent?text=玻璃门损坏怎么报修",
                "POST": "请求 POST /intent，请求体为 JSON：{\"text\": \"用户问题\"}",
            },
            "request_body_example": {"text": "用户输入的问题文本"},
            "response_format": {
                "intent": "意图标签，如 project monitoring、purchasing record summary",
                "label_id": "0=project monitoring，1=purchasing record summary，2=general_response",
                "confidence": "置信度 0~1",
                "suggested_route": "建议调用的模型/路由名（可选，由服务端环境变量配置）",
                "stage1_label": "调试字段，可选：in_scope/general_response",
                "stage1_confidence": "调试字段，可选：阶段1置信度",
            },
            "response_example": {
                "intent": "purchasing record summary",
                "label_id": 1,
                "confidence": 0.92,
                "suggested_route": None,
                "stage1_label": "in_scope",
                "stage1_confidence": 0.98,
            },
            "curl_examples": [
                "curl -X GET 'http://172.31.61.220:8001/intent?text=你好'",
                "curl -X POST http://172.31.61.220:8001/intent -H 'Content-Type: application/json' -d '{\"text\": \"采购订单查询\"}'",
            ],
        }
        return Response(
            content=json.dumps(doc, ensure_ascii=False, indent=2),
            media_type="application/json; charset=utf-8",
        )
    return _intent_response_from_text(text.strip())


@app.post("/intent", response_model=IntentResponse)
async def get_intent(req: PredictSingleRequest):
    """POST 方式：请求体 {"text": "用户问题"}，供其他应用/路由层调用。"""
    return _intent_response_from_text(req.text)


@app.post("/predict", response_model=List[PredictItem])
async def predict_single(req: PredictSingleRequest):
    """单条预测：请求体为 {"text": "用户问题"}。"""
    return _do_predict([req.text])


@app.post("/predict/batch", response_model=List[PredictItem])
async def predict_batch(req: PredictBatchRequest):
    """批量预测：请求体为 {"texts": ["问题1", "问题2", ...]}。"""
    return _do_predict(req.texts)


@app.post("/feedback", response_model=FeedbackResponse)
async def feedback(req: FeedbackRequest):
    """HITL 反馈：用户标记预测正确或错误；仅当 is_correct=False 时将 (text, correct_label_id) 追加到 HITL 数据文件。"""
    if not req.is_correct and req.correct_label_id is not None:
        if _id_to_label is None:
            raise HTTPException(status_code=503, detail="模型未加载，请稍后重试")
        max_id = len(_id_to_label) - 1
        if req.correct_label_id < 0 or req.correct_label_id > max_id:
            raise HTTPException(status_code=400, detail=f"correct_label_id 超出范围：0~{max_id}")
        _append_hitl_sample(req.text, req.correct_label_id, _id_to_label)
        return FeedbackResponse(ok=True, message="已记录，将用于后续训练")
    return FeedbackResponse(ok=True, message="感谢反馈")


def _do_predict(texts: List[str]) -> List[dict]:
    if not texts:
        return []
    max_length = int(os.environ.get("MAX_LENGTH", "512"))
    if _is_two_stage:
        if _stage1_model is None or _stage2_model is None or _stage1_id_to_label is None or _stage2_id_to_label is None:
            raise HTTPException(status_code=503, detail="二阶段模型未加载，请检查服务启动日志。")
        out: List[dict] = []
        for text in texts:
            s1 = predict_with_loaded(
                _stage1_model, _stage1_tokenizer, _stage1_id_to_label, [text], max_length=max_length
            )[0]
            s1_label = s1["label"]
            s1_conf = s1["confidence"]
            if s1_label == "general_response":
                out.append(
                    {
                        "text": text,
                        "label": "general_response",
                        "label_id": 2,
                        "confidence": s1_conf,
                        "stage1_label": s1_label,
                        "stage1_confidence": s1_conf,
                    }
                )
                continue

            s2 = predict_with_loaded(
                _stage2_model, _stage2_tokenizer, _stage2_id_to_label, [text], max_length=max_length
            )[0]
            out.append(
                {
                    "text": text,
                    "label": s2["label"],
                    "label_id": s2["label_id"],  # stage2 恰好与最终 0/1 对齐
                    "confidence": s2["confidence"],
                    "stage1_label": s1_label,
                    "stage1_confidence": s1_conf,
                }
            )
        return out

    if _model is None or _tokenizer is None or _id_to_label is None:
        raise HTTPException(status_code=503, detail="模型未加载，请检查服务启动日志。")
    return predict_with_loaded(_model, _tokenizer, _id_to_label, texts, max_length=max_length)


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8001"))
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=port,
        reload=False,
    )
