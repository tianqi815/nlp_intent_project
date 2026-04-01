# NLP 意图识别项目

面向智能客服 / Open WebUI 等场景的**中文意图路由**：将用户问题分类为业务意图（如工程监控、采购记录）或通用回复，便于下游调用不同模型或工具。

---

## 项目做什么

- **输入**：用户自然语言问题。
- **输出**：意图标签、置信度，以及（二阶段模式下）阶段 1 的调试信息；可配合环境变量映射到具体下游路由名。
- **典型标签**：`project monitoring`、`purchasing record summary`、`general_response`（与 `data/dataset_loader.py` 中定义一致）。

---

## 核心亮点：双阶段 / 双模型训练框架

传统**单模型多分类**时，细粒度业务类与「泛化/闲聊」类混在同一 softmax 中，容易互相拉扯、边界模糊。

本项目默认采用**二阶段、两套 NLP 分类模型**（同一基座可独立微调）：

| 阶段 | 任务 | 作用 |
|------|------|------|
| **Stage 1** | `in_scope` vs `general_response` | 先区分是否落在业务域，避免细类与闲聊在同一决策边界竞争。 |
| **Stage 2** | `project monitoring` vs `purchasing record summary` | 仅在 Stage 1 判为业务域时，再在两类间细分。 |

**推理链路**：Stage 1 → 若为 `general_response` 则直接输出；否则进入 Stage 2 得到最终业务标签。这样用**任务分解**降低多分类干扰，与「一个三分类头」相比通常更稳、更可解释。

可选 **`--single-stage`** 保留**单模型三分类**流程，便于对比或与旧管线兼容。

技术栈要点：**ModernBERT**（可配置）、**HuggingFace Transformers + Trainer**、可选 **LoRA（PEFT）**、训练指标含 F1 等（见 `src/model.py`）。

---

## 环境要求与安装

### 建议环境

- **Python**：3.10+（推荐 3.10 或 3.11）。
- **GPU**：训练与推理建议使用 NVIDIA GPU + CUDA；无 GPU 时训练脚本会回退 CPU（较慢，并自动调整部分训练参数）。
- **操作系统**：Windows / Linux 均可；Linux 下 HITL 反馈写文件使用文件锁，Windows 下为无 `fcntl` 的兼容写入路径。

### 安装步骤

在项目根目录 `nlp_intent_project` 下执行：

```bash
# 建议使用虚拟环境
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
# source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

首次运行会从 Hugging Face 拉取基座模型（如 `modernbert-base`），请保证网络可访问模型仓库或提前配置镜像 / 离线缓存。

---

## 训练：如何启动 `src/train.py`

**务必在项目根目录执行**，以便 `data`、`src` 等包路径正确。

### 推荐方式（模块入口）

```bash
python -m src.train
```

等价于直接运行脚本（根目录下）：

```bash
python src/train.py
```

### 默认行为

- **不加 `--single-stage`**：执行**二阶段训练**，输出目录默认为 `intent_two_stage_modernbert`，内含：
  - `stage1_model/`、`stage2_model/`
  - `pipeline_config.json`（管线说明）
- **`--single-stage`**：只训练**一个三分类模型**，`--model-path` 指向该单模型目录。

### 常用参数

| 参数 | 说明 |
|------|------|
| `--data` | 训练数据 JSON 路径；不传则按 `data/dataset_loader.py` 逻辑：优先 `data/train_dataset.json`，否则尝试 `Question-data.xlsx`。 |
| `--model-name` | 基座模型名或本地目录，默认 `modernbert-base`。 |
| `--model-path` | 模型保存根目录，默认 `intent_two_stage_modernbert`。 |
| `--epochs` / `--batch-size` / `--learning-rate` | 覆盖 `src/config.py` 中的默认训练超参。 |
| `--lora` / `--no-lora` | 启用 LoRA 或强制全参数微调；不设则遵循 `src/config.py` 的 `USE_LORA`。 |
| `--gpu-id` | 指定 GPU 编号。 |
| `--single-stage` | 单阶段三分类训练。 |
| `--keep-old-output` | 不覆盖已有输出目录（默认会归档旧结果到 `_training_archives/`）。 |
| `--keep-last-archives N` | 归档保留份数，默认 `2`。 |

### 示例

```bash
# 二阶段训练 + LoRA + 指定数据与输出目录
python -m src.train --data data/train_dataset.json --model-path ./my_two_stage --lora --epochs 3

# 单模型三分类（兼容旧流程）
python -m src.train --single-stage --model-path ./my_single_model --lora
```

训练结束会在各模型目录写入 `label_mapping.json`、`config.json`、权重与 tokenizer；二阶段根目录另有 `pipeline_config.json`。

---

## API：启动、调试与对外暴露

### 启动服务

在项目根目录：

```bash
python run_api.py
```

默认监听 **`0.0.0.0:8001`**（全网卡可访问，便于局域网调试；生产请配合防火墙与反向代理）。

### 环境变量

| 变量 | 含义 | 默认 |
|------|------|------|
| `MODEL_PATH` | 训练产物目录：二阶段为**父目录**（内含 `stage1_model` 与 `stage2_model`），单模型为**该模型目录** | `intent_two_stage_modernbert` |
| `PORT` | 服务端口 | `8001` |
| `MAX_LENGTH` | 推理最大序列长度 | `512` |
| `CORS_ORIGINS` | 允许的跨域来源，逗号分隔；不设则宽松允许（生产建议收紧） | `*` |
| `ROUTE_0`、`ROUTE_1`、`ROUTE_2` | 与 `label_id` 对应的下游路由名（可选） | 无 |
| `HITL_DATA_PATH` | HITL 反馈 JSON 路径（可选） | `data/hitl_collected.json` |

修改 `MODEL_PATH` 或重新训练后需**重启 API** 才能加载新权重。

### 调试方式

- 浏览器打开 **`http://<主机>:<端口>/`**：静态 Web 测试页（若存在 `api/static/index.html`）。
- **Swagger**：FastAPI 自动生成 **`/docs`**（交互式接口文档）。
- **健康检查**：`GET /health`。
- **意图接口**：`GET /intent?text=...` 或 `POST /intent`，请求体 `{"text":"..."}`。
- **批量预测**：`POST /predict/batch`，请求体 `{"texts":["...", "..."]}`。

---

## 数据集：格式、扩展与编辑

### 数据来源优先级（训练加载）

1. 若 `--data` 指向**存在的 JSON 文件**，则使用该文件。
2. 否则若存在 **`data/train_dataset.json`**，则使用它。
3. 否则若存在 **`Question-data.xlsx`**（项目根目录），则按 Excel 多 Sheet 读取（见下）。
4. 若以上皆无，会报错或需自备 JSON。

### 二阶段训练所需的 JSON（分组格式）

二阶段管线要求 **按意图名称分组的字典**，键为标签名，值为该类的文本列表，例如：

```json
{
  "project monitoring": ["示例问题1", "示例问题2"],
  "purchasing record summary": ["示例问题3"],
  "general_response": ["你好", "谢谢"]
}
```

**不要**使用仅含 `"text"` / `"label"` 数组的「扁平 raw」格式作为二阶段唯一数据源（加载器会校验）；单阶段三分类仍可使用 raw 格式（见下）。

扩展数据：在对应键下**追加字符串**即可；新增类别需同步修改 `data/dataset_loader.py` 中的标签常量与映射逻辑，并调整 API 中的最终标签约定。

### 单阶段训练支持的 JSON

- **Raw 格式**：`{"text": ["..."], "label": [0,1,2,...]}`，标签 id 与 `FINAL_LABEL_ORDER` 顺序一致。
- **分组格式**：与上文相同，加载器会推断标签顺序并转为样本列表。

### Excel（`Question-data.xlsx`）

- 两个 Sheet，名称须为 **`project monitoring`**、**`purchasing record summary`**（顺序见代码中的 `EXCEL_SHEETS_IN_ORDER`）。
- Excel **不含** `general_response`；该类需通过 JSON 或后续合并提供。

### HITL 与人工纠错

- **单模型部署**时，接口 **`POST /feedback`** 可将误判样本以正确 `label_id` 追加到 HITL 文件（默认 `data/hitl_collected.json`）。
- **二阶段部署**时，若 `/feedback` 返回模型未加载类错误，可直接**编辑或合并** `data/hitl_collected.json`（分组格式，键为三分类标签名），再重新训练；训练时 **`load_raw_data` 等默认会 merge HITL**。旧版 `text`/`label` 列表也会被加载器迁移为分组格式。

---

## 目录结构（简要）

- `src/train.py`：训练入口（单阶段 / 二阶段）。
- `src/config.py`：默认训练超参、LoRA 开关。
- `src/model.py`：模型构建、指标、GPU 与 tokenizer。
- `src/predict.py`：推理加载与预测。
- `data/dataset_loader.py`：数据集格式、划分、Stage1/Stage2 派生逻辑。
- `api/main.py`：FastAPI 应用与路由。
- `run_api.py`：Uvicorn 启动入口。

---

## 许可证与致谢

使用 **Transformers、PEFT、FastAPI** 等开源库；基座模型版权与许可以各模型卡为准。

若需将本文档中的默认主机、端口或内网 IP 改为你的部署地址，请直接替换文档或环境配置中的示例 URL。
