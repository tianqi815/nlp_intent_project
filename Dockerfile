# NLP 意图识别训练环境（与宿主机环境隔离）
FROM python:3.10-slim

WORKDIR /app

# 安装常用依赖（按需增删）
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# 先拷贝依赖文件，便于利用 Docker 缓存
COPY requirements.txt .

# 安装 Python 依赖（在容器内，不影响别人环境）
RUN pip install --no-cache-dir -r requirements.txt

# 拷贝项目代码（后续开发时可用挂载卷代替）
COPY . .

# 默认保持运行，方便你进入容器做训练
CMD ["bash"]
