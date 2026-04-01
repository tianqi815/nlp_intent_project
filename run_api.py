#!/usr/bin/env python3
"""
启动 FastAPI 服务与 Web 测试页：在项目根目录执行  python run_api.py  。
可通过环境变量 MODEL_PATH、PORT、MAX_LENGTH 配置。
"""
import os
import uvicorn

if __name__ == "__main__":
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=int(os.environ.get("PORT", "8001")),
        reload=False,
    )
