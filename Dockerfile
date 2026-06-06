# 帧神AI (Zhensage AI) — Docker 部署镜像
# 基于 Python 3.11 轻量镜像

FROM python:3.11-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件并安装 Python 包
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 复制应用代码
COPY app.py .
COPY templates/ ./templates/
COPY dialogpt_model/ ./dialogpt_model/

# 暴露端口
EXPOSE 5000

# 启动应用（生产环境使用 gunicorn）
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--timeout", "120", "app:app"]
