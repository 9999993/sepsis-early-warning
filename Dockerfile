# 使用轻量级Python镜像
FROM python:3.9-slim

# 设置工作目录
WORKDIR /app

# 安装系统依赖
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 复制依赖文件
COPY requirements.txt .

# 安装Python依赖
RUN pip install --no-cache-dir -r requirements.txt

# 复制项目文件
COPY . .

# 创建必要的目录
RUN mkdir -p data models logs

# 暴露端口
EXPOSE 8050

# 默认命令：训练模型并启动服务
CMD ["sh", "-c", "python data_generator.py && python train.py && python app.py"]