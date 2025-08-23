
# DOCKER_DOTS_OCR 服务配置指南

## 1. 环境依赖

1. **NVIDIA Docker**（需支持GPU加速）
2. **Python 3.8+**（用于下载模型权重）
3. **依赖包**
	```bash
	pip install modelscope huggingface_hub
	```

## 2. 下载模型权重

在本目录下运行：
```bash
python download_model.py
```
模型会下载到 `weights/DotsOCR/` 目录。

## 3. 配置 Docker

### 主要文件说明
- `docker-compose.yml`：服务编排，已配置GPU、端口、挂载权重目录等。
- `Dockerfile`：如需自定义镜像可参考。

### docker-compose.yml需要注意的配置
- `shm_size: 2g`：保证NCCL通信，建议不低于2g。
- `NVIDIA_VISIBLE_DEVICES=0,1`：指定可用GPU编号。
- `volumes`：权重目录需与本地一致。

## 4. 启动与管理服务

### 拉取镜像
```bash
docker compose pull
```

### 启动服务
```bash
docker compose up -d
```

### 查看日志
```bash
docker compose logs -f
```

### 停止服务
```bash
docker compose down
```