#!/bin/bash

# BioChemInsight 启动脚本
# 用于同时启动前端和后端服务

echo "🧪 启动 BioChemInsight 服务..."

# 检查是否安装了必要的依赖
if ! command -v node &> /dev/null; then
    echo "❌ Node.js 未安装。请先安装 Node.js 18+"
    exit 1
fi

if ! command -v uvicorn &> /dev/null; then
    echo "❌ uvicorn 未安装。请运行: pip install fastapi uvicorn"
    exit 1
fi

# 检查 constants.py 是否存在
if [ ! -f "constants.py" ]; then
    echo "⚠️  constants.py 文件不存在，正在复制示例文件..."
    if [ -f "constants_example.py" ]; then
        cp constants_example.py constants.py
        echo "✅ 已创建 constants.py，请编辑此文件配置您的API密钥和设置"
    else
        echo "❌ constants_example.py 文件不存在"
        exit 1
    fi
fi

# 安装前端依赖
echo "📦 安装前端依赖..."
cd frontend/ui
if [ ! -d "node_modules" ]; then
    npm install
fi
cd ../..

# 启动后端服务（后台运行）
echo "🚀 启动后端服务 (端口 8000)..."
cd frontend/backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
cd ../..

# 等待后端启动
sleep 3

# 启动前端服务（后台运行）
echo "🎨 启动前端服务 (端口 5173)..."
cd frontend/ui
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "✅ 服务启动完成！"
echo "🌐 前端地址: http://localhost:5173"
echo "🔧 后端 API: http://localhost:8000"
echo "📚 API 文档: http://localhost:8000/docs"
echo ""
echo "按 Ctrl+C 停止所有服务"

# 等待用户中断
wait_for_interrupt() {
    while true; do
        sleep 1
    done
}

# 清理函数
cleanup() {
    echo ""
    echo "🛑 正在停止服务..."
    kill $BACKEND_PID 2>/dev/null
    kill $FRONTEND_PID 2>/dev/null
    echo "✅ 服务已停止"
    exit 0
}

# 设置信号处理
trap cleanup SIGINT SIGTERM

# 等待中断
wait_for_interrupt