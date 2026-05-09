# 法律合同智能问答系统（RAG）
基于 DeepSeek 大模型 + 本地检索的合同PDF问答工具，不幻觉、不联网、不上传文档、可本地部署。

## 功能
- 上传合同 PDF，基于合同内容精准回答
- 不依赖外网向量模型，国内可流畅运行
- 前后端分离，网页聊天界面演示
- 支持 Windows / Mac 直接运行

## 技术栈
- Python
- FastAPI
- DeepSeek Chat
- 本地 TF-IDF 检索（无外网依赖）
- HTML 前端界面

## 快速启动

### 1. 安装依赖
pip install -r requirements.txt

### 2. 配置 API Key
复制 `.env.example` 为 `.env`，填入你的 DeepSeek API Key

### 3. 替换 PDF
放入你的合同文档，命名为 `law.pdf`

### 4. 启动接口
python api.py

### 5. 打开前端
双击 `index.html` 开始对话

## 注意
本项目使用本地检索方案，不依赖 HuggingFace，不下载模型，国内可直接运行。