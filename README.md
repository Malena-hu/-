# 校园智能挂失系统

一个面向校园失物招领场景的多模态 RAG 原型系统。系统维护“失物区”和“招领区”两套向量库，支持用户上传丢失物品或拾取物品，自动进行图文特征提取、向量召回、候选匹配、DeepSeek 智能裁决和站内通知闭环。

## 功能概览

- 用户登录/注册：支持普通用户和管理员身份。
- 失物登记：用户上传丢失物品图片和描述，系统写入失物区并检索招领候选。
- 招领登记：用户或管理员上传拾取物品，系统写入招领区并后台匹配失主。
- 图文识别：调用千问视觉模型提取物品 JSON 元数据、颜色、材质、形态、标识、磨损等信息。
- 向量检索：使用本地图文向量模型和 ChromaDB 维护失物/招领向量库。
- DeepSeek 精排：对召回候选进行智能判断，决定是否发送单候选或多候选通知。
- 通知闭环：右上角通知中心展示待办，用户确认“已取回”后自动更新失物和招领记录状态。
- 管理员维护：支持批量删除、地点索引维护、同义词词典查看、审计日志和闭环记录导出。
- 软搜索：搜索词会结合本地同义词词典，必要时可通过 DeepSeek 扩展同义词。

## 技术栈

- Python 3.11
- Streamlit
- ChromaDB
- PyTorch
- Transformers / Sentence Transformers
- DashScope / 千问视觉模型
- DeepSeek Chat API
- Pillow / NumPy

## 目录结构

```text
myproject/
├── app.py                 # Streamlit 用户端和管理员端界面
├── workflow.py            # 业务流程、通知闭环、日志和状态管理
├── vector_db.py           # 图文向量库封装、检索、去重和清空
├── ai_services.py         # 千问视觉、DeepSeek、图文编码服务
├── config.py              # 环境变量和默认配置
├── requirements.txt       # Python 依赖
├── run_app.sh             # 本地启动脚本
├── .env.example           # 环境变量模板
└── runtime_data/          # 本地运行数据，不能上传 GitHub
```

## GitHub 上传建议

建议上传：

- `app.py`
- `workflow.py`
- `vector_db.py`
- `ai_services.py`
- `config.py`
- `requirements.txt`
- `run_app.sh`
- `.env.example`
- `.gitignore`
- `README.md`
- 你确认可以公开的项目文档，例如项目书或 prompt 文档

不要上传：

- `.env`
- `.venv/`
- `__pycache__/`
- `runtime_data/`
- `runtime_data_*/`
- 用户上传图片
- ChromaDB 向量库文件
- 本地模型缓存
- 任何包含真实 API Key、手机号、学号、工号、用户通知、审计日志的数据文件

## 环境准备

推荐使用 Python 3.11。

```bash
conda create -n campus311 python=3.11 -y
conda activate campus311
python -m pip install -U pip setuptools wheel
pip install -r requirements.txt
```

Apple Silicon 机器建议使用 arm64 原生 Miniforge/Miniconda 环境，避免 x86 Anaconda 与 PyTorch/Transformers 依赖混用。

## 配置环境变量

复制环境变量模板：

```bash
cp .env.example .env
```

然后在 `.env` 中填写自己的 Key：

```env
DASHSCOPE_API_KEY=your_dashscope_key_here
DEEPSEEK_API_KEY=your_deepseek_key_here
QWEN_VL_MODEL=qwen-vl-plus
DEEPSEEK_MODEL=deepseek-chat
DEEPSEEK_BASE_URL=https://api.deepseek.com
```

说明：

- `DASHSCOPE_API_KEY` 用于千问视觉识别。
- `DEEPSEEK_API_KEY` 用于智能精排、通知判断和同义词扩展。
- 不要把真实 `.env` 上传到 GitHub。

## 启动项目

进入项目目录：

```bash
cd myproject
```

启动：

```bash
streamlit run app.py
```

也可以使用脚本：

```bash
chmod +x run_app.sh
./run_app.sh
```

默认访问地址一般是：

```text
http://localhost:8501
```

## 使用流程

1. 进入系统后注册或登录账号。
2. 用户选择“我丢了东西”上传失物图片和描述。
3. 用户选择“我捡到东西”上传拾物图片、地点和联系方式。
4. 系统自动提取物品结构化特征并写入对应向量库。
5. 系统检索另一侧库中的候选物品，并由 DeepSeek 判断是否通知失主。
6. 失主在右上角通知中心查看候选。
7. 到现场确认后点击“已取回”，系统完成状态闭环。

## 管理员功能

管理员端可进行：

- 查看失物区和招领区所有记录。
- 按状态、时间、地点批量删除记录。
- 维护校园地点索引。
- 查看同义词词典。
- 查看 DeepSeek 决策日志、通知日志、闭环记录。
- 导出匹配成功和 badcase 记录。
- 一键清空系统业务记录。

## 数据与隐私

本项目目前使用本地 JSON 文件和 ChromaDB 保存运行数据，适合作为课程项目、原型系统和本地演示。正式部署前建议：

- 将账号、通知、审计日志迁移到 MySQL/PostgreSQL。
- 将图片迁移到对象存储。
- 加入密码加盐哈希、权限校验和访问日志。
- 对上传图片和手机号等敏感信息做脱敏和权限隔离。
- 对管理员的一键删除功能加入二次确认和操作审计。

## 常见问题

### 1. 为什么不要上传 runtime_data？

`runtime_data` 中包含用户上传图片、向量库、通知、审计日志和本地账号数据，可能包含隐私信息，也会让仓库非常大。

### 2. 没有 API Key 能运行吗？

部分界面可以打开，但千问视觉识别、DeepSeek 精排和智能通知会不可用或进入本地兜底逻辑。

### 3. 本地模型很慢怎么办？

Apple M2 8GB 内存可以运行轻量图文模型，但首次加载和批量处理会比较慢。建议减少批量上传数量，或后续改为云端向量服务。

### 4. GitHub 上要不要放演示图片？

可以放少量不含隐私的示例图片到单独的 `examples/` 目录；不要上传真实用户图片。

## 后续优化方向

- 接入更稳定的多模态向量模型。
- 支持一件物品多角度图片聚合入库。
- 增加对象检测/主体裁剪，降低背景干扰。
- 引入数据库和对象存储，支持多人在线部署。
- 将通知升级为短信、企业微信或小程序订阅消息。
- 对 DeepSeek 精排结果进行可视化审计和人工反馈学习。
