# 帧神AI — Zhensage AI

> 以帧造物，即是真神 — Build all things by frames, be the ultimate sage

基于 DialoGPT-small（1.24 亿参数 Transformer）的生成式 AI 对话系统。

---

## 项目结构

```
/workspace/
├── app.py                  # Flask 后端
├── templates/
│   └── index.html          # 前端聊天界面
├── dialogpt_model/         # 预训练模型权重（240MB）
├── Dockerfile              # Docker 镜像配置
├── requirements.txt        # Python 依赖
├── render.yaml             # Render.com 部署配置
└── README.md               # 本文件
```

---

## 本地运行

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 启动服务
python app.py

# 3. 浏览器访问
open http://127.0.0.1:5000
```

---

## Render.com 免费部署（推荐）

### 前置条件
- GitHub 账号
- Render 账号（用 GitHub 登录即可）

### 步骤

**1. 将代码推送到 GitHub**

```bash
cd /workspace
git init
git add .
git commit -m "Initial commit"
git remote add origin https://github.com/你的用户名/zhensage-ai.git
git push -u origin main
```

**2. 在 Render 创建服务**

1. 访问 https://dashboard.render.com/
2. 点击 **New +** → **Web Service**
3. 选择你的 GitHub 仓库 `zhensage-ai`
4. 配置：
   - **Runtime**: Docker
   - **Plan**: Free
   - **Branch**: main
5. 点击 **Create Web Service**

**3. 等待部署完成**

- 首次构建约 5-10 分钟（需要下载 PyTorch + Transformers）
- 部署成功后获得类似 `https://zhensage-ai.onrender.com` 的地址

**4. 绑定自定义域名**

1. 在 Render Dashboard 进入你的服务
2. 点击 **Settings** → **Custom Domains**
3. 添加 `zhenshenai.com`
4. 按提示在域名服务商添加 CNAME 记录

---

## 云服务器部署（Linux + Nginx）

### 1. 服务器准备

购买一台云服务器（推荐配置：2核4G，Ubuntu 22.04）：
- 阿里云 ECS
- 腾讯云 CVM
- AWS EC2
- 或其他任意 VPS

### 2. 环境安装

```bash
# 更新系统
sudo apt update && sudo apt upgrade -y

# 安装 Python 3.11
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# 安装 Nginx
sudo apt install -y nginx

# 克隆代码
git clone https://github.com/你的用户名/zhensage-ai.git
cd zhensage-ai

# 创建虚拟环境
python3.11 -m venv venv
source venv/bin/activate

# 安装依赖
pip install -r requirements.txt
```

### 3. 使用 Gunicorn 运行

```bash
# 后台运行
nohup gunicorn --bind 127.0.0.1:5000 --workers 1 --timeout 120 app:app > app.log 2>&1 &
```

### 4. Nginx 反向代理

```bash
sudo nano /etc/nginx/sites-available/zhensage-ai
```

写入：

```nginx
server {
    listen 80;
    server_name zhenshenai.com www.zhenshenai.com;

    location / {
        proxy_pass http://127.0.0.1:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

启用配置：

```bash
sudo ln -s /etc/nginx/sites-available/zhensage-ai /etc/nginx/sites-enabled/
sudo nginx -t
sudo systemctl restart nginx
```

### 5. HTTPS（Let's Encrypt 免费证书）

```bash
sudo apt install -y certbot python3-certbot-nginx
sudo certbot --nginx -d zhenshenai.com -d www.zhenshenai.com
```

---

## Docker 部署

```bash
# 构建镜像
docker build -t zhensage-ai .

# 运行容器
docker run -d -p 5000:5000 --name zhensage-ai zhensage-ai

# 访问
open http://localhost:5000
```

---

## 域名配置

| 记录类型 | 主机记录 | 记录值 |
|---------|---------|--------|
| A | @ | 你的服务器 IP |
| CNAME | www | zhenshenai.com |

> 在域名服务商（阿里云/腾讯云/Namecheap/Cloudflare）的 DNS 管理页面添加以上记录。

---

## 常见问题

**Q: 模型加载很慢？**
A: 首次启动需要加载 240MB 模型文件到内存，约需 10-30 秒。后续请求秒回。

**Q: 免费版 Render 会休眠？**
A: 免费实例在 15 分钟无请求后会进入休眠，下次访问需等待 30 秒冷启动。

**Q: 能否用 GPU 加速？**
A: 当前模型在 CPU 上推理足够快（1-3 秒/回复）。如需 GPU，需升级 Render 付费计划或使用云服务器 GPU 实例。

---

## 技术栈

- **后端**: Flask + Gunicorn
- **模型**: Microsoft DialoGPT-small（124M 参数 Transformer）
- **推理**: PyTorch + Transformers（HuggingFace）
- **前端**: 原生 HTML/CSS/JS
- **部署**: Docker / Render / Nginx
