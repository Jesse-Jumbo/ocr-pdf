# 🚀 OCR Web應用部署指南

## 1. Streamlit Cloud (最簡單，推薦)

### 步驟：
1. **推送代碼到GitHub**
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git remote add origin https://github.com/你的用戶名/ocr-app.git
   git push -u origin main
   ```

2. **部署到Streamlit Cloud**
   - 訪問 [share.streamlit.io](https://share.streamlit.io/)
   - 用GitHub帳號登入
   - 點擊 "New app"
   - 選擇你的倉庫
   - 設置：
     - Main file path: `streamlit_app.py`
     - Python version: 3.11
   - 點擊 "Deploy!"

3. **獲得公開URL**
   - 部署完成後會得到類似 `https://your-app-name.streamlit.app` 的URL
   - 任何人都可以訪問這個URL使用OCR功能

## 2. Railway (簡單快速)

### 步驟：
1. **推送代碼到GitHub** (同上)

2. **部署到Railway**
   - 訪問 [railway.app](https://railway.app/)
   - 用GitHub帳號登入
   - 點擊 "New Project" → "Deploy from GitHub repo"
   - 選擇你的倉庫
   - Railway會自動檢測到Streamlit應用並部署

3. **獲得公開URL**
   - 部署完成後會得到一個隨機的URL
   - 可以自定義域名

## 3. Heroku (傳統選擇)

### 步驟：
1. **安裝Heroku CLI**
   ```bash
   # macOS
   brew install heroku/brew/heroku
   
   # 或下載安裝包
   # https://devcenter.heroku.com/articles/heroku-cli
   ```

2. **登入Heroku**
   ```bash
   heroku login
   ```

3. **創建Heroku應用**
   ```bash
   heroku create your-ocr-app-name
   ```

4. **部署**
   ```bash
   git add .
   git commit -m "Deploy to Heroku"
   git push heroku main
   ```

5. **獲得公開URL**
   - 會得到類似 `https://your-ocr-app-name.herokuapp.com` 的URL

## 4. Docker部署 (自建服務器)

### 步驟：
1. **構建Docker鏡像**
   ```bash
   docker build -t ocr-app .
   ```

2. **運行容器**
   ```bash
   docker run -p 8501:8501 ocr-app
   ```

3. **使用Nginx反向代理** (可選)
   ```nginx
   server {
       listen 80;
       server_name your-domain.com;
       
       location / {
           proxy_pass http://localhost:8501;
           proxy_set_header Host $host;
           proxy_set_header X-Real-IP $remote_addr;
       }
   }
   ```

## 5. VPS部署 (完全控制)

### 步驟：
1. **租用VPS** (推薦：DigitalOcean, Linode, Vultr)

2. **安裝依賴**
   ```bash
   # Ubuntu/Debian
   sudo apt update
   sudo apt install python3.11 python3.11-pip tesseract-ocr poppler-utils
   
   # 安裝中文語言包
   sudo apt install tesseract-ocr-chi-sim tesseract-ocr-chi-tra
   ```

3. **部署應用**
   ```bash
   git clone https://github.com/你的用戶名/ocr-app.git
   cd ocr-app
   python3.11 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0
   ```

4. **使用PM2保持運行**
   ```bash
   npm install -g pm2
   pm2 start "streamlit run streamlit_app.py --server.port=8501 --server.address=0.0.0.0" --name ocr-app
   pm2 save
   pm2 startup
   ```

## 成本比較

| 平台 | 免費額度 | 付費價格 | 易用性 | 推薦度 |
|------|----------|----------|--------|--------|
| Streamlit Cloud | 3個應用 | 免費 | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| Railway | 有限制 | $5/月起 | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| Heroku | 有限制 | $7/月起 | ⭐⭐⭐ | ⭐⭐⭐ |
| VPS | 無 | $5-20/月 | ⭐⭐ | ⭐⭐⭐ |

## 推薦方案

**新手推薦**: Streamlit Cloud
- 完全免費
- 一鍵部署
- 自動更新

**進階用戶**: Railway
- 功能豐富
- 價格合理
- 部署簡單

**企業用戶**: VPS + Docker
- 完全控制
- 可擴展
- 成本可控
