# 🚀 快速部署指南

## 最簡單的部署方式 - Streamlit Cloud

### 1. 準備GitHub倉庫
```bash
# 初始化Git倉庫
git init

# 添加所有文件
git add .

# 提交
git commit -m "Initial commit: OCR Web App"

# 連接到GitHub (替換為你的倉庫URL)
git remote add origin https://github.com/你的用戶名/ocr-app.git

# 推送到GitHub
git push -u origin main
```

### 2. 部署到Streamlit Cloud
1. 訪問 [share.streamlit.io](https://share.streamlit.io/)
2. 用GitHub帳號登入
3. 點擊 "New app"
4. 選擇你的倉庫 `你的用戶名/ocr-app`
5. 設置：
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11`
6. 點擊 "Deploy!"

### 3. 完成！
- 等待5-10分鐘部署完成
- 獲得公開URL，例如：`https://your-app-name.streamlit.app`
- 任何人都可以訪問使用OCR功能

## 其他部署選項

### Railway (推薦)
1. 訪問 [railway.app](https://railway.app/)
2. 連接GitHub倉庫
3. 自動部署

### Heroku
1. 安裝Heroku CLI
2. 創建應用：`heroku create your-app-name`
3. 部署：`git push heroku main`

### Docker
```bash
# 構建鏡像
docker build -t ocr-app .

# 運行
docker run -p 8501:8501 ocr-app
```

## 注意事項

- **免費額度**: Streamlit Cloud提供3個免費應用
- **文件大小**: 建議上傳的PDF不超過50MB
- **處理時間**: 大文件可能需要較長處理時間
- **並發限制**: 免費版本有並發用戶限制

## 故障排除

### 如果部署失敗
1. 檢查 `requirements.txt` 是否包含所有依賴
2. 確保 `streamlit_app.py` 在根目錄
3. 檢查Python版本設置為3.11

### 如果OCR不工作
1. 確保PaddleOCR模型下載完成
2. 檢查Tesseract語言包是否安裝
3. 查看Streamlit Cloud的日誌輸出

## 成本比較

| 平台 | 免費額度 | 付費價格 | 推薦度 |
|------|----------|----------|--------|
| Streamlit Cloud | 3個應用 | 免費 | ⭐⭐⭐⭐⭐ |
| Railway | 有限制 | $5/月 | ⭐⭐⭐⭐ |
| Heroku | 有限制 | $7/月 | ⭐⭐⭐ |
