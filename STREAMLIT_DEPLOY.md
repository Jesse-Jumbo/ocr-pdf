# Streamlit Cloud 部署指南

## 🚀 快速部署步驟

### 1. 準備GitHub倉庫
```bash
git init
git add .
git commit -m "Initial commit: OCR web application"

# 推送到GitHub
git remote add origin https://github.com/你的用戶名/你的倉庫名.git
git push -f origin main
```

### 2. 部署到Streamlit Cloud
1. 訪問 [share.streamlit.io](https://share.streamlit.io)
2. 用GitHub帳號登入
3. 點擊 "New app"
4. 選擇你的倉庫
5. 設置：
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11` (推薦)
6. 點擊 "Deploy!"

### 3. 等待部署完成
- 通常需要5-10分鐘
- 查看部署日誌確認沒有錯誤

## 🔧 故障排除

### 常見問題

#### 1. ModuleNotFoundError: No module named 'cv2'
**解決方案**：
- 使用 `opencv-python-headless` 而不是 `opencv-python`
- 確保 `requirements.txt` 包含正確版本

#### 2. 大文件上傳失敗
**解決方案**：
- 刪除 `.git` 目錄重新開始
- 確保 `.gitignore` 排除大文件

#### 3. 依賴安裝失敗
**解決方案**：
- 使用 `requirements_streamlit.txt` 替代 `requirements.txt`
- 檢查Python版本兼容性

## 📁 文件結構
```
OCR/
├── streamlit_app.py          # 主應用文件
├── requirements.txt          # 依賴列表
├── requirements_streamlit.txt # Streamlit Cloud優化版本
├── check_deployment.py       # 部署檢查腳本
├── .gitignore               # Git忽略文件
├── README.md                # 項目說明
└── .streamlit/
    └── config.toml          # Streamlit配置
```

## ⚙️ 配置說明

### requirements.txt 關鍵依賴
- `opencv-python-headless`: 無GUI的OpenCV版本
- `paddlepaddle==2.6.2`: PaddleOCR核心
- `paddleocr==2.7.3`: OCR引擎
- `streamlit==1.28.1`: Web框架

### 環境變量
Streamlit Cloud會自動處理大部分環境變量，但可以設置：
- `TESSERACT_CMD`: Tesseract路徑（通常自動檢測）

## 🎯 部署後測試

1. 上傳一個小PDF文件測試
2. 檢查OCR功能是否正常
3. 確認下載功能正常
4. 測試不同DPI設置

## 📞 支援

如果遇到問題：
1. 檢查Streamlit Cloud的部署日誌
2. 運行 `python check_deployment.py` 本地測試
3. 確認所有依賴版本兼容
