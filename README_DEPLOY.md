# Streamlit Cloud 部署指南

## 🚀 快速部署

### 1. 準備GitHub倉庫
```bash
git add .
git commit -m "Prepare for Streamlit Cloud deployment"
git push origin main
```

### 2. 部署到Streamlit Cloud
1. 訪問 [share.streamlit.io](https://share.streamlit.io)
2. 用GitHub帳號登入
3. 點擊 "New app"
4. 設置：
   - **Repository**: `Jesse-Jumbo/ocr-pdf`
   - **Branch**: `main`
   - **Main file path**: `streamlit_app.py`
   - **Python version**: `3.11`
5. 點擊 "Deploy!"

## 📋 文件說明

### 主要文件
- `streamlit_app.py` - 主應用文件（Streamlit Cloud版本）
- `requirements.txt` - 依賴列表（只包含基本依賴）
- `packages.txt` - 系統依賴（poppler等）

### 備用文件
- `streamlit_app_old.py` - 原始完整版OCR應用
- `streamlit_simple.py` - 簡化版測試應用
- `streamlit_basic.py` - 基本版測試應用

## 🔧 依賴說明

### 當前requirements.txt包含：
```
streamlit==1.28.1
numpy==1.24.3
Pillow==10.1.0
pandas==2.1.4
```

### 如果需要完整OCR功能：
1. 將`streamlit_app_old.py`重命名為`streamlit_app.py`
2. 使用完整的requirements.txt
3. 確保packages.txt包含所有系統依賴

## ⚠️ 注意事項

1. **Streamlit Cloud限制**：
   - 某些依賴可能被限制
   - 系統依賴安裝有限制
   - 記憶體和CPU使用有限制

2. **依賴安裝**：
   - Streamlit Cloud只讀取根目錄的requirements.txt
   - 必須指定版本號
   - 系統依賴需要packages.txt

3. **如果部署失敗**：
   - 檢查Streamlit Cloud日誌
   - 確認依賴版本兼容性
   - 考慮使用其他部署平台

## 🎯 預期結果

部署成功後應該看到：
- ✅ 基本依賴檢查通過
- ✅ 文件上傳功能正常
- ✅ OCR依賴狀態顯示
- ✅ 系統信息正常顯示

## 📞 故障排除

如果遇到問題：
1. 檢查Streamlit Cloud的部署日誌
2. 確認requirements.txt格式正確
3. 嘗試不同的Python版本
4. 考慮使用本地部署或Docker
