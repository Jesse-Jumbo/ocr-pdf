# 免費OCR文本識別系統

一個完全免費的Streamlit OCR應用，支持Tesseract和PaddleOCR兩種OCR引擎。

## ✨ 功能特點

- **雙引擎支持**: 支持Tesseract和PaddleOCR兩種OCR引擎
- **完全免費**: 所有功能完全免費，無需API密鑰
- **中文優化**: 針對中文字體優化
- **多格式輸出**: 支持JSON和文本格式下載
- **即時處理**: 實時顯示OCR處理進度

## 🚀 快速開始

### 本地運行

1. **克隆項目**
   ```bash
   git clone https://github.com/your-username/free-ocr-app.git
   cd free-ocr-app
   ```

2. **安裝依賴**
   ```bash
   # 安裝Python依賴
   pip install -r requirements_new.txt
   
   # 安裝系統依賴 (macOS)
   brew install tesseract poppler
   
   # 安裝系統依賴 (Ubuntu/Debian)
   sudo apt-get install tesseract-ocr tesseract-ocr-chi-tra tesseract-ocr-chi-sim poppler-utils
   ```

3. **運行應用**
   ```bash
   streamlit run streamlit_app_new.py
   ```

### Streamlit Cloud部署

1. **Fork此項目**到你的GitHub帳戶

2. **在Streamlit Cloud上部署**:
   - 前往 [Streamlit Cloud](https://share.streamlit.io/)
   - 點擊 "New app"
   - 選擇你的GitHub倉庫
   - 設置:
     - Main file path: `streamlit_app_new.py`
     - Requirements file: `requirements_new.txt`
     - Packages file: `packages_new.txt`

3. **部署完成**後即可使用

## 📖 使用說明

1. **選擇OCR引擎**: 在側邊欄選擇Tesseract或PaddleOCR
2. **上傳文件**: 選擇要處理的PDF文件
3. **開始處理**: 點擊"開始OCR處理"按鈕
4. **查看結果**: 查看識別結果和下載文件

## 🔧 OCR引擎比較

| 引擎 | 優點 | 缺點 | 適用場景 |
|------|------|------|----------|
| Tesseract | 穩定可靠、處理速度快 | 中文識別準確度一般 | 一般文檔處理 |
| PaddleOCR | 中文識別準確度高 | 處理速度較慢 | 中文文檔處理 |

## 📁 項目結構

```
free-ocr-app/
├── streamlit_app_new.py      # 主應用程式
├── requirements_new.txt      # Python依賴
├── packages_new.txt         # 系統依賴
├── README_new.md           # 說明文檔
└── .gitignore              # Git忽略文件
```

## 🛠️ 技術棧

- **前端**: Streamlit
- **OCR引擎**: Tesseract, PaddleOCR
- **圖像處理**: OpenCV, PIL
- **PDF處理**: pdf2image
- **部署**: Streamlit Cloud

## 📝 開發說明

### 添加新的OCR引擎

1. 在`streamlit_app_new.py`中創建新的OCR類
2. 實現`pdf_to_images`和`extract_text`方法
3. 在`process_pdf_with_ocr`函數中添加新引擎選項

### 自定義圖像預處理

修改`TesseractOCR.preprocess_image`或`PaddleOCRProcessor`中的預處理邏輯。

## 🤝 貢獻

歡迎提交Issue和Pull Request！

## 📄 許可證

MIT License - 完全免費使用

## 🙏 致謝

- [Tesseract OCR](https://github.com/tesseract-ocr/tesseract)
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)
- [Streamlit](https://streamlit.io/)
