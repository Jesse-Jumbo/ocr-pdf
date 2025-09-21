# 📄 OCR文本識別Web應用

一個基於Streamlit的Web應用，專門用於處理中文PDF文件的OCR識別，支援直式和橫式文本。

## ✨ 功能特點

- **🌐 Web界面**: 現代化響應式設計，無需命令行
- **🔧 雙引擎OCR**: 結合PaddleOCR和Tesseract，提高識別準確度
- **📐 方向檢測**: 自動檢測直式/橫式文本
- **🇨🇳 中文優化**: 針對中文字體優化
- **📊 實時進度**: 處理過程可視化
- **💾 多格式輸出**: JSON、純文本、統計報告
- **☁️ 雲端部署**: 支持多種部署平台

## 🚀 快速開始

### 本地運行
```bash
# 一鍵啟動
./run_web_app.sh

# 或手動啟動
source ocr_env/bin/activate
streamlit run streamlit_app.py
```

### 雲端部署
查看 [QUICK_DEPLOY.md](QUICK_DEPLOY.md) 了解詳細部署步驟。

## 📖 使用說明

### Web界面操作
1. **上傳文件**: 在"上傳處理"頁面選擇PDF文件
2. **開始處理**: 點擊"開始OCR處理"按鈕
3. **查看進度**: 實時查看處理進度和狀態
4. **瀏覽結果**: 在"結果查看"頁面查看識別結果
5. **下載數據**: 下載JSON、文本或統計報告

### 支持的格式
- **輸入**: PDF文件
- **輸出**: JSON、純文本、統計報告
- **語言**: 簡體中文、繁體中文、英文
- **方向**: 直式、橫式文本

## 🛠️ 技術架構

- **前端**: Streamlit Web框架
- **OCR引擎**: PaddleOCR + Tesseract
- **圖像處理**: OpenCV + PIL
- **PDF處理**: pdf2image + poppler
- **部署**: 支持Docker、Streamlit Cloud、Railway等

## 📊 輸出格式

每個PDF會生成包含以下信息的JSON文件：

```json
{
  "file_name": "文件名.pdf",
  "total_pages": 頁數,
  "pages": [
    {
      "page_number": 1,
      "text_direction": "vertical|horizontal",
      "total_text_blocks": 文本塊數量,
      "text_blocks": [
        {
          "text": "識別的文字",
          "confidence": 0.95,
          "position": {"x": 100, "y": 200, "width": 50, "height": 20},
          "source": "paddle|tesseract"
        }
      ],
      "full_text": "完整頁面文字"
    }
  ]
}
```

## 🔧 技術細節

- **PaddleOCR**: 主要用於中文識別，準確度高
- **Tesseract**: 輔助識別，提高覆蓋率
- **方向檢測**: 基於文本塊長寬比智能判斷
- **圖像預處理**: 去噪、二值化、角度校正
- **結果合併**: 智能合併雙引擎結果，避免重複

## 📝 授權

MIT License - 可自由使用和修改

## 🤝 貢獻

歡迎提交Issue和Pull Request來改進這個項目！
