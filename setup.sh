#!/bin/bash

# OCR環境設置腳本

echo "正在設置OCR環境..."

# 檢查是否安裝了Homebrew
if ! command -v brew &> /dev/null; then
    echo "請先安裝Homebrew: https://brew.sh/"
    exit 1
fi

# 檢查Python3
if ! command -v python3 &> /dev/null; then
    echo "請先安裝Python3"
    exit 1
fi

# 安裝Tesseract
echo "安裝Tesseract..."
brew install tesseract tesseract-lang

# 創建虛擬環境
echo "創建虛擬環境..."
python3 -m venv ocr_env

# 激活虛擬環境
echo "激活虛擬環境..."
source ocr_env/bin/activate

# 升級pip
echo "升級pip..."
pip install --upgrade pip

# 安裝Python依賴
echo "安裝Python依賴..."
pip install -r requirements.txt

# 下載PaddleOCR模型（首次運行時會自動下載）
echo "下載PaddleOCR模型..."
python -c "from paddleocr import PaddleOCR; PaddleOCR(use_angle_cls=True, lang='ch')"

echo "環境設置完成！"
echo ""
echo "使用方法："
echo "1. 激活虛擬環境: source ocr_env/bin/activate"
echo "2. 運行OCR: python ocr_processor.py"
echo "3. 退出虛擬環境: deactivate"
