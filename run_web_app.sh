#!/bin/bash

# 啟動Streamlit Web應用

echo "正在啟動OCR Web應用..."

# 檢查虛擬環境是否存在
if [ ! -d "ocr_env" ]; then
    echo "虛擬環境不存在，正在創建..."
    python3 -m venv ocr_env
    source ocr_env/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
else
    # 激活虛擬環境
    source ocr_env/bin/activate
fi

# 啟動Streamlit應用
echo "啟動Web應用，請在瀏覽器中打開 http://localhost:8501"
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0
