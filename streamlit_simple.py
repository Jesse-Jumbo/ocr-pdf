#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
簡化版OCR Web應用 - 用於Streamlit Cloud部署測試
"""

import streamlit as st
import os
import json
import tempfile
from pathlib import Path
import time
import logging

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 頁面配置
st.set_page_config(
    page_title="OCR文本識別系統",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 自定義CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        text-align: center;
        margin-bottom: 2rem;
        color: #1f77b4;
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        margin-top: 2rem;
        margin-bottom: 1rem;
        color: #2e8b57;
    }
    .info-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1f77b4;
        margin: 1rem 0;
    }
    .error-box {
        background-color: #ffe6e6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #ff4444;
        margin: 1rem 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #44ff44;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

def main():
    """主函數"""
    # 標題
    st.markdown('<h1 class="main-header">📄 OCR文本識別系統</h1>', unsafe_allow_html=True)
    
    # 檢查依賴
    st.markdown("### 🔍 依賴檢查")
    
    # 檢查基本依賴
    basic_deps = {
        'streamlit': 'Web框架',
        'numpy': '數值計算',
        'Pillow': '圖像處理',
        'pandas': '數據處理'
    }
    
    missing_deps = []
    available_deps = []
    
    for dep, desc in basic_deps.items():
        try:
            __import__(dep)
            available_deps.append(f"✅ {dep} - {desc}")
        except ImportError:
            missing_deps.append(f"❌ {dep} - {desc}")
    
    # 檢查OCR依賴
    ocr_deps = {
        'cv2': 'OpenCV圖像處理',
        'pdf2image': 'PDF轉圖像',
        'paddleocr': 'PaddleOCR引擎',
        'pytesseract': 'Tesseract引擎'
    }
    
    ocr_available = True
    for dep, desc in ocr_deps.items():
        try:
            if dep == 'cv2':
                import cv2
            elif dep == 'pdf2image':
                import pdf2image
            elif dep == 'paddleocr':
                from paddleocr import PaddleOCR
            elif dep == 'pytesseract':
                import pytesseract
            available_deps.append(f"✅ {dep} - {desc}")
        except ImportError as e:
            missing_deps.append(f"❌ {dep} - {desc} (錯誤: {str(e)})")
            ocr_available = False
    
    # 顯示結果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ 已安裝的依賴:**")
        for dep in available_deps:
            st.markdown(dep)
    
    with col2:
        st.markdown("**❌ 缺失的依賴:**")
        for dep in missing_deps:
            st.markdown(dep)
    
    # 如果OCR依賴不可用，顯示安裝指令
    if not ocr_available:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ OCR依賴庫未正確安裝，無法使用OCR功能")
        st.markdown("### 請檢查以下依賴是否正確安裝：")
        st.code("""
        pip install streamlit numpy Pillow opencv-python-headless
        pip install paddlepaddle paddleocr pytesseract
        pip install pdf2image pandas tqdm
        """)
        
        # 添加診斷信息
        st.markdown("### 🔍 診斷信息：")
        st.info("如果依賴仍然無法安裝，可能的原因：")
        st.markdown("""
        1. **Streamlit Cloud限制**：某些依賴可能被限制
        2. **Python版本不兼容**：嘗試使用Python 3.9或3.10
        3. **依賴衝突**：某些包版本可能衝突
        4. **系統依賴缺失**：需要packages.txt安裝系統依賴
        """)
        
        st.markdown("### 📋 建議的解決方案：")
        st.markdown("""
        1. **檢查packages.txt**：確保包含poppler-utils
        2. **使用requirements_minimal.txt**：更簡單的依賴列表
        3. **嘗試不同Python版本**：3.9, 3.10, 3.11
        4. **檢查Streamlit Cloud日誌**：查看詳細錯誤信息
        """)
        
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # 如果所有依賴都可用，顯示成功信息
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.success("🎉 所有依賴都已正確安裝！OCR功能可以使用。")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 基本功能測試
    st.markdown("### 🧪 功能測試")
    
    # 文件上傳
    uploaded_file = st.file_uploader(
        "選擇PDF文件",
        type=['pdf'],
        help="上傳PDF文件進行OCR處理"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ 文件上傳成功: {uploaded_file.name}")
        st.info(f"文件大小: {uploaded_file.size / 1024:.2f} KB")
        
        # 保存文件
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        st.success(f"✅ 文件已保存到: {tmp_path}")
        
        # 清理臨時文件
        try:
            os.unlink(tmp_path)
            st.info("✅ 臨時文件已清理")
        except:
            pass
    
    # 系統信息
    st.markdown("### ℹ️ 系統信息")
    st.info(f"Python版本: {os.sys.version}")
    st.info(f"工作目錄: {os.getcwd()}")
    st.info(f"Streamlit版本: {st.__version__}")

if __name__ == "__main__":
    main()
