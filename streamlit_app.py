#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Web應用 - Streamlit Cloud版本
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
    
    # 檢查基本依賴
    st.markdown("### 🔍 依賴檢查")
    
    # 檢查基本依賴
    basic_deps = {
        'streamlit': 'Web框架',
        'numpy': '數值計算'
    }
    
    missing_deps = []
    available_deps = []
    
    for dep, desc in basic_deps.items():
        try:
            __import__(dep)
            available_deps.append(f"✅ {dep} - {desc}")
        except ImportError as e:
            missing_deps.append(f"❌ {dep} - {desc} (錯誤: {str(e)})")
    
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
    
    # 如果基本依賴不可用，顯示錯誤
    if missing_deps:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.error("❌ 基本依賴未正確安裝")
        st.markdown("### 請檢查以下依賴是否正確安裝：")
        st.code("""
        pip install streamlit numpy Pillow pandas
        """)
        st.markdown("</div>", unsafe_allow_html=True)
        st.stop()
    
    # 如果所有基本依賴都可用，顯示成功信息
    st.markdown('<div class="success-box">', unsafe_allow_html=True)
    st.success("🎉 基本依賴都已正確安裝！")
    st.markdown("</div>", unsafe_allow_html=True)
    
    # 基本功能測試
    st.markdown("### 🧪 基本功能測試")
    
    # 文件上傳
    uploaded_file = st.file_uploader(
        "選擇PDF文件",
        type=['pdf'],
        help="上傳PDF文件進行處理"
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
    
    # OCR依賴檢查
    st.markdown("### 🔍 OCR依賴檢查")
    
    ocr_deps = {
        'PIL': 'Pillow圖像處理',
        'pandas': '數據處理',
        'cv2': 'OpenCV圖像處理',
        'pdf2image': 'PDF轉圖像',
        'paddleocr': 'PaddleOCR引擎',
        'pytesseract': 'Tesseract引擎'
    }
    
    ocr_available = True
    ocr_missing = []
    ocr_available_list = []
    
    for dep, desc in ocr_deps.items():
        try:
            if dep == 'PIL':
                from PIL import Image
            elif dep == 'pandas':
                import pandas
            elif dep == 'cv2':
                import cv2
            elif dep == 'pdf2image':
                import pdf2image
            elif dep == 'paddleocr':
                from paddleocr import PaddleOCR
            elif dep == 'pytesseract':
                import pytesseract
            ocr_available_list.append(f"✅ {dep} - {desc}")
        except ImportError as e:
            ocr_missing.append(f"❌ {dep} - {desc} (錯誤: {str(e)})")
            ocr_available = False
    
    # 顯示OCR依賴結果
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**✅ OCR依賴已安裝:**")
        for dep in ocr_available_list:
            st.markdown(dep)
    
    with col2:
        st.markdown("**❌ OCR依賴缺失:**")
        for dep in ocr_missing:
            st.markdown(dep)
    
    if not ocr_available:
        st.markdown('<div class="error-box">', unsafe_allow_html=True)
        st.warning("⚠️ OCR依賴未完全安裝，但基本功能可用")
        st.markdown("### OCR功能需要以下依賴：")
        st.code("""
        pip install Pillow pandas
        pip install opencv-python-headless
        pip install pdf2image
        pip install paddlepaddle paddleocr
        pip install pytesseract
        """)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown('<div class="success-box">', unsafe_allow_html=True)
        st.success("🎉 所有OCR依賴都已正確安裝！")
        st.markdown("</div>", unsafe_allow_html=True)
    
    # 系統信息
    st.markdown("### ℹ️ 系統信息")
    st.info(f"Python版本: {os.sys.version}")
    st.info(f"工作目錄: {os.getcwd()}")
    st.info(f"Streamlit版本: {st.__version__}")
    
    # 部署建議
    st.markdown("### 📋 部署建議")
    st.markdown("""
    **如果OCR依賴無法安裝，建議：**
    
    1. **使用本地部署**：在本地環境運行完整OCR功能
    2. **使用Docker**：容器化部署，避免依賴衝突
    3. **使用其他平台**：如Railway、Heroku等
    4. **簡化功能**：只使用基本文件處理功能
    
    **Streamlit Cloud限制：**
    - 某些依賴可能被限制
    - 系統依賴安裝有限制
    - 記憶體和CPU使用有限制
    """)

if __name__ == "__main__":
    main()
