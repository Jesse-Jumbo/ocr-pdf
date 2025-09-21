#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Web應用 - 完全免費版本
支持Tesseract和PaddleOCR兩種OCR引擎
"""

import streamlit as st
import os
import json
import tempfile
import time
from pathlib import Path
import logging

# 設置頁面配置
st.set_page_config(
    page_title="免費OCR文本識別系統",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 延遲導入OCR相關庫
try:
    import cv2
    import numpy as np
    from PIL import Image
    import pdf2image
    from paddleocr import PaddleOCR
    import pytesseract
    st.success("✅ 所有OCR依賴庫導入成功")
    OCR_AVAILABLE = True
except ImportError as e:
    st.error(f"❌ OCR依賴庫導入失敗: {e}")
    st.error("請確保所有依賴都已正確安裝")
    OCR_AVAILABLE = False

class TesseractOCR:
    """Tesseract OCR處理器 - 完全免費"""
    
    def __init__(self):
        """初始化Tesseract OCR"""
        # 設置Tesseract路徑 (自動檢測)
        import shutil
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
        else:
            # 常見路徑
            possible_paths = [
                '/opt/homebrew/bin/tesseract',  # macOS Homebrew
                '/usr/local/bin/tesseract',     # macOS/Linux
                '/usr/bin/tesseract',           # Linux
                'tesseract'                     # 系統PATH
            ]
            for path in possible_paths:
                if shutil.which(path):
                    pytesseract.pytesseract.tesseract_cmd = path
                    break
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300):
        """將PDF轉換為圖像"""
        try:
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=None,
                last_page=None,
                fmt='RGB',
                thread_count=4,
                poppler_path=None,
                grayscale=False,
                size=None,
                transparent=False,
                single_file=False,
                output_file=None,
                jpegopt=None,
                strict=False,
                use_pdftocairo=False,
                timeout=600
            )
            return [np.array(img) for img in images]
        except Exception as e:
            logger.error(f"PDF轉換失敗: {e}")
            return []
    
    def preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """圖像預處理"""
        # 轉換為灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 對比度增強
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 銳化
        kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 二值化
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 11, 2
        )
        
        return binary
    
    def extract_text(self, image: np.ndarray) -> list:
        """使用Tesseract提取文本"""
        try:
            # 預處理圖像
            processed_image = self.preprocess_image(image)
            
            # 使用多種配置嘗試OCR
            configs = [
                '--psm 3 -c preserve_interword_spaces=1',  # 自動頁面分割
                '--psm 4 -c preserve_interword_spaces=1',  # 單列文本
                '--psm 6 -c preserve_interword_spaces=1',  # 單一文本塊
            ]
            
            lang = 'chi_tra+chi_sim+eng'  # 繁體中文+簡體中文+英文
            all_texts = []
            
            for config in configs:
                try:
                    data = pytesseract.image_to_data(
                        processed_image, 
                        lang=lang, 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    for i in range(len(data['text'])):
                        text = data['text'][i].strip()
                        confidence = int(data['conf'][i])
                        
                        if text and confidence > 30 and len(text) > 0:
                            all_texts.append({
                                "text": text,
                                "confidence": confidence / 100.0,
                                "position": {
                                    "x": data['left'][i],
                                    "y": data['top'][i],
                                    "width": data['width'][i],
                                    "height": data['height'][i]
                                }
                            })
                except Exception as e:
                    logger.warning(f"Tesseract配置 {config} 失敗: {e}")
                    continue
            
            # 去重
            unique_texts = self._deduplicate_texts(all_texts)
            return unique_texts
            
        except Exception as e:
            logger.error(f"Tesseract OCR失敗: {e}")
            return []
    
    def _deduplicate_texts(self, texts: list) -> list:
        """去重文本"""
        unique_texts = []
        for text_data in texts:
            is_duplicate = False
            for existing in unique_texts:
                if (self._texts_overlap(text_data, existing) and 
                    abs(len(text_data["text"]) - len(existing["text"])) < 2):
                    if text_data["confidence"] > existing["confidence"]:
                        unique_texts.remove(existing)
                        unique_texts.append(text_data)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text_data)
        
        return unique_texts
    
    def _texts_overlap(self, text1: dict, text2: dict) -> bool:
        """檢查文本是否重疊"""
        pos1 = text1["position"]
        pos2 = text2["position"]
        
        overlap_x = not (pos1["x"] + pos1["width"] < pos2["x"] or pos2["x"] + pos2["width"] < pos1["x"])
        overlap_y = not (pos1["y"] + pos1["height"] < pos2["y"] or pos2["y"] + pos2["height"] < pos1["y"])
        
        return overlap_x and overlap_y

class PaddleOCRProcessor:
    """PaddleOCR處理器 - 完全免費"""
    
    def __init__(self):
        """初始化PaddleOCR"""
        self.paddle_ocr = None
        self._initialized = False
    
    def _init_paddle_ocr(self):
        """延遲初始化PaddleOCR"""
        if not self._initialized:
            try:
                self.paddle_ocr = PaddleOCR(
                    use_angle_cls=True,
                    lang='ch',  # 中文
                    use_gpu=False,
                    show_log=False,
                    use_space_char=True,
                    det_limit_side_len=960,
                    det_limit_type='max',
                    rec_batch_num=1,
                    max_text_length=25,
                    rec_algorithm='CRNN',
                    cls_thresh=0.9,
                    det_thresh=0.1,
                    det_db_thresh=0.1,
                    det_db_box_thresh=0.3,
                    det_db_unclip_ratio=1.5,
                    det_algorithm='DB',
                    use_dilation=False,
                    det_db_score_mode='fast'
                )
                self._initialized = True
                logger.info("PaddleOCR初始化成功")
            except Exception as e:
                logger.error(f"PaddleOCR初始化失敗: {e}")
                self.paddle_ocr = None
                self._initialized = True
    
    def pdf_to_images(self, pdf_path: str, dpi: int = 300):
        """將PDF轉換為圖像"""
        try:
            images = pdf2image.convert_from_path(
                pdf_path, 
                dpi=dpi,
                first_page=None,
                last_page=None,
                fmt='RGB',
                thread_count=4,
                poppler_path=None,
                grayscale=False,
                size=None,
                transparent=False,
                single_file=False,
                output_file=None,
                jpegopt=None,
                strict=False,
                use_pdftocairo=False,
                timeout=600
            )
            return [np.array(img) for img in images]
        except Exception as e:
            logger.error(f"PDF轉換失敗: {e}")
            return []
    
    def extract_text(self, image: np.ndarray) -> list:
        """使用PaddleOCR提取文本"""
        try:
            self._init_paddle_ocr()
            
            if self.paddle_ocr is None:
                return []
            
            result = self.paddle_ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return []
            
            texts = []
            for line in result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # 計算位置
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    texts.append({
                        "text": text,
                        "confidence": confidence,
                        "position": {
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "width": max(x_coords) - min(x_coords),
                            "height": max(y_coords) - min(y_coords)
                        }
                    })
            
            return texts
            
        except Exception as e:
            logger.error(f"PaddleOCR失敗: {e}")
            return []

def process_pdf_with_ocr(pdf_path: str, ocr_engine: str, dpi: int = 300) -> dict:
    """使用指定的OCR引擎處理PDF"""
    try:
        # 選擇OCR引擎
        if ocr_engine == "Tesseract":
            ocr_processor = TesseractOCR()
        elif ocr_engine == "PaddleOCR":
            ocr_processor = PaddleOCRProcessor()
        else:
            return {"error": "不支持的OCR引擎"}
        
        # 轉換PDF為圖像
        images = ocr_processor.pdf_to_images(pdf_path, dpi=dpi)
        
        if not images:
            return {"error": "PDF轉換失敗"}
        
        result = {
            "file_name": os.path.basename(pdf_path),
            "total_pages": len(images),
            "pages": [],
            "ocr_engine": ocr_engine
        }
        
        # 處理每一頁
        for page_num, image in enumerate(images, 1):
            texts = ocr_processor.extract_text(image)
            
            # 分類文本
            classified_blocks = []
            for text_data in texts:
                classified_blocks.append({
                    "text": text_data["text"],
                    "type": "content",  # 簡單分類
                    "confidence": text_data["confidence"],
                    "position": text_data["position"]
                })
            
            # 按位置排序
            classified_blocks.sort(key=lambda x: (x["position"]["y"], x["position"]["x"]))
            
            page_result = {
                "page_number": page_num,
                "text_blocks": classified_blocks,
                "full_text": "\n".join([block["text"] for block in classified_blocks])
            }
            result["pages"].append(page_result)
        
        return result
        
    except Exception as e:
        return {"error": f"OCR處理失敗: {str(e)}"}

def display_results(result: dict):
    """顯示處理結果"""
    st.markdown("### 📊 處理結果概覽")
    
    # 基本統計
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總頁數", result["total_pages"])
    
    with col2:
        total_blocks = sum(len(page["text_blocks"]) for page in result["pages"])
        st.metric("文本塊總數", total_blocks)
    
    with col3:
        st.metric("OCR引擎", result["ocr_engine"])
    
    with col4:
        avg_confidence = sum(
            sum(block["confidence"] for block in page["text_blocks"]) 
            for page in result["pages"]
        ) / max(total_blocks, 1)
        st.metric("平均置信度", f"{avg_confidence:.2f}")
    
    # 頁面選擇器
    st.markdown("### 📄 頁面詳細信息")
    page_options = [f"第 {page['page_number']} 頁" for page in result["pages"]]
    selected_page_idx = st.selectbox("選擇要查看的頁面:", range(len(page_options)), format_func=lambda x: page_options[x])
    
    if selected_page_idx is not None:
        selected_page = result["pages"][selected_page_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 頁面信息")
            st.write(f"**頁碼:** {selected_page['page_number']}")
            st.write(f"**文本塊數量:** {len(selected_page['text_blocks'])}")
            
            if selected_page["text_blocks"]:
                avg_conf = sum(block["confidence"] for block in selected_page["text_blocks"]) / len(selected_page["text_blocks"])
                st.write(f"**平均置信度:** {avg_conf:.2f}")
        
        with col2:
            st.markdown("#### 識別文本")
            if selected_page["text_blocks"]:
                for i, block in enumerate(selected_page["text_blocks"]):
                    st.write(f"**{i+1}.** {block['text']} (置信度: {block['confidence']:.2f})")
            else:
                st.write("此頁面沒有識別到文本")

def create_download_links(result: dict):
    """創建下載鏈接"""
    st.markdown("### 💾 下載結果")
    
    # JSON下載
    json_data = json.dumps(result, ensure_ascii=False, indent=2)
    st.download_button(
        label="📄 下載JSON文件",
        data=json_data,
        file_name=f"{result['file_name']}_ocr.json",
        mime="application/json",
        key=f"download_json_{int(time.time())}"
    )
    
    # 純文本下載
    full_text = "\n\n".join([f"=== 第 {page['page_number']} 頁 ===" + "\n" + page['full_text'] for page in result["pages"]])
    st.download_button(
        label="📝 下載純文本文件",
        data=full_text,
        file_name=f"{result['file_name']}_text.txt",
        mime="text/plain",
        key=f"download_text_{int(time.time())}"
    )

def main():
    """主函數"""
    # 標題
    st.markdown('<h1 class="main-header">📄 免費OCR文本識別系統</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">支持Tesseract和PaddleOCR兩種完全免費的OCR引擎</p>', unsafe_allow_html=True)
    
    # 檢查OCR可用性
    if not OCR_AVAILABLE:
        st.error("❌ OCR依賴庫未正確安裝，無法使用OCR功能")
        st.markdown("### 請檢查以下依賴是否正確安裝：")
        st.code("""
        pip install streamlit numpy Pillow opencv-python-headless
        pip install paddlepaddle paddleocr pytesseract
        pip install pdf2image
        """)
        st.stop()
    
    # 側邊欄
    with st.sidebar:
        st.markdown("## 🔧 OCR引擎選擇")
        
        # OCR引擎選擇
        ocr_engine = st.radio(
            "選擇OCR引擎:",
            ["Tesseract", "PaddleOCR"],
            help="Tesseract: 穩定可靠，PaddleOCR: 中文識別更準確"
        )
        
        # 處理參數
        st.markdown("### 處理參數")
        dpi = st.slider("圖像DPI", 150, 600, 300, help="更高的DPI會提高識別精度但處理時間更長")
        
        # 引擎信息
        st.markdown("### ℹ️ 引擎信息")
        if ocr_engine == "Tesseract":
            st.info("**Tesseract OCR**\n- 完全免費\n- 穩定可靠\n- 支持多語言\n- 處理速度較快")
        else:
            st.info("**PaddleOCR**\n- 完全免費\n- 中文識別準確\n- 深度學習模型\n- 處理速度較慢")
    
    # 主要內容
    st.markdown("### 📤 上傳PDF文件")
    
    # 文件上傳
    uploaded_file = st.file_uploader(
        "選擇PDF文件",
        type=['pdf'],
        help="支持中文文本的PDF文件"
    )
    
    if uploaded_file is not None:
        # 顯示文件信息
        st.markdown("#### 文件信息")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write(f"**文件名:** {uploaded_file.name}")
        with col2:
            st.write(f"**文件大小:** {uploaded_file.size / 1024 / 1024:.2f} MB")
        with col3:
            st.write(f"**OCR引擎:** {ocr_engine}")
        
        # 處理按鈕
        if st.button("🚀 開始OCR處理", type="primary"):
            # 保存上傳的文件到臨時目錄
            with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name
            
            # 創建進度條
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 處理文件
            status_text.text("正在處理PDF文件...")
            progress_bar.progress(20)
            
            result = process_pdf_with_ocr(tmp_file_path, ocr_engine, dpi)
            
            progress_bar.progress(80)
            status_text.text("正在整理結果...")
            
            # 清理臨時文件
            os.unlink(tmp_file_path)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
            else:
                progress_bar.progress(100)
                status_text.text("處理完成！")
                st.success("✅ OCR處理完成！")
                
                # 顯示結果
                display_results(result)
                create_download_links(result)
    
    # 使用說明
    with st.expander("📚 使用說明"):
        st.markdown("""
        #### 🎯 功能特點
        
        - **雙引擎支持**: 支持Tesseract和PaddleOCR兩種OCR引擎
        - **完全免費**: 所有功能完全免費，無需API密鑰
        - **中文優化**: 針對中文字體優化
        - **多格式輸出**: 支持JSON和文本格式下載
        
        #### 📋 使用步驟
        
        1. **選擇OCR引擎**: 在側邊欄選擇Tesseract或PaddleOCR
        2. **上傳文件**: 選擇要處理的PDF文件
        3. **開始處理**: 點擊"開始OCR處理"按鈕
        4. **查看結果**: 查看識別結果和下載文件
        
        #### ⚙️ 技術參數
        
        - **圖像DPI**: 可調整，建議300-600
        - **語言支持**: 簡體中文、繁體中文、英文
        - **處理時間**: 根據文件大小和頁數而定
        
        #### 🔧 系統要求
        
        - 支持PDF格式文件
        - 建議文件大小不超過50MB
        - 處理時間與文件大小成正比
        """)

if __name__ == "__main__":
    main()
