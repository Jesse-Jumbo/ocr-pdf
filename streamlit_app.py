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

# 添加自定義CSS
st.markdown("""
<style>
.scrollable-container {
    max-height: 600px;
    overflow-y: auto;
    border: 1px solid #e0e0e0;
    border-radius: 5px;
    padding: 10px;
    background-color: #fafafa;
}
</style>
""", unsafe_allow_html=True)

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
        """使用Tesseract提取文本 - 改善標點符號和標題識別"""
        try:
            # 預處理圖像
            processed_image = self.preprocess_image(image)
            
            # 使用多種配置來改善識別
            configs = [
                '--psm 3 -c preserve_interword_spaces=1 -c textord_min_linesize=1.5 -c textord_old_baselines=0',
                '--psm 4 -c preserve_interword_spaces=1 -c textord_min_linesize=1.5',
                '--psm 6 -c preserve_interword_spaces=1 -c textord_min_linesize=1.5'
            ]
            
            lang = 'chi_tra+chi_sim+eng'  # 繁體中文+簡體中文+英文
            best_result = ""
            best_confidence = 0
            
            # 嘗試多種配置，選擇最佳結果
            for config in configs:
                try:
                    text_result = pytesseract.image_to_string(
                        processed_image, 
                        lang=lang, 
                        config=config
                    )
                    
                    # 計算平均置信度
                    data = pytesseract.image_to_data(
                        processed_image, 
                        lang=lang, 
                        config=config, 
                        output_type=pytesseract.Output.DICT
                    )
                    
                    confidences = [int(conf) for conf in data['conf'] if int(conf) > 0]
                    avg_confidence = sum(confidences) / len(confidences) if confidences else 0
                    
                    if avg_confidence > best_confidence:
                        best_result = text_result
                        best_confidence = avg_confidence
                        best_data = data
                        
                except Exception as e:
                    logger.warning(f"Tesseract配置 {config} 失敗: {e}")
                    continue
            
            if not best_result:
                return []
            
            # 後處理文本，改善標點符號識別
            processed_text = self._post_process_text(best_result)
            lines = processed_text.strip().split('\n')
            
            # 創建文本塊
            text_blocks = []
            current_line = ""
            current_y = None
            current_x = 0
            current_width = 0
            current_height = 0
            max_confidence = 0
            
            for i in range(len(best_data['text'])):
                text = best_data['text'][i].strip()
                confidence = int(best_data['conf'][i])
                x = best_data['left'][i]
                y = best_data['top'][i]
                width = best_data['width'][i]
                height = best_data['height'][i]
                
                if text and confidence > 30:
                    # 如果是新行或位置差距較大，保存當前行
                    if current_y is not None and abs(y - current_y) > 15:
                        if current_line.strip():
                            # 後處理當前行
                            processed_line = self._post_process_text(current_line.strip())
                            text_blocks.append({
                                "text": processed_line,
                                "confidence": max_confidence / 100.0,
                                "position": {
                                    "x": current_x,
                                    "y": current_y,
                                    "width": current_width,
                                    "height": current_height
                                }
                            })
                        current_line = text
                        current_y = y
                        current_x = x
                        current_width = width
                        current_height = height
                        max_confidence = confidence
                    else:
                        # 同一行，累積文本
                        if current_line:
                            current_line += text
                        else:
                            current_line = text
                            current_y = y
                            current_x = x
                            current_width = width
                            current_height = height
                            max_confidence = confidence
                        
                        # 更新位置信息
                        current_width = max(current_width, x + width - current_x)
                        current_height = max(current_height, y + height - current_y)
                        max_confidence = max(max_confidence, confidence)
            
            # 保存最後一行
            if current_line.strip():
                processed_line = self._post_process_text(current_line.strip())
                text_blocks.append({
                    "text": processed_line,
                    "confidence": max_confidence / 100.0,
                    "position": {
                        "x": current_x,
                        "y": current_y,
                        "width": current_width,
                        "height": current_height
                    }
                })
            
            return text_blocks
            
        except Exception as e:
            logger.error(f"Tesseract OCR失敗: {e}")
            return []
    
    def _post_process_text(self, text: str) -> str:
        """後處理文本，改善標點符號和格式"""
        if not text:
            return text
        
        # 修復常見的標點符號識別錯誤
        replacements = {
            # 修復句號
            '。': '。',
            '．': '。',
            '·': '。',
            '•': '。',
            'o': '。',
            'O': '。',
            '0': '。',
            
            # 修復逗號
            '，': '，',
            ',': '，',
            '、': '，',
            
            # 修復冒號
            '：': '：',
            ':': '：',
            
            # 修復分號
            '；': '；',
            ';': '；',
            
            # 修復問號
            '？': '？',
            '?': '？',
            
            # 修復感嘆號
            '！': '！',
            '!': '！',
            
            # 修復括號
            '（': '（',
            '(': '（',
            '）': '）',
            ')': '）',
            
            # 修復引號
            '"': '"',
            '"': '"',
            "'": "'",
            "'": "'",
            
            # 修復破折號
            '—': '—',
            '-': '—',
            '–': '—',
            
            # 修復省略號
            '…': '…',
            '...': '…',
        }
        
        # 應用替換
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        # 修復常見的識別錯誤
        text = text.replace('o ', '。')  # o 後面跟空格 -> 句號
        text = text.replace('o\n', '。\n')  # o 後面跟換行 -> 句號
        text = text.replace('o', '。')  # 單獨的 o -> 句號
        
        # 修復列表符號
        text = text.replace('o ', '• ')  # o 開頭 -> 列表符號
        text = text.replace('o\n', '•\n')  # o 開頭換行 -> 列表符號
        
        # 修復標題識別（大寫字母開頭且較短的行）
        lines = text.split('\n')
        processed_lines = []
        
        for line in lines:
            line = line.strip()
            if not line:
                processed_lines.append(line)
                continue
                
            # 檢查是否可能是標題
            if (len(line) < 50 and 
                (line[0].isupper() or line[0].isdigit()) and
                not line.endswith(('。', '，', '：', '；', '？', '！'))):
                # 可能是標題，確保以句號結尾
                if not line.endswith('。'):
                    line += '。'
            
            processed_lines.append(line)
        
        return '\n'.join(processed_lines)
    
    def _merge_nearby_texts(self, texts: list) -> list:
        """合併相近的文本，解決斷句問題"""
        if not texts:
            return []
        
        # 按位置排序
        sorted_texts = sorted(texts, key=lambda x: (x["position"]["y"], x["position"]["x"]))
        merged_texts = []
        
        for text_data in sorted_texts:
            if not merged_texts:
                merged_texts.append(text_data)
                continue
            
            last_text = merged_texts[-1]
            
            # 檢查是否應該合併
            should_merge = False
            
            # 1. 檢查垂直位置是否相近（同一行）
            y_diff = abs(text_data["position"]["y"] - last_text["position"]["y"])
            if y_diff < 20:  # 20像素內視為同一行
                # 2. 檢查水平位置是否連續
                x_gap = text_data["position"]["x"] - (last_text["position"]["x"] + last_text["position"]["width"])
                if x_gap < 50:  # 50像素內視為連續文本
                    should_merge = True
            
            if should_merge:
                # 合併文本
                merged_text = last_text["text"] + text_data["text"]
                merged_position = {
                    "x": min(last_text["position"]["x"], text_data["position"]["x"]),
                    "y": min(last_text["position"]["y"], text_data["position"]["y"]),
                    "width": max(last_text["position"]["x"] + last_text["position"]["width"], 
                               text_data["position"]["x"] + text_data["position"]["width"]) - 
                            min(last_text["position"]["x"], text_data["position"]["x"]),
                    "height": max(last_text["position"]["y"] + last_text["position"]["height"], 
                                text_data["position"]["y"] + text_data["position"]["height"]) - 
                             min(last_text["position"]["y"], text_data["position"]["y"])
                }
                
                # 更新最後一個文本
                merged_texts[-1] = {
                    "text": merged_text,
                    "confidence": max(last_text["confidence"], text_data["confidence"]),
                    "position": merged_position
                }
            else:
                merged_texts.append(text_data)
        
        return merged_texts
    
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


def process_pdf_with_ocr(pdf_path: str, ocr_engine: str, dpi: int = 300, progress_callback=None) -> dict:
    """使用Tesseract OCR處理PDF - 支持即時回調"""
    try:
        # 只使用Tesseract
        ocr_processor = TesseractOCR()
        
        # 轉換PDF為圖像
        images = ocr_processor.pdf_to_images(pdf_path, dpi=dpi)
        
        if not images:
            return {"error": "PDF轉換失敗"}
        
        result = {
            "file_name": os.path.basename(pdf_path),
            "total_pages": len(images),
            "pages": [],
            "ocr_engine": "Tesseract"
        }
        
        # 處理每一頁
        for page_num, image in enumerate(images, 1):
            if progress_callback:
                progress_callback(f"正在處理第 {page_num} 頁...", page_num, len(images))
            
            texts = ocr_processor.extract_text(image)
            
            # 直接使用文本，保持原始格式
            page_result = {
                "page_number": page_num,
                "text_blocks": texts,
                "full_text": "\n".join([block["text"] for block in texts])
            }
            result["pages"].append(page_result)
            
            # 即時回調，讓UI更新
            if progress_callback:
                progress_callback(f"第 {page_num} 頁處理完成", page_num, len(images), result)
        
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

def display_comparison_view(result: dict, pdf_images=None):
    """顯示對比視窗 - 原文件與識別結果並排顯示"""
    st.markdown("### 🔍 原文件與識別結果對比")
    
    # 創建兩列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 原文件")
        if pdf_images:
            # 在可滾動的容器中顯示所有頁面
            st.markdown('<div class="scrollable-container">', unsafe_allow_html=True)
            for i, image in enumerate(pdf_images):
                st.image(image, caption=f"第 {i+1} 頁", use_column_width=True)
                if i < len(pdf_images) - 1:  # 不在最後一頁後添加分隔線
                    st.markdown("---")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.info("原文件圖像預覽功能需要重新上傳文件")
    
    with col2:
        st.markdown("#### 📝 識別結果")
        
        # 頁面選擇器
        page_options = [f"第 {page['page_number']} 頁" for page in result["pages"]]
        selected_page_idx = st.selectbox("選擇要查看的頁面:", range(len(page_options)), format_func=lambda x: page_options[x])
        
        if selected_page_idx is not None:
            selected_page = result["pages"][selected_page_idx]
            
            if selected_page["text_blocks"]:
                # 顯示識別到的文本
                for i, block in enumerate(selected_page["text_blocks"]):
                    st.write(f"**{i+1}.** {block['text']}")
            else:
                st.write("此頁面沒有識別到文本")
            
            # 顯示完整文本
            st.markdown("#### 📄 完整文本")
            st.text_area("識別結果:", value=selected_page['full_text'], height=300, key=f"text_area_{selected_page_idx}")

def download_text_file(result: dict, key_suffix: str = ""):
    """下載文本文件"""
    full_text = "\n\n".join([f"=== 第 {page['page_number']} 頁 ===" + "\n" + page['full_text'] for page in result["pages"]])
    
    st.download_button(
        label="📝 下載TXT文件",
        data=full_text,
        file_name=f"{result['file_name']}_text.txt",
        mime="text/plain",
        key=f"download_txt_{key_suffix}_{int(time.time())}"
    )

def download_json_file(result: dict, key_suffix: str = ""):
    """下載JSON文件"""
    # 簡化的JSON格式（純文字）
    simplified_result = {
        "file_name": result["file_name"],
        "total_pages": result["total_pages"],
        "ocr_engine": result["ocr_engine"],
        "pages": []
    }
    
    for page in result["pages"]:
        simplified_page = {
            "page_number": page["page_number"],
            "text": page["full_text"]  # 只保留純文字
        }
        simplified_result["pages"].append(simplified_page)
    
    json_data = json.dumps(simplified_result, ensure_ascii=False, indent=2)
    
    st.download_button(
        label="📄 下載JSON文件",
        data=json_data,
        file_name=f"{result['file_name']}_ocr.json",
        mime="application/json",
        key=f"download_json_{key_suffix}_{int(time.time())}"
    )

def main():
    """主函數"""
    # 標題
    st.markdown('<h1 class="main-header">📄 免費OCR文本識別系統</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">使用Tesseract OCR引擎，完全保留文字樣式和斷句</p>', unsafe_allow_html=True)
    
    # 檢查OCR可用性
    if not OCR_AVAILABLE:
        st.error("❌ OCR依賴庫未正確安裝，無法使用OCR功能")
        st.markdown("### 請檢查以下依賴是否正確安裝：")
        st.code("""
        pip install streamlit numpy Pillow opencv-python-headless
        pip install pytesseract pdf2image
        """)
        st.stop()
    
    # 初始化session state
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = None
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'history' not in st.session_state:
        st.session_state.history = []
    if 'pdf_images' not in st.session_state:
        st.session_state.pdf_images = None
    if 'current_progress' not in st.session_state:
        st.session_state.current_progress = None
    
    # 側邊欄
    with st.sidebar:
        st.markdown("## 🔧 OCR引擎")
        
        # 固定使用Tesseract
        ocr_engine = "Tesseract"
        st.success("✅ 使用 Tesseract OCR")
        
        # 處理參數
        st.markdown("### 處理參數")
        dpi = st.slider("圖像DPI", 150, 600, 300, help="更高的DPI會提高識別精度但處理時間更長")
        
        # 引擎信息
        st.markdown("### ℹ️ 引擎信息")
        st.info("**Tesseract OCR**\n- 完全免費\n- 穩定可靠\n- 支持多語言\n- 處理速度較快\n- 完全保留文字樣式和斷句\n- 優化的中文識別")
        
        # 歷史記錄
        if st.session_state.history:
            st.markdown("### 📚 歷史記錄")
            for i, item in enumerate(st.session_state.history):
                with st.expander(f"📄 {item['file_name']} ({item['total_pages']}頁)"):
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        if st.button("👁️ 預覽", key=f"preview_{i}"):
                            st.session_state.processing_results = item
                            st.session_state.current_file = item['file_name']
                    with col2:
                        download_text_file(item, f"hist_{i}")
                    with col3:
                        download_json_file(item, f"hist_{i}")
    
    # 主要內容
    st.markdown("### 📤 上傳PDF文件")
    
    # 文件上傳
    uploaded_file = st.file_uploader(
        "選擇PDF文件",
        type=['pdf'],
        help="支持中文文本的PDF文件"
    )
    
    # 如果有處理結果，顯示下載按鈕
    if st.session_state.processing_results:
        st.markdown("### 💾 下載結果")
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            download_text_file(st.session_state.processing_results, "main")
        
        with col2:
            download_json_file(st.session_state.processing_results, "main")
        
        with col3:
            if st.button("🗑️ 清除結果", type="secondary"):
                st.session_state.processing_results = None
                st.session_state.current_file = None
                st.rerun()
    
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
        
        # 保存文件到臨時目錄
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # 處理按鈕
        if st.button("🚀 開始OCR處理", type="primary"):
            st.session_state.is_processing = True
            
            # 創建進度條和狀態顯示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 轉換所有頁面為圖像並保存到session state
            all_images = pdf2image.convert_from_path(tmp_file_path, dpi=150)
            st.session_state.pdf_images = all_images
            
            # 定義進度回調函數
            def progress_callback(message, current_page, total_pages, partial_result=None):
                progress = current_page / total_pages
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current_page}/{total_pages})")
            
            # 處理文件
            result = process_pdf_with_ocr(tmp_file_path, ocr_engine, dpi, progress_callback)
            
            # 清理臨時文件
            os.unlink(tmp_file_path)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                st.session_state.is_processing = False
            else:
                progress_bar.progress(100)
                status_text.text("處理完成！")
                st.success("✅ OCR處理完成！")
                
                # 保存結果到session state
                st.session_state.processing_results = result
                st.session_state.current_file = uploaded_file.name
                
                # 添加到歷史記錄
                if result not in st.session_state.history:
                    st.session_state.history.insert(0, result)
                    # 限制歷史記錄數量
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[:10]
                
                st.session_state.is_processing = False
                st.rerun()
    
    # 如果有處理結果，顯示對比視窗
    if st.session_state.processing_results and not st.session_state.is_processing:
        display_comparison_view(st.session_state.processing_results, st.session_state.pdf_images)
    
    # 使用說明
    with st.expander("📚 使用說明"):
        st.markdown("""
        #### 🎯 功能特點
        
        - **Tesseract OCR**: 使用穩定可靠的Tesseract OCR引擎
        - **完全免費**: 所有功能完全免費，無需API密鑰
        - **中文優化**: 針對中文字體優化，完全保留文字樣式和斷句
        - **多格式輸出**: 支持JSON和文本格式下載
        - **歷史記錄**: 保存處理歷史，方便重新下載
        - **對比視窗**: 原文件與識別結果並排顯示
        
        #### 📋 使用步驟
        
        1. **上傳文件**: 選擇要處理的PDF文件
        2. **預覽文件**: 查看文件第一頁預覽
        3. **開始處理**: 點擊"開始OCR處理"按鈕
        4. **查看結果**: 在對比視窗中查看原文件與識別結果
        5. **下載文件**: 下載TXT或JSON格式結果
        6. **歷史記錄**: 在側邊欄查看和重新下載歷史結果
        
        #### ⚙️ 技術參數
        
        - **圖像DPI**: 可調整，建議300-600
        - **語言支持**: 簡體中文、繁體中文、英文
        - **處理時間**: 根據文件大小和頁數而定
        - **文字保留**: 完全保留原始文字樣式和斷句
        
        #### 🔧 系統要求
        
        - 支持PDF格式文件
        - 建議文件大小不超過50MB
        - 處理時間與文件大小成正比
        """)

if __name__ == "__main__":
    main()
