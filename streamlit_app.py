#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Web應用 - 完全免費版本
支持Tesseract OCR引擎
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
    border: 2px solid #d0d0d0;
    border-radius: 8px;
    padding: 15px;
    background-color: #ffffff;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* 程式碼風格的識別結果顯示 */
.scrollable-container p {
    margin: 0;
    line-height: 1.2;
    font-family: 'Courier New', monospace;
    font-size: 14px;
}

.scrollable-container div {
    margin: 0;
    line-height: 1.2;
}

/* 程式碼風格的文本顯示 */
.code-style-text {
    font-family: 'Courier New', monospace;
    font-size: 14px;
    line-height: 1.2;
    margin: 0;
    padding: 10px;
    white-space: pre-wrap;
    background-color: #f8f9fa;
    border: 1px solid #e9ecef;
    border-radius: 4px;
    color: #333333;
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
        """簡化的圖像預處理 - 避免過度處理"""
        # 轉換為灰度
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image.copy()
        
        # 輕度去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, h=3, templateWindowSize=7, searchWindowSize=21)
        
        # 輕度對比度增強
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 返回增強後的圖像，不進行二值化
        return enhanced
    
    def extract_text(self, image: np.ndarray, use_preprocessing: bool = True, line_sensitivity: float = 0.8) -> list:
        """使用Tesseract提取文本 - 改善標點符號和標題識別"""
        try:
            # 根據參數決定是否預處理圖像
            if use_preprocessing:
                processed_image = self.preprocess_image(image)
            else:
                # 只轉換為灰度，不進行其他預處理
                if len(image.shape) == 3:
                    processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                else:
                    processed_image = image.copy()
            
            # 簡化的Tesseract配置 - 先測試基本功能
            configs = [
                # 配置1: 自動頁面分割，使用OEM 2混合模式
                '--psm 3 --oem 2',
                # 配置2: 單列文本塊
                '--psm 4 --oem 2',
                # 配置3: 單一文本塊
                '--psm 6 --oem 2',
                # 配置4: 單一文本行
                '--psm 7 --oem 2',
                # 配置5: 單一詞
                '--psm 8 --oem 2',
                # 配置6: 嘗試OEM 3 LSTM
                '--psm 3 --oem 3',
                # 配置7: 嘗試OEM 1 傳統模式
                '--psm 3 --oem 1'
            ]
            
            lang = 'chi_tra+chi_sim+eng'  # 繁體中文+簡體中文+英文
            best_result = ""
            best_confidence = 0
            
            # 嘗試多種配置，選擇最佳結果
            for i, config in enumerate(configs):
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
                    
                    # 記錄每個配置的結果
                    logger.info(f"配置 {i+1} ({config}): 置信度={avg_confidence:.2f}, 結果='{text_result[:50]}...'")
                    
                    if avg_confidence > best_confidence:
                        best_result = text_result
                        best_confidence = avg_confidence
                        best_data = data
                        logger.info(f"新的最佳配置: {config}, 置信度={avg_confidence:.2f}")
                        
                except Exception as e:
                    logger.warning(f"Tesseract配置 {config} 失敗: {e}")
                    continue
            
            if not best_result:
                return []
            
            # 後處理文本，改善標點符號識別
            processed_text = self._post_process_text(best_result)
            lines = processed_text.strip().split('\n')
            
            # 改進的文本塊創建 - 更準確的分行識別
            text_blocks = []
            
            # 首先收集所有有效的文本元素
            text_elements = []
            for i in range(len(best_data['text'])):
                text = best_data['text'][i].strip()
                confidence = int(best_data['conf'][i])
                x = best_data['left'][i]
                y = best_data['top'][i]
                width = best_data['width'][i]
                height = best_data['height'][i]
                
                if text and confidence > 30:
                    text_elements.append({
                        'text': text,
                        'confidence': confidence,
                        'x': x,
                        'y': y,
                        'width': width,
                        'height': height
                    })
            
            if not text_elements:
                return []
            
            # 按Y座標排序，然後按X座標排序
            text_elements.sort(key=lambda elem: (elem['y'], elem['x']))
            
            # 計算平均行高
            heights = [elem['height'] for elem in text_elements]
            avg_height = sum(heights) / len(heights) if heights else 20
            
            # 分行邏輯：Y座標差距超過平均行高的指定倍數就認為是新行
            line_threshold = avg_height * line_sensitivity
            
            current_line_elements = []
            current_y = None
            
            for elem in text_elements:
                if current_y is None or abs(elem['y'] - current_y) <= line_threshold:
                    # 同一行
                    current_line_elements.append(elem)
                    current_y = elem['y'] if current_y is None else min(current_y, elem['y'])
                else:
                    # 新行，處理當前行
                    if current_line_elements:
                        # 按X座標排序當前行
                        current_line_elements.sort(key=lambda e: e['x'])
                        
                        # 合併當前行文本
                        line_text = ""
                        line_confidence = 0
                        line_x = min(e['x'] for e in current_line_elements)
                        line_y = min(e['y'] for e in current_line_elements)
                        line_width = max(e['x'] + e['width'] for e in current_line_elements) - line_x
                        line_height = max(e['y'] + e['height'] for e in current_line_elements) - line_y
                        
                        for e in current_line_elements:
                            line_text += e['text']
                            line_confidence = max(line_confidence, e['confidence'])
                        
                        # 後處理文本
                        processed_text = self._post_process_text(line_text.strip())
                        
                        text_blocks.append({
                            "text": processed_text,
                            "confidence": line_confidence / 100.0,
                            "position": {
                                "x": line_x,
                                "y": line_y,
                                "width": line_width,
                                "height": line_height
                            }
                        })
                    
                    # 開始新行
                    current_line_elements = [elem]
                    current_y = elem['y']
            
            # 處理最後一行
            if current_line_elements:
                current_line_elements.sort(key=lambda e: e['x'])
                
                line_text = ""
                line_confidence = 0
                line_x = min(e['x'] for e in current_line_elements)
                line_y = min(e['y'] for e in current_line_elements)
                line_width = max(e['x'] + e['width'] for e in current_line_elements) - line_x
                line_height = max(e['y'] + e['height'] for e in current_line_elements) - line_y
                
                for e in current_line_elements:
                    line_text += e['text']
                    line_confidence = max(line_confidence, e['confidence'])
                
                processed_text = self._post_process_text(line_text.strip())
                
                text_blocks.append({
                    "text": processed_text,
                    "confidence": line_confidence / 100.0,
                    "position": {
                        "x": line_x,
                        "y": line_y,
                        "width": line_width,
                        "height": line_height
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
            # 如有需要 "x": "y"
        }
        
        # 應用替換
        for old, new in replacements.items():
            text = text.replace(old, new)
        
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
    


def process_pdf_with_ocr(pdf_path: str, ocr_engine: str, dpi: int = 300, use_preprocessing: bool = True, line_sensitivity: float = 0.8, progress_callback=None) -> dict:
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
            logger.info(f"開始處理第 {page_num} 頁，總共 {len(images)} 頁")
            
            if progress_callback:
                progress_callback(f"正在處理第 {page_num} 頁...", page_num, len(images))
            
            try:
                texts = ocr_processor.extract_text(image, use_preprocessing, line_sensitivity)
                logger.info(f"第 {page_num} 頁識別到 {len(texts)} 個文本塊")
                
                # 直接使用文本，保持原始格式
                page_result = {
                    "page_number": page_num,
                    "text_blocks": texts,
                    "full_text": "\n".join([block["text"] for block in texts])
                }
                result["pages"].append(page_result)
                
                logger.info(f"第 {page_num} 頁處理完成，已處理 {len(result['pages'])} 頁")
                
                # 每頁完成後立即回調，讓UI即時更新
                if progress_callback:
                    progress_callback(f"第 {page_num} 頁處理完成", page_num, len(images), result)
                    
            except Exception as e:
                logger.error(f"處理第 {page_num} 頁時發生錯誤: {e}")
                # 即使某頁失敗，也繼續處理下一頁
                page_result = {
                    "page_number": page_num,
                    "text_blocks": [],
                    "full_text": f"處理錯誤: {str(e)}"
                }
                result["pages"].append(page_result)
                
                if progress_callback:
                    progress_callback(f"第 {page_num} 頁處理失敗: {str(e)}", page_num, len(images), result)
        
        return result
        
    except Exception as e:
        return {"error": f"OCR處理失敗: {str(e)}"}


def display_comparison_view(result: dict, pdf_images=None):
    """顯示對比視窗 - 原文件與識別結果並排顯示"""
    st.markdown("### 🔍 原文件與識別結果對比")
    
    # 創建兩列布局
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### 📄 原文件")
        if pdf_images:
            # 使用Streamlit的expander來創建可折疊的嵌入視窗效果
            for i, image in enumerate(pdf_images):
                with st.expander(f"第 {i+1} 頁", expanded=(i==0)):  # 第一頁默認展開
                    st.image(image, use_column_width=True)
        else:
            st.info("原文件圖像預覽功能需要重新上傳文件")
    
    with col2:
        st.markdown("#### 📝 識別結果")
        
        if result["pages"]:
            for page in result["pages"]:
                with st.expander(f"第 {page['page_number']} 頁", expanded=(page['page_number']==1)):  # 第一頁默認展開
                    if page["text_blocks"]:
                        # 將所有文本塊合併成一個完整的文本
                        full_text = ""
                        for block in page["text_blocks"]:
                            full_text += block['text'] + "\n"
                        
                        # 按行分割並添加行號
                        lines = full_text.strip().split('\n')
                        numbered_text = ""
                        for i, line in enumerate(lines, 1):
                            if line.strip():  # 只顯示非空行
                                numbered_text += f"{i:3d} | {line}\n"
                        
                        # 使用Streamlit的code組件顯示
                        st.code(numbered_text, language=None)
                    else:
                        st.write("此頁面沒有識別到文本")
        else:
            st.info("尚未開始OCR處理或沒有識別結果")

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
        use_preprocessing = st.checkbox("使用圖像預處理", value=True, help="關閉此選項使用原始圖像進行OCR")
        line_sensitivity = st.slider("分行敏感度", 0.3, 1.5, 0.8, help="較低值會更嚴格地分行，較高值會更寬鬆地分行")
        
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
        
        # 立即轉換PDF為圖像並顯示
        if 'pdf_images' not in st.session_state or st.session_state.current_file != uploaded_file.name:
            with st.spinner("正在轉換PDF為圖像..."):
                all_images = pdf2image.convert_from_path(tmp_file_path, dpi=150)
                st.session_state.pdf_images = all_images
                st.session_state.current_file = uploaded_file.name
                st.session_state.processing_results = None  # 重置處理結果
        
        # 移除PDF預覽區塊
        
        # 處理按鈕
        if st.button("🚀 開始OCR處理", type="primary"):
            st.session_state.is_processing = True
            
            # 創建進度條和狀態顯示
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # 初始化處理結果
            st.session_state.processing_results = {
                "file_name": uploaded_file.name,
                "total_pages": len(st.session_state.pdf_images),
                "pages": [],
                "ocr_engine": "Tesseract"
            }
            
            # 定義進度回調函數 - 不立即rerun
            def progress_callback(message, current_page, total_pages, partial_result=None):
                progress = current_page / total_pages
                progress_bar.progress(progress)
                status_text.text(f"{message} ({current_page}/{total_pages})")
                
                # 更新處理結果
                if partial_result:
                    st.session_state.processing_results = partial_result
            
            # 處理文件
            result = process_pdf_with_ocr(tmp_file_path, ocr_engine, dpi, use_preprocessing, line_sensitivity, progress_callback)
            
            # 清理臨時文件
            os.unlink(tmp_file_path)
            
            if "error" in result:
                st.error(f"❌ {result['error']}")
                st.session_state.is_processing = False
            else:
                progress_bar.progress(100)
                status_text.text("處理完成！")
                st.success("✅ OCR處理完成！")
                
                # 保存最終結果
                st.session_state.processing_results = result
                
                # 添加到歷史記錄
                if result not in st.session_state.history:
                    st.session_state.history.insert(0, result)
                    # 限制歷史記錄數量
                    if len(st.session_state.history) > 10:
                        st.session_state.history = st.session_state.history[:10]
                
                st.session_state.is_processing = False
                st.rerun()
    
    # 如果有處理結果，顯示對比視窗（包括處理中）
    if st.session_state.processing_results and st.session_state.processing_results.get("pages"):
        # 顯示調試信息
        if st.session_state.is_processing:
            st.info(f"正在處理中... 已處理 {len(st.session_state.processing_results['pages'])} 頁，總共 {st.session_state.processing_results['total_pages']} 頁")
        
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
