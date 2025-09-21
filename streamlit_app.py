#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR Web應用 - 基於Streamlit的PDF OCR處理平台
"""

import streamlit as st
import os
import json
import tempfile
import zipfile
from pathlib import Path
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
# OCR處理器類別
import cv2
import numpy as np
from PIL import Image
import pdf2image
from paddleocr import PaddleOCR
import pytesseract
from typing import List, Dict, Any, Tuple
import logging
import re

# 設置日誌
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class OCRProcessor:
    def __init__(self):
        """初始化OCR處理器"""
        # 延遲初始化PaddleOCR，避免多執行緒衝突
        self.paddle_ocr = None
        self._paddle_initialized = False
        self._batch_size = 3  # 批次處理大小
        
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
        
        # 文本方向檢測參數
        self.vertical_threshold = 0.7  # 直式文本閾值
        
        # 文本分類規則
        self.text_classification_rules = {
            'title': {
                'keywords': ['標題', '題目', '主題', '名稱', '題', '第.*章', '第.*節'],
                'position_threshold': 0.1,  # 位於頁面頂部
                'font_size_threshold': 1.5  # 字體較大
            },
            'caption': {
                'keywords': ['圖', '表', '說明', '注釋', '備註', '圖.*說明', '表.*說明'],
                'position_threshold': 0.8,  # 位於頁面底部
                'font_size_threshold': 0.8  # 字體較小
            },
            'table': {
                'keywords': ['表', '表格', '統計', '數據', '項目', '內容'],
                'pattern': r'\|.*\|',  # 包含表格符號
                'line_count_threshold': 3  # 多行文本
            },
            'content': {
                'default': True  # 默認為內文
            }
        }
        
        # 執行緒鎖，確保PaddleOCR初始化是執行緒安全的
        import threading
        self._init_lock = threading.Lock()
    
    def _init_paddle_ocr(self):
        """延遲初始化PaddleOCR，確保執行緒安全"""
        if not self._paddle_initialized:
            with self._init_lock:
                if not self._paddle_initialized:
                    # 在Streamlit Cloud上完全禁用PaddleOCR，避免CPU指令集問題
                    logger.info("在Streamlit Cloud環境中，將只使用Tesseract進行OCR")
                    self.paddle_ocr = None
                    self._paddle_initialized = True
    
    def _cleanup_memory(self):
        """清理記憶體，避免累積過多"""
        import gc
        gc.collect()
        if hasattr(self, 'paddle_ocr') and self.paddle_ocr:
            # 清理PaddleOCR內部快取
            try:
                del self.paddle_ocr
                self.paddle_ocr = None
                self._paddle_initialized = False
            except:
                pass

    def process_single_page(self, page_data, enable_postprocess=True):
        """處理單一頁面 - 用於批次處理（避免Streamlit context問題）"""
        page_num, image, page_height = page_data
        
        # 確保PaddleOCR已初始化
        self._init_paddle_ocr()
        
        # 檢測每個文本塊的方向（圖像已預處理）
        text_blocks = self.detect_text_direction_per_block(image)
        
        # 如果PaddleOCR不可用，使用增強版Tesseract
        if self.paddle_ocr is None and not text_blocks:
            logger.info("使用增強版Tesseract進行OCR")
            # 使用增強版Tesseract
            tesseract_results = self.extract_text_tesseract_enhanced(image, "horizontal")
            
            # 轉換為text_blocks格式
            text_blocks = []
            for result in tesseract_results:
                width = result["position"]["width"]
                height = result["position"]["height"]
                
                if height > 0:
                    aspect_ratio = width / height
                    direction = "vertical" if aspect_ratio < self.vertical_threshold else "horizontal"
                else:
                    direction = "horizontal"
                
                text_blocks.append({
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "bbox": [],  # Tesseract不提供bbox
                    "direction": direction,
                    "position": result["position"]
                })
        
        # 分類每個文本塊
        classified_blocks = []
        
        for block in text_blocks:
            # 文字後處理（辭典糾錯/正規化）
            clean_text = self.correct_text(block["text"]) if enable_postprocess else block["text"]
            
            # 分類文本類型
            text_type = self.classify_text_type(block, page_height)
            
            # 簡化的文本塊結構
            simplified_block = {
                "text": clean_text,
                "type": text_type,
                "direction": block["direction"],
                "confidence": block["confidence"]
            }
            classified_blocks.append(simplified_block)
        
        # 按閱讀順序組織
        if classified_blocks:
            if any(block["direction"] == "vertical" for block in classified_blocks):
                # 混合方向，按位置排序
                organized_texts = sorted(classified_blocks, 
                                       key=lambda x: (x.get("position", {}).get("y", 0), 
                                                    x.get("position", {}).get("x", 0)))
            else:
                # 純橫式，按行排序
                organized_texts = sorted(classified_blocks, 
                                       key=lambda x: (x.get("position", {}).get("y", 0), 
                                                    x.get("position", {}).get("x", 0)))
        else:
            organized_texts = []
        
        # 構建頁面結果
        page_result = {
            "page_number": page_num,
            "text_blocks": organized_texts,
            "full_text": "\n".join([block["text"] for block in organized_texts])
        }
        
        return page_result, organized_texts
        
    def pdf_to_images(self, pdf_path: str, dpi: int = 300) -> List[np.ndarray]:
        """將PDF轉換為高質量圖像"""
        logger.info(f"正在轉換PDF: {pdf_path}")
        try:
            # 使用更高的DPI和更好的轉換參數
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
    
    def detect_text_direction_per_block(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """檢測每個文本塊的方向 (直式/橫式)"""
        # 如果PaddleOCR不可用，使用Tesseract
        if self.paddle_ocr is None:
            return self._detect_text_direction_tesseract(image)
        
        try:
            # 使用PaddleOCR檢測文本方向
            result = self.paddle_ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return self._detect_text_direction_tesseract(image)
            
            text_blocks = []
            for line in result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # 計算文本框的長寬比和角度
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    width = max(x_coords) - min(x_coords)
                    height = max(y_coords) - min(y_coords)
                    
                    # 判斷方向
                    if height > 0:
                        aspect_ratio = width / height
                        direction = "vertical" if aspect_ratio < self.vertical_threshold else "horizontal"
                    else:
                        direction = "horizontal"
                    
                    text_blocks.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "direction": direction,
                        "position": {
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "width": width,
                            "height": height
                        }
                    })
            
            return text_blocks
        except Exception as e:
            logger.error(f"PaddleOCR檢測失敗: {e}")
            return self._detect_text_direction_tesseract(image)
    
    def _detect_text_direction_tesseract(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用Tesseract檢測文本方向"""
        try:
            # 使用Tesseract檢測文本
            tesseract_results = self.extract_text_tesseract(image, "horizontal")
            
            text_blocks = []
            for result in tesseract_results:
                # 簡單的方向判斷：基於文本框的長寬比
                width = result["position"]["width"]
                height = result["position"]["height"]
                
                if height > 0:
                    aspect_ratio = width / height
                    direction = "vertical" if aspect_ratio < self.vertical_threshold else "horizontal"
                else:
                    direction = "horizontal"
                
                text_blocks.append({
                    "text": result["text"],
                    "confidence": result["confidence"],
                    "bbox": [],  # Tesseract不提供bbox
                    "direction": direction,
                    "position": result["position"]
                })
            
            return text_blocks
        except Exception as e:
            logger.error(f"Tesseract方向檢測失敗: {e}")
            return []
    
    def classify_text_type(self, text_block: Dict[str, Any], page_height: int) -> str:
        """分類文本類型 (標題、內文、表格、圖片說明等)"""
        text = text_block["text"]
        position = text_block["position"]
        confidence = text_block["confidence"]
        
        # 計算相對位置
        relative_y = position["y"] / page_height if page_height > 0 else 0
        
        # 計算字體大小 (基於文本框高度)
        font_size = position["height"]
        
        # 檢查標題
        title_rules = self.text_classification_rules['title']
        if (relative_y < title_rules['position_threshold'] and 
            font_size > title_rules['font_size_threshold'] * 20):  # 20是基準字體大小
            for keyword in title_rules['keywords']:
                if re.search(keyword, text):
                    return "title"
            # 如果位置和字體符合標題特徵，也歸為標題
            if relative_y < 0.2 and font_size > 30:
                return "title"
        
        # 檢查圖片說明
        caption_rules = self.text_classification_rules['caption']
        if (relative_y > caption_rules['position_threshold'] and 
            font_size < caption_rules['font_size_threshold'] * 20):
            for keyword in caption_rules['keywords']:
                if re.search(keyword, text):
                    return "caption"
        
        # 檢查表格
        table_rules = self.text_classification_rules['table']
        if re.search(table_rules['pattern'], text):
            return "table"
        
        # 檢查是否包含表格關鍵詞
        for keyword in table_rules['keywords']:
            if re.search(keyword, text):
                return "table"
        
        # 默認為內文
        return "content"
    
    def preprocess_image(self, image: np.ndarray, direction: str, scale: float = 1.0) -> np.ndarray:
        """Chrome級別圖像預處理（含可選超解析）"""
        # 0. 可選超解析（高質量版）：LANCZOS4插值 + 銳化
        if scale and scale > 1.0:
            new_w = int(image.shape[1] * scale)
            new_h = int(image.shape[0] * scale)
            image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
            # 使用更強的銳化核
            kernel = np.array([[-1,-1,-1], [-1,12,-1], [-1,-1,-1]])
            image = cv2.filter2D(image, -1, kernel)
        
        # 轉換為灰度圖
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # 1. 高級去噪 - 使用非局部均值去噪
        denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
        
        # 2. 對比度增強 - 使用更強的CLAHE
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
        enhanced = clahe.apply(denoised)
        
        # 3. 銳化 - 使用更強的銳化核
        kernel = np.array([[-1,-1,-1], [-1,12,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, kernel)
        
        # 4. 邊緣增強
        edges = cv2.Canny(sharpened, 50, 150)
        sharpened = cv2.addWeighted(sharpened, 0.8, edges, 0.2, 0)
        
        # 5. 自適應二值化 - 使用更精確的參數
        binary = cv2.adaptiveThreshold(
            sharpened, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY, 15, 3
        )
        
        # 6. 形態學操作 - 更精細的處理
        # 先閉運算連接文字
        kernel_close = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel_close)
        
        # 再開運算去除噪點
        kernel_open = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
        binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel_open)
        
        # 7. 如果是直式文本，嘗試旋轉
        if direction == "vertical":
            binary = self.rotate_if_needed(binary)
        
        return binary

    def correct_text(self, text: str) -> str:
        """字元正規化與辭典式糾錯（繁體中文優化版）"""
        if not text:
            return text
        # 1) Unicode 正規化
        import unicodedata, re as _re
        t = unicodedata.normalize('NFKC', text)
        
        # 2) 常見OCR錯誤修正（繁體中文）
        substitutions = {
            '。·': '。', '·。': '。', '…': '…', '—': '-', '–': '-',
            '圖書禮卷': '圖書禮券', '禮卷': '禮券',
            '○': '○',
            # 繁體中文常見OCR錯誤
            '部暑': '部署', '访問': '訪問', '准': '準備', '食库': '倉庫',
            '防間': '訪問', '帐號': '帳號', '點擎': '點擊', '選': '選擇',
            '設置': '設置', '等待': '等待', '獲得': '獲得', '访問': '訪問',
            '選頂': '選項', '推蔗': '推薦', '防周': '訪問', '盒库': '倉庫',
            '安装': '安裝', '創建': '創建', '注意事頂': '注意事項',
            '免费': '免費', '建上傅': '建議上傳', '较長': '較長',
            '發限制': '使用限制', '查': '檢查', '检查': '檢查',
            '下载': '下載', '查': '檢查', '日翰出': '日誌輸出',
            '比较': '比較', '额度': '額度', '付費': '付費', '推度': '推薦',
            '應用': '應用', '限制': '限制', '月': '月'
        }
        for a, b in substitutions.items():
            t = t.replace(a, b)
        
        # 3) 數字/英文字混淆
        t = _re.sub(r'(?<=\d)[Oo](?=\d)', '0', t)
        t = _re.sub(r'(?<=\d)[lI](?=\d)', '1', t)
        
        # 4) 移除法文和亂碼（常見OCR錯誤）
        t = _re.sub(r'[A-Z\s]+[A-Z][A-Z\s]+', '', t)  # 移除法文模式
        t = _re.sub(r'BTS.*?EIFFEL', '', t)  # 移除法文學校名
        t = _re.sub(r'PrOFesseur.*?rOS', '', t)  # 移除法文教授名
        t = _re.sub(r'\d+/\d+', '', t)  # 移除分數格式
        
        # 5) 清理多餘空格和換行
        t = _re.sub(r'\s+', ' ', t).strip()
        
        # 6) 簡轉繁（若可用且啟用）
        if st.session_state.get('enable_text_postprocess', True):
            try:
                from opencc import OpenCC  # type: ignore
                t = OpenCC('s2t').convert(t)
            except Exception:
                pass
        return t
    
    def rotate_if_needed(self, image: np.ndarray) -> np.ndarray:
        """檢測並旋轉圖像"""
        # 使用Tesseract檢測角度
        try:
            osd = pytesseract.image_to_osd(image, output_type=pytesseract.Output.DICT)
            angle = osd['rotate']
            if angle != 0:
                # 旋轉圖像
                (h, w) = image.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, angle, 1.0)
                rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
                return rotated
        except:
            pass
        
        return image
    
    def extract_text_paddle(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """使用PaddleOCR提取文本"""
        if self.paddle_ocr is None:
            return []
        
        try:
            result = self.paddle_ocr.ocr(image, cls=True)
            
            if not result or not result[0]:
                return []
            
            extracted_texts = []
            for line in result[0]:
                if len(line) >= 2:
                    bbox = line[0]
                    text = line[1][0]
                    confidence = line[1][1]
                    
                    # 計算文本位置
                    x_coords = [point[0] for point in bbox]
                    y_coords = [point[1] for point in bbox]
                    
                    extracted_texts.append({
                        "text": text,
                        "confidence": confidence,
                        "bbox": bbox,
                        "position": {
                            "x": min(x_coords),
                            "y": min(y_coords),
                            "width": max(x_coords) - min(x_coords),
                            "height": max(y_coords) - min(y_coords)
                        }
                    })
            
            return extracted_texts
        except Exception as e:
            logger.error(f"PaddleOCR文本提取失敗: {e}")
            return []
    
    def extract_text_tesseract(self, image: np.ndarray, direction: str) -> List[Dict[str, Any]]:
        """使用高配置Tesseract提取文本 - 繁體中文優化"""
        # 設置語言和配置 - 優先繁體中文
        lang = 'chi_tra+chi_sim+eng'  # 繁體中文+簡體中文+英文
        
        # 使用更精確的PSM模式
        if direction == "vertical":
            # 直式文本配置 - 使用單行文本模式
            config = '--psm 7 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千萬億零壹貳參肆伍陸柒捌玖拾佰仟萬億'
        else:
            # 橫式文本配置 - 使用自動頁面分割模式
            config = '--psm 1 -c preserve_interword_spaces=1 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz一二三四五六七八九十百千萬億零壹貳參肆伍陸柒捌玖拾佰仟萬億'
        
        try:
            # 獲取詳細信息
            data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
            
            extracted_texts = []
            n_boxes = len(data['text'])
            
            for i in range(n_boxes):
                text = data['text'][i].strip()
                confidence = int(data['conf'][i])
                
                # 大幅降低置信度閾值，檢測更多文字
                if text and confidence > 10 and len(text) > 0:
                    # 過濾明顯的噪點
                    if len(text) == 1 and confidence < 30:
                        continue
                    
                    extracted_texts.append({
                        "text": text,
                        "confidence": confidence / 100.0,
                        "position": {
                            "x": data['left'][i],
                            "y": data['top'][i],
                            "width": data['width'][i],
                            "height": data['height'][i]
                        }
                    })
            
            return extracted_texts
        except Exception as e:
            logger.error(f"Tesseract OCR失敗: {e}")
            return []
    
    def extract_text_tesseract_enhanced(self, image: np.ndarray, direction: str) -> List[Dict[str, Any]]:
        """增強版Tesseract OCR - 使用多種配置嘗試"""
        results = []
        
        # 嘗試多種配置
        configs = [
            # 配置1：標準配置
            '--psm 1 -c preserve_interword_spaces=1',
            # 配置2：單列文本
            '--psm 4 -c preserve_interword_spaces=1',
            # 配置3：單行文本
            '--psm 7 -c preserve_interword_spaces=1',
            # 配置4：單詞
            '--psm 8 -c preserve_interword_spaces=1'
        ]
        
        lang = 'chi_tra+chi_sim+eng'
        
        for config in configs:
            try:
                data = pytesseract.image_to_data(image, lang=lang, config=config, output_type=pytesseract.Output.DICT)
                
                for i in range(len(data['text'])):
                    text = data['text'][i].strip()
                    confidence = int(data['conf'][i])
                    
                    if text and confidence > 20 and len(text) > 0:
                        # 檢查是否與現有結果重疊
                        is_duplicate = False
                        for existing in results:
                            if self._texts_overlap_simple(existing, {
                                "x": data['left'][i],
                                "y": data['top'][i],
                                "width": data['width'][i],
                                "height": data['height'][i]
                            }):
                                is_duplicate = True
                                break
                        
                        if not is_duplicate:
                            results.append({
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
        
        return results
    
    def _texts_overlap_simple(self, text1: Dict, pos2: Dict) -> bool:
        """簡單的重疊檢測"""
        pos1 = text1["position"]
        
        # 簡單的重疊檢測
        overlap_x = not (pos1["x"] + pos1["width"] < pos2["x"] or pos2["x"] + pos2["width"] < pos1["x"])
        overlap_y = not (pos1["y"] + pos1["height"] < pos2["y"] or pos2["y"] + pos2["height"] < pos1["y"])
        
        return overlap_x and overlap_y
    
    def merge_ocr_results(self, paddle_results: List[Dict], tesseract_results: List[Dict]) -> List[Dict[str, Any]]:
        """合併PaddleOCR和Tesseract的結果"""
        merged = []
        
        # 優先使用PaddleOCR結果（對中文支持更好）
        for result in paddle_results:
            merged.append({
                "text": result["text"],
                "confidence": result["confidence"],
                "position": result["position"],
                "source": "paddle"
            })
        
        # 補充Tesseract結果（如果PaddleOCR沒有識別到）
        for t_result in tesseract_results:
            # 檢查是否與現有結果重疊
            is_duplicate = False
            for m_result in merged:
                if self.texts_overlap(t_result, m_result):
                    is_duplicate = True
                    break
            
            if not is_duplicate and t_result["text"].strip():
                merged.append({
                    "text": t_result["text"],
                    "confidence": t_result["confidence"],
                    "position": t_result["position"],
                    "source": "tesseract"
                })
        
        return merged
    
    def texts_overlap(self, text1: Dict, text2: Dict) -> bool:
        """檢查兩個文本是否重疊"""
        pos1 = text1["position"]
        pos2 = text2["position"]
        
        # 簡單的重疊檢測
        overlap_x = not (pos1["x"] + pos1["width"] < pos2["x"] or pos2["x"] + pos2["width"] < pos1["x"])
        overlap_y = not (pos1["y"] + pos1["height"] < pos2["y"] or pos2["y"] + pos2["height"] < pos1["y"])
        
        return overlap_x and overlap_y
    
    def organize_text_by_reading_order(self, texts: List[Dict], direction: str) -> List[Dict[str, Any]]:
        """按閱讀順序組織文本"""
        if direction == "vertical":
            # 直式文本：從右到左，從上到下
            sorted_texts = sorted(texts, key=lambda x: (-x["position"]["x"], x["position"]["y"]))
        else:
            # 橫式文本：從左到右，從上到下
            sorted_texts = sorted(texts, key=lambda x: (x["position"]["y"], x["position"]["x"]))
        
        return sorted_texts
import pandas as pd

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
        font-size: 1.2rem;
        text-align: center;
        margin-bottom: 2rem;
        color: #666;
    }
    .success-box {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .info-box {
        background-color: #d1ecf1;
        border: 1px solid #bee5eb;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .progress-container {
        background-color: #f8f9fa;
        border-radius: 0.375rem;
        padding: 1rem;
        margin: 1rem 0;
    }
    .file-info {
        background-color: #e9ecef;
        border-radius: 0.375rem;
        padding: 0.75rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

def init_session_state():
    """初始化session state"""
    if 'ocr_processor' not in st.session_state:
        st.session_state.ocr_processor = None
    if 'processing_results' not in st.session_state:
        st.session_state.processing_results = {}
    if 'current_file' not in st.session_state:
        st.session_state.current_file = None
    if 'is_processing' not in st.session_state:
        st.session_state.is_processing = False
    if 'stop_processing' not in st.session_state:
        st.session_state.stop_processing = False

def load_ocr_processor():
    """載入OCR處理器"""
    if st.session_state.ocr_processor is None:
        with st.spinner("正在初始化OCR引擎..."):
            try:
                st.session_state.ocr_processor = OCRProcessor()
                st.success("OCR引擎初始化完成！")
            except Exception as e:
                st.error(f"OCR引擎初始化失敗: {str(e)}")
                return False
    return True

def process_pdf_file(uploaded_file, progress_bar, status_text, realtime_container):
    """處理上傳的PDF文件 - 支持即時更新和暫停/停止"""
    try:
        # 保存上傳的文件到臨時目錄
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name
        
        # 更新狀態
        status_text.text("正在轉換PDF為圖像...")
        progress_bar.progress(10)
        
        # 轉換PDF為圖像 - 添加進度更新
        try:
            # 使用預設DPI或從session state獲取
            dpi = st.session_state.get('dpi', 300)
            images = st.session_state.ocr_processor.pdf_to_images(tmp_file_path, dpi=dpi)
            if not images:
                return None, "PDF轉換失敗"
            
            # 更新進度
            progress_bar.progress(20)
            status_text.text(f"PDF轉換完成，共 {len(images)} 頁，開始OCR處理...")
            
        except Exception as e:
            logger.error(f"PDF轉換錯誤: {e}")
            return None, f"PDF轉換失敗: {str(e)}"
        
        total_pages = len(images)
        result = {
            "file_name": uploaded_file.name,
            "total_pages": total_pages,
            "pages": []
        }
        
        # 開始處理
        status_text.text("開始處理PDF頁面...")
        
        # 準備頁面數據 - 預處理圖像
        page_data_list = []
        for page_num, image in enumerate(images, 1):
            page_height = image.shape[0]
            # 預先應用超解析預處理
            processed_image = st.session_state.ocr_processor.preprocess_image(
                image, "horizontal", scale=st.session_state.upscale_factor
            )
            page_data_list.append((page_num, processed_image, page_height))
        
        # 使用批次處理，平衡速度和穩定性
        completed_pages = 0
        enable_postprocess = st.session_state.get('enable_text_postprocess', True)
        batch_size = st.session_state.get('batch_size', 3)
        
        # 初始化累積文字
        if 'accumulated_text' not in st.session_state:
            st.session_state.accumulated_text = ""
        
        # 批次處理頁面
        for i in range(0, len(page_data_list), batch_size):
            # 檢查是否被停止
            if st.session_state.get('stop_processing', False):
                status_text.text("處理已停止")
                # 保存部分結果
                result["pages"].sort(key=lambda x: x["page_number"])
                return result, "處理被用戶停止"
            
            # 獲取當前批次的頁面
            batch_pages = page_data_list[i:i + batch_size]
            batch_results = []
            
            # 處理當前批次
            for page_data in batch_pages:
                page_num, image, page_height = page_data
                
                try:
                    # 處理單一頁面
                    page_result, organized_texts = st.session_state.ocr_processor.process_single_page(page_data, enable_postprocess)
                    batch_results.append((page_result, organized_texts, page_num))
                    
                except Exception as e:
                    logger.error(f"處理第 {page_num} 頁時發生錯誤: {e}")
                    continue
            
            # 批次完成後更新結果
            for page_result, organized_texts, page_num in batch_results:
                result["pages"].append(page_result)
                completed_pages += 1
                
                # 添加當前頁面的文字到累積文字
                page_text = f"\n=== 第 {page_num} 頁 ===\n"
                for block in organized_texts:
                    page_text += f"[{block['type']}] {block['text']}\n"
                st.session_state.accumulated_text += page_text
                
                # 即時顯示累積結果
                title_count = sum(1 for block in organized_texts if block["type"] == "title")
                content_count = sum(1 for block in organized_texts if block["type"] == "content")
                other_count = len(organized_texts) - title_count - content_count
                stats_line = f"已完成: {completed_pages}/{total_pages} 頁｜當前頁文本塊: {len(organized_texts)}｜標題: {title_count}｜內文: {content_count}｜其他: {other_count}"
                
                # 在容器中顯示統計和文字
                with realtime_container:
                    st.write(stats_line)
                    st.text_area("即時文字結果", st.session_state.accumulated_text, height=300, key=f"realtime_text_{completed_pages}", label_visibility="collapsed")
                
                # 更新進度條 - 顯示具體頁數
                progress = 10 + (completed_pages / total_pages) * 80
                progress_bar.progress(int(progress))
                status_text.text(f"正在處理第 {completed_pages}/{total_pages} 頁...")
            
            # 批次完成後清理記憶體
            if completed_pages % (batch_size * 2) == 0:  # 每處理2個批次就清理一次
                st.session_state.ocr_processor._cleanup_memory()
                status_text.text(f"記憶體清理完成，繼續處理第 {completed_pages + 1}/{total_pages} 頁...")
                
                # 短暫暫停，讓記憶體完全釋放
                import time
                time.sleep(0.5)
        
        # 按頁碼排序結果
        result["pages"].sort(key=lambda x: x["page_number"])
        
        # 清理臨時文件
        os.unlink(tmp_file_path)
        
        status_text.text("處理完成！")
        progress_bar.progress(100)
        
        return result, None
        
    except Exception as e:
        return None, f"處理過程中發生錯誤: {str(e)}"

def display_results(result):
    """顯示處理結果"""
    st.markdown("### 📊 處理結果概覽")
    
    # 計算統計數據
    total_pages = result["total_pages"]
    total_text_blocks = sum(len(page["text_blocks"]) for page in result["pages"])
    
    # 統計文本類型
    title_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "title") for page in result["pages"])
    content_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "content") for page in result["pages"])
    caption_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "caption") for page in result["pages"])
    table_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "table") for page in result["pages"])
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("總頁數", total_pages)
    
    with col2:
        st.metric("文本塊總數", total_text_blocks)
    
    with col3:
        st.metric("標題數量", title_count)
    
    with col4:
        st.metric("內文數量", content_count)
    
    # 文本類型分佈
    st.markdown("### 📈 文本類型分佈")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("標題", title_count)
    with col2:
        st.metric("內文", content_count)
    with col3:
        st.metric("圖片說明", caption_count)
    with col4:
        st.metric("表格", table_count)
    
    # 頁面詳細信息
    st.markdown("### 📄 頁面詳細信息")
    
    # 創建頁面選擇器
    page_options = [f"第 {page['page_number']} 頁" for page in result["pages"]]
    selected_page_idx = st.selectbox("選擇要查看的頁面:", range(len(page_options)), format_func=lambda x: page_options[x], key="page_selector")
    
    if selected_page_idx is not None:
        selected_page = result["pages"][selected_page_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("#### 頁面信息")
            st.write(f"**頁碼:** {selected_page['page_number']}")
            st.write(f"**文本塊數量:** {len(selected_page['text_blocks'])}")
            
            # 文本類型統計
            if selected_page["text_blocks"]:
                types = [block["type"] for block in selected_page["text_blocks"]]
                type_counts = pd.Series(types).value_counts()
                
                st.markdown("#### 文本類型統計")
                for text_type, count in type_counts.items():
                    st.write(f"**{text_type}:** {count} 個")
        
        with col2:
            st.markdown("#### 分類文本內容")
            if selected_page["text_blocks"]:
                # 按類型分組顯示
                for text_type in ["title", "content", "table", "caption"]:
                    type_blocks = [block for block in selected_page["text_blocks"] if block["type"] == text_type]
                    if type_blocks:
                        st.markdown(f"**{text_type.upper()}:**")
                        for block in type_blocks:
                            st.write(f"- {block['text']}")
                        st.write("")
            else:
                st.write("此頁面沒有識別到文本")

def create_download_links(result):
    """創建下載鏈接"""
    st.markdown("### 💾 下載結果")
    
    # 簡化的JSON下載 (只包含文本內容和性質)
    simplified_result = {
        "file_name": result["file_name"],
        "total_pages": result["total_pages"],
        "pages": []
    }
    
    for page in result["pages"]:
        simplified_page = {
            "page_number": page["page_number"],
            "text_blocks": [
                {
                    "text": block["text"],
                    "type": block["type"],
                    "direction": block["direction"]
                }
                for block in page["text_blocks"]
            ]
        }
        simplified_result["pages"].append(simplified_page)
    
    json_data = json.dumps(simplified_result, ensure_ascii=False, indent=2)
    st.download_button(
        label="📄 下載簡化JSON文件",
        data=json_data,
        file_name=f"{result['file_name']}_ocr.json",
        mime="application/json",
        key="download_json"
    )
    
    # 按類型分類的文本下載
    classified_text = ""
    for page in result["pages"]:
        classified_text += f"=== 第 {page['page_number']} 頁 ===\n\n"
        
        for text_type in ["title", "content", "table", "caption"]:
            type_blocks = [block for block in page["text_blocks"] if block["type"] == text_type]
            if type_blocks:
                classified_text += f"【{text_type.upper()}】\n"
                for block in type_blocks:
                    classified_text += f"{block['text']}\n"
                classified_text += "\n"
        
        classified_text += "\n"
    
    st.download_button(
        label="📝 下載分類文本文件",
        data=classified_text,
        file_name=f"{result['file_name']}_classified.txt",
        mime="text/plain",
        key="download_classified"
    )
    
    # 純文本下載
    full_text = "\n\n".join([f"=== 第 {page['page_number']} 頁 ===" + "\n" + page['full_text'] for page in result["pages"]])
    st.download_button(
        label="📄 下載純文本文件",
        data=full_text,
        file_name=f"{result['file_name']}_text.txt",
        mime="text/plain",
        key="download_text"
    )
    
    # 統計報告下載
    title_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "title") for page in result["pages"])
    content_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "content") for page in result["pages"])
    caption_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "caption") for page in result["pages"])
    table_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "table") for page in result["pages"])
    
    stats = {
        "file_name": result["file_name"],
        "total_pages": result["total_pages"],
        "total_text_blocks": sum(len(page["text_blocks"]) for page in result["pages"]),
        "text_type_distribution": {
            "title": title_count,
            "content": content_count,
            "caption": caption_count,
            "table": table_count
        },
        "pages_detail": [
            {
                "page_number": page["page_number"],
                "text_blocks_count": len(page["text_blocks"]),
                "text_length": len(page["full_text"]),
                "text_types": {
                    text_type: sum(1 for block in page["text_blocks"] if block["type"] == text_type)
                    for text_type in ["title", "content", "table", "caption"]
                }
            }
            for page in result["pages"]
        ]
    }
    
    stats_json = json.dumps(stats, ensure_ascii=False, indent=2)
    st.download_button(
        label="📊 下載統計報告",
        data=stats_json,
        file_name=f"{result['file_name']}_stats.json",
        mime="application/json",
        key="download_stats"
    )

def main():
    """主函數"""
    init_session_state()
    
    # 標題
    st.markdown('<h1 class="main-header">📄 OCR文本識別系統</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">支持中文直式/橫式文本識別，生成結構化JSON數據</p>', unsafe_allow_html=True)
    
    # 側邊欄
    with st.sidebar:
        st.markdown("## 🔧 系統設置")
        
        # OCR引擎狀態
        if st.session_state.ocr_processor is not None:
            st.success("✅ OCR引擎已就緒")
        else:
            st.warning("⚠️ OCR引擎未初始化")
        
        # 處理參數
        st.markdown("### 處理參數")
        dpi = st.slider("圖像DPI", 150, 600, 500, help="更高的DPI會提高識別精度但處理時間更長")
        st.session_state.upscale_factor = st.slider("超解析倍率", 1.0, 3.0, 2.5, 0.1, help=">1 會先放大影像再OCR，細字更清晰")
        st.session_state.batch_size = st.slider("批次處理大小", 1, 5, 3, help="每批處理的頁數，較大值速度更快但記憶體使用更多")
        st.session_state.enable_text_postprocess = st.toggle("啟用文字後處理（正規化/辭典）", value=True, help="包含NFKC、數字/英字修正、可用時簡轉繁")
        
        # 系統信息
        st.markdown("### ℹ️ 系統信息")
        st.write("**支持格式:** PDF")
        st.write("**支持語言:** 中文（簡體/繁體）")
        st.write("**文本方向:** 直式/橫式")
        st.write("**OCR引擎:** Tesseract (增強版)")
        st.info("ℹ️ 在Streamlit Cloud環境中，使用增強版Tesseract進行OCR處理")
        
        # 清除狀態按鈕
        st.markdown("### 🔧 系統控制")
        if st.button("🗑️ 清除所有狀態", type="secondary"):
            # 清除所有session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
    
    # 主要內容區域
    tab1, tab2, tab3 = st.tabs(["📤 上傳處理", "📊 結果查看", "📚 使用說明"])
    
    with tab1:
        st.markdown("### 📤 上傳PDF文件")
        
        # 文件上傳
        uploaded_file = st.file_uploader(
            "選擇PDF文件",
            type=['pdf'],
            help="支持中文直式和橫式文本的PDF文件"
        )
        
        if uploaded_file is not None:
            # 顯示文件信息
            st.markdown('<div class="file-info">', unsafe_allow_html=True)
            st.write(f"**文件名:** {uploaded_file.name}")
            st.write(f"**文件大小:** {uploaded_file.size / 1024 / 1024:.2f} MB")
            st.markdown('</div>', unsafe_allow_html=True)
            
            # 處理控制按鈕
            col1, col2, col3 = st.columns([1, 1, 2])
            
            with col1:
                if not st.session_state.is_processing:
                    if st.button("🚀 開始OCR處理", type="primary"):
                        # 初始化OCR引擎
                        if not load_ocr_processor():
                            st.stop()
                        
                        # 設置處理狀態
                        st.session_state.is_processing = True
                        st.session_state.stop_processing = False
                        # 重置累積文字
                        st.session_state.accumulated_text = ""
                        st.rerun()
                else:
                    st.button("⏸️ 處理中...", disabled=True)
            
            with col2:
                if st.session_state.is_processing:
                    if st.button("⏹️ 停止處理", type="secondary", key="stop_processing_btn"):
                        st.session_state.stop_processing = True
                        st.session_state.is_processing = False
                        st.rerun()
            
            # 處理進度和即時結果
            if st.session_state.is_processing:
                # 創建進度條和狀態顯示（唯一placeholder，避免重複）
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                # 即時結果顯示區域 - 使用可滾動容器
                st.markdown("### 📊 即時處理結果")
                realtime_container = st.container()
                
                # 處理文件
                result, error = process_pdf_file(uploaded_file, progress_bar, status_text, realtime_container)
                
                # 處理完成
                st.session_state.is_processing = False
                
                if error:
                    if "處理被用戶停止" in error:
                        st.warning("⚠️ 處理已停止")
                        
                        # 如果有部分結果，提供選項
                        if result["pages"]:
                            st.session_state.processing_results[uploaded_file.name] = result
                            st.session_state.current_file = uploaded_file.name
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if st.button("🔄 重新處理", type="primary", key="restart_processing"):
                                    # 重置所有處理狀態
                                    st.session_state.stop_processing = False
                                    st.session_state.is_processing = False
                                    st.session_state.accumulated_text = ""
                                    st.rerun()
                            with col2:
                                if st.button("📥 匯出部分結果", type="secondary", key="export_partial"):
                                    st.session_state.stop_processing = False
                                    st.session_state.is_processing = False
                                    st.rerun()
                            with col3:
                                if st.button("⏭️ 繼續處理", type="secondary", key="continue_processing"):
                                    st.session_state.stop_processing = False
                                    st.session_state.is_processing = True
                                    st.rerun()
                            
                            # 顯示部分結果
                            st.markdown("### 📊 部分處理結果")
                            display_results(result)
                            create_download_links(result)
                        else:
                            st.info("沒有處理任何頁面，請重新開始處理。")
                            # 重置狀態
                            st.session_state.stop_processing = False
                            st.session_state.is_processing = False
                    else:
                        st.error(f"❌ {error}")
                        # 重置狀態
                        st.session_state.stop_processing = False
                        st.session_state.is_processing = False
                else:
                    st.session_state.stop_processing = False
                    st.session_state.is_processing = False
                    st.success("✅ 處理完成！")
                    
                    # 保存結果到session state
                    st.session_state.processing_results[uploaded_file.name] = result
                    st.session_state.current_file = uploaded_file.name
                    
                    # 顯示最終結果
                    display_results(result)
                    create_download_links(result)
    
    with tab2:
        st.markdown("### 📊 查看處理結果")
        
        if st.session_state.processing_results:
            # 文件選擇器
            file_names = list(st.session_state.processing_results.keys())
            selected_file = st.selectbox("選擇已處理的文件:", file_names, key="file_selector")
            
            if selected_file:
                result = st.session_state.processing_results[selected_file]
                
                # 簡化的結果概覽
                st.markdown("### 📈 結果概覽")
                col1, col2, col3, col4 = st.columns(4)
                
                total_pages = result["total_pages"]
                total_text_blocks = sum(len(page["text_blocks"]) for page in result["pages"])
                title_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "title") for page in result["pages"])
                content_count = sum(sum(1 for block in page["text_blocks"] if block["type"] == "content") for page in result["pages"])
                
                with col1:
                    st.metric("總頁數", total_pages)
                with col2:
                    st.metric("文本塊總數", total_text_blocks)
                with col3:
                    st.metric("標題數量", title_count)
                with col4:
                    st.metric("內文數量", content_count)
                
                # 預覽區域 - 使用可滾動容器
                st.markdown("### 👀 內容預覽")
                
                # 創建預覽選項
                preview_type = st.radio("預覽類型:", ["JSON格式", "分類文本", "純文本"], horizontal=True, key="preview_type")
                
                # 使用容器創建可滾動區域
                with st.container():
                    if preview_type == "JSON格式":
                        # 完整的JSON預覽
                        simplified_result = {
                            "file_name": result["file_name"],
                            "total_pages": result["total_pages"],
                            "pages": []
                        }
                        
                        for page in result["pages"]:
                            simplified_page = {
                                "page_number": page["page_number"],
                                "text_blocks": [
                                    {
                                        "text": block["text"],
                                        "type": block["type"],
                                        "direction": block["direction"]
                                    }
                                    for block in page["text_blocks"]
                                ]
                            }
                            simplified_result["pages"].append(simplified_page)
                        
                        # 使用可滾動的JSON顯示
                        st.json(simplified_result)
                    
                    elif preview_type == "分類文本":
                        # 完整的分類文本預覽
                        preview_text = ""
                        for page in result["pages"]:
                            preview_text += f"=== 第 {page['page_number']} 頁 ===\n"
                            for text_type in ["title", "content", "table", "caption"]:
                                type_blocks = [block for block in page["text_blocks"] if block["type"] == text_type]
                                if type_blocks:
                                    preview_text += f"【{text_type.upper()}】\n"
                                    for block in type_blocks:
                                        preview_text += f"{block['text']}\n"
                                    preview_text += "\n"
                            preview_text += "\n"
                        
                        # 使用可滾動的文本區域
                        st.text_area("classified_preview", preview_text, height=500, key="classified_preview", label_visibility="collapsed")
                    
                    else:  # 純文本
                        preview_text = ""
                        for page in result["pages"]:
                            preview_text += f"=== 第 {page['page_number']} 頁 ===\n"
                            preview_text += page['full_text']
                            preview_text += "\n\n"
                        
                        # 使用可滾動的文本區域
                        st.text_area("text_preview", preview_text, height=500, key="text_preview", label_visibility="collapsed")
                
                # 下載按鈕
                create_download_links(result)
        else:
            st.info("尚未處理任何文件，請先上傳並處理PDF文件。")
    
    with tab3:
        st.markdown("### 📚 使用說明")
        
        st.markdown("""
        #### 🎯 功能特點
        
        - **雙引擎OCR**: 結合PaddleOCR和Tesseract，提高識別準確度
        - **方向檢測**: 自動檢測直式/橫式文本
        - **中文優化**: 針對中文字體優化
        - **結構化輸出**: 生成詳細的JSON數據，包含位置信息
        
        #### 📋 使用步驟
        
        1. **上傳文件**: 在"上傳處理"標籤頁選擇PDF文件
        2. **開始處理**: 點擊"開始OCR處理"按鈕
        3. **查看結果**: 在"結果查看"標籤頁查看處理結果
        4. **下載數據**: 下載JSON、文本或統計報告
        
        #### 📊 輸出格式
        
        每個PDF會生成包含以下信息的JSON文件：
        
        - **文件基本信息**: 文件名、總頁數
        - **頁面詳細信息**: 每頁的文本方向、文本塊數量
        - **文本內容**: 識別的文字內容和位置信息
        - **置信度**: OCR識別的置信度分數
        
        #### ⚙️ 技術參數
        
        - **圖像DPI**: 可調整，建議300-600
        - **文本方向**: 自動檢測直式/橫式
        - **語言支持**: 簡體中文、繁體中文、英文
        - **處理時間**: 根據文件大小和頁數而定
        
        #### 🔧 系統要求
        
        - 支持PDF格式文件
        - 建議文件大小不超過100MB
        - 處理時間與文件大小成正比
        """)

if __name__ == "__main__":
    main()