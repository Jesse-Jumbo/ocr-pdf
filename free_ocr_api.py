#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
免費OCR API服務集成
使用免費的在線OCR服務進行高精度文本識別
"""

import requests
import base64
import json
import time
import os
import io
from typing import Dict, Any, List
import streamlit as st

class FreeOCRAPI:
    """免費OCR API服務"""
    
    def __init__(self):
        self.services = {
            "ocr_space": {
                "url": "https://api.ocr.space/parse/image",
                "free_limit": 500,  # 每月免費次數
                "supported_languages": ["chi_sim", "chi_tra", "eng"]
            },
            "google_vision": {
                "url": "https://vision.googleapis.com/v1/images:annotate",
                "free_limit": 1000,  # 每月免費次數
                "supported_languages": ["zh", "en"]
            }
        }
    
    def ocr_with_ocr_space(self, image_data: bytes, language: str = "chi_tra") -> Dict[str, Any]:
        """使用OCR.space API進行OCR"""
        try:
            # 將圖像編碼為base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # API請求參數
            payload = {
                'apikey': 'helloworld',  # 免費API密鑰
                'language': language,
                'isOverlayRequired': False,
                'filetype': 'PNG',
                'base64Image': f'data:image/png;base64,{image_base64}'
            }
            
            # 發送請求
            response = requests.post(
                self.services["ocr_space"]["url"],
                data=payload,
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                if result.get('IsErroredOnProcessing', False):
                    return {"error": f"OCR處理錯誤: {result.get('ErrorMessage', '未知錯誤')}"}
                
                # 提取文本
                text_blocks = []
                if 'ParsedResults' in result:
                    for parsed_result in result['ParsedResults']:
                        if 'TextOverlay' in parsed_result:
                            for line in parsed_result['TextOverlay']['Lines']:
                                for word in line['Words']:
                                    text_blocks.append({
                                        "text": word['WordText'],
                                        "confidence": word['Confidence'] / 100.0,
                                        "position": {
                                            "x": word['Left'],
                                            "y": word['Top'],
                                            "width": word['Width'],
                                            "height": word['Height']
                                        }
                                    })
                
                return {
                    "success": True,
                    "text_blocks": text_blocks,
                    "full_text": result.get('ParsedResults', [{}])[0].get('ParsedText', '')
                }
            else:
                return {"error": f"API請求失敗: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"OCR.space API錯誤: {e}"}
    
    def ocr_with_google_vision(self, image_data: bytes, language: str = "zh") -> Dict[str, Any]:
        """使用Google Vision API進行OCR（需要API密鑰）"""
        try:
            # 這裡需要Google Vision API密鑰
            api_key = st.secrets.get("GOOGLE_VISION_API_KEY", "")
            if not api_key:
                return {"error": "Google Vision API密鑰未配置"}
            
            # 將圖像編碼為base64
            image_base64 = base64.b64encode(image_data).decode('utf-8')
            
            # API請求
            url = f"{self.services['google_vision']['url']}?key={api_key}"
            payload = {
                "requests": [{
                    "image": {
                        "content": image_base64
                    },
                    "features": [{
                        "type": "TEXT_DETECTION",
                        "maxResults": 1
                    }],
                    "imageContext": {
                        "languageHints": [language]
                    }
                }]
            }
            
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                text_blocks = []
                
                if 'responses' in result and result['responses']:
                    text_annotations = result['responses'][0].get('textAnnotations', [])
                    if text_annotations:
                        # 第一個是完整文本
                        full_text = text_annotations[0].get('description', '')
                        
                        # 其餘是單詞
                        for annotation in text_annotations[1:]:
                            vertices = annotation.get('boundingPoly', {}).get('vertices', [])
                            if len(vertices) >= 2:
                                x = vertices[0].get('x', 0)
                                y = vertices[0].get('y', 0)
                                width = vertices[2].get('x', 0) - x if len(vertices) > 2 else 0
                                height = vertices[2].get('y', 0) - y if len(vertices) > 2 else 0
                                
                                text_blocks.append({
                                    "text": annotation.get('description', ''),
                                    "confidence": 0.9,  # Google Vision不提供置信度
                                    "position": {
                                        "x": x,
                                        "y": y,
                                        "width": width,
                                        "height": height
                                    }
                                })
                
                return {
                    "success": True,
                    "text_blocks": text_blocks,
                    "full_text": full_text
                }
            else:
                return {"error": f"Google Vision API請求失敗: {response.status_code}"}
                
        except Exception as e:
            return {"error": f"Google Vision API錯誤: {e}"}
    
    def process_pdf_with_free_ocr(self, pdf_path: str) -> Dict[str, Any]:
        """使用免費OCR處理PDF"""
        try:
            # 這裡需要先將PDF轉換為圖像
            # 由於我們在Streamlit環境中，使用現有的PDF轉換功能
            from pdf2image import convert_from_path
            import cv2
            import numpy as np
            
            # 轉換PDF為圖像
            images = convert_from_path(pdf_path, dpi=300)
            
            result = {
                "file_name": os.path.basename(pdf_path),
                "total_pages": len(images),
                "pages": []
            }
            
            for page_num, image in enumerate(images, 1):
                # 轉換PIL圖像為bytes
                img_bytes = io.BytesIO()
                image.save(img_bytes, format='PNG')
                img_bytes = img_bytes.getvalue()
                
                # 使用OCR.space進行OCR
                ocr_result = self.ocr_with_ocr_space(img_bytes, "chi_tra")
                
                if "error" in ocr_result:
                    st.warning(f"第 {page_num} 頁OCR失敗: {ocr_result['error']}")
                    continue
                
                # 分類文本
                classified_blocks = []
                for text_data in ocr_result["text_blocks"]:
                    classified_blocks.append({
                        "text": text_data["text"],
                        "type": "content",  # 簡單分類
                        "direction": "horizontal",
                        "confidence": text_data["confidence"]
                    })
                
                page_result = {
                    "page_number": page_num,
                    "text_blocks": classified_blocks,
                    "full_text": ocr_result.get("full_text", "")
                }
                result["pages"].append(page_result)
            
            return result
            
        except Exception as e:
            return {"error": f"免費OCR處理失敗: {e}"}

def show_free_ocr_setup():
    """顯示免費OCR設置指南"""
    st.markdown("""
    ## 🆓 免費OCR服務設置指南
    
    ### 1. OCR.space (推薦)
    - **費用**: 免費（每月500次）
    - **準確度**: 中等
    - **設置**: 無需API密鑰
    - **支持語言**: 中文、英文
    
    ### 2. Google Vision API
    - **費用**: 免費（每月1000次）
    - **準確度**: 很高
    - **設置**: 需要API密鑰
    - **支持語言**: 多語言
    
    ### 3. 使用方式
    1. 選擇OCR服務
    2. 上傳PDF文件
    3. 系統自動處理並返回結果
    """)

if __name__ == "__main__":
    st.title("免費OCR服務")
    show_free_ocr_setup()
