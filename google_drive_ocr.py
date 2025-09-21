#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Drive OCR 集成模組
使用Google Drive API進行高精度OCR處理
"""

import os
import io
import json
import time
import tempfile
from typing import Dict, Any, List
import streamlit as st

try:
    from google.oauth2.credentials import Credentials
    from google_auth_oauthlib.flow import Flow
    from google.auth.transport.requests import Request
    from googleapiclient.discovery import build
    from googleapiclient.http import MediaIoBaseUpload
    import pickle
    GOOGLE_DRIVE_AVAILABLE = True
except ImportError:
    GOOGLE_DRIVE_AVAILABLE = False

class GoogleDriveOCR:
    """Google Drive OCR處理器"""
    
    def __init__(self):
        self.service = None
        self.credentials = None
        self.scopes = ['https://www.googleapis.com/auth/drive.file']
        self.client_secrets_file = 'client_secrets.json'
        self.token_file = 'token.pickle'
    
    def authenticate(self):
        """認證Google Drive API"""
        if not GOOGLE_DRIVE_AVAILABLE:
            st.error("❌ Google Drive API未安裝，請安裝：pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
            return False
        
        try:
            # 檢查是否有保存的憑證
            if os.path.exists(self.token_file):
                with open(self.token_file, 'rb') as token:
                    self.credentials = pickle.load(token)
            
            # 如果沒有有效憑證，進行OAuth流程
            if not self.credentials or not self.credentials.valid:
                if self.credentials and self.credentials.expired and self.credentials.refresh_token:
                    self.credentials.refresh(Request())
                else:
                    if not os.path.exists(self.client_secrets_file):
                        st.warning("⚠️ 請先配置Google Drive API密鑰")
                        return False
                    
                    flow = Flow.from_client_secrets_file(
                        self.client_secrets_file, 
                        scopes=self.scopes
                    )
                    flow.redirect_uri = 'urn:ietf:wg:oauth:2.0:oob'
                    
                    auth_url, _ = flow.authorization_url(prompt='consent')
                    st.markdown(f"請訪問以下網址進行認證：")
                    st.markdown(f"[{auth_url}]({auth_url})")
                    
                    auth_code = st.text_input("請輸入授權碼：")
                    if auth_code:
                        flow.fetch_token(code=auth_code)
                        self.credentials = flow.credentials
                        
                        # 保存憑證
                        with open(self.token_file, 'wb') as token:
                            pickle.dump(self.credentials, token)
                    else:
                        return False
            
            # 建立服務
            self.service = build('drive', 'v3', credentials=self.credentials)
            return True
            
        except Exception as e:
            st.error(f"Google Drive認證失敗: {e}")
            return False
    
    def upload_and_ocr_pdf(self, pdf_path: str) -> Dict[str, Any]:
        """上傳PDF並使用Google Drive OCR"""
        try:
            if not self.service:
                if not self.authenticate():
                    return {"error": "Google Drive認證失敗"}
            
            # 上傳PDF到Google Drive
            file_metadata = {
                'name': os.path.basename(pdf_path),
                'mimeType': 'application/pdf'
            }
            
            media = MediaIoBaseUpload(
                io.FileIO(pdf_path, 'rb'),
                mimetype='application/pdf',
                resumable=True
            )
            
            file = self.service.files().create(
                body=file_metadata,
                media_body=media,
                fields='id'
            ).execute()
            
            file_id = file.get('id')
            st.info(f"PDF已上傳到Google Drive，文件ID: {file_id}")
            
            # 使用Google Docs打開PDF（自動OCR）
            doc_metadata = {
                'name': f"{os.path.splitext(os.path.basename(pdf_path))[0]}_ocr",
                'parents': [file_id]
            }
            
            # 這裡需要實現Google Docs OCR
            # 由於API限制，我們提供一個替代方案
            return self._extract_text_from_drive_file(file_id)
            
        except Exception as e:
            return {"error": f"Google Drive OCR失敗: {e}"}
    
    def _extract_text_from_drive_file(self, file_id: str) -> Dict[str, Any]:
        """從Google Drive文件提取文本"""
        try:
            # 這裡需要實現文本提取邏輯
            # 由於Google Drive API的限制，我們提供一個簡化版本
            st.warning("⚠️ Google Drive OCR功能需要進一步配置")
            return {"error": "Google Drive OCR功能暫未完全實現"}
            
        except Exception as e:
            return {"error": f"文本提取失敗: {e}"}
    
    def cleanup_file(self, file_id: str):
        """清理Google Drive上的文件"""
        try:
            if self.service:
                self.service.files().delete(fileId=file_id).execute()
        except Exception as e:
            st.warning(f"清理文件失敗: {e}")

def create_google_drive_setup_guide():
    """創建Google Drive API設置指南"""
    st.markdown("""
    ## 🔧 Google Drive OCR 設置指南
    
    ### 1. 創建Google Cloud項目
    1. 前往 [Google Cloud Console](https://console.cloud.google.com/)
    2. 創建新項目或選擇現有項目
    3. 啟用 Google Drive API
    
    ### 2. 創建OAuth 2.0憑證
    1. 前往「憑證」頁面
    2. 點擊「建立憑證」→「OAuth 2.0 用戶端 ID」
    3. 選擇「桌面應用程式」
    4. 下載JSON文件並重命名為 `client_secrets.json`
    5. 將文件放在項目根目錄
    
    ### 3. 安裝依賴
    ```bash
    pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib
    ```
    
    ### 4. 使用方式
    1. 上傳 `client_secrets.json` 到項目根目錄
    2. 重新啟動應用程式
    3. 首次使用時會要求認證
    """)

# 替代方案：使用在線OCR服務
class OnlineOCRServices:
    """在線OCR服務集成"""
    
    @staticmethod
    def get_available_services():
        """獲取可用的在線OCR服務"""
        return {
            "Google Cloud Vision API": {
                "description": "Google的高精度OCR服務",
                "cost": "按使用量付費",
                "accuracy": "很高",
                "setup": "需要API密鑰"
            },
            "Azure Computer Vision": {
                "description": "Microsoft的OCR服務",
                "cost": "按使用量付費",
                "accuracy": "很高",
                "setup": "需要API密鑰"
            },
            "AWS Textract": {
                "description": "Amazon的OCR服務",
                "cost": "按使用量付費",
                "accuracy": "很高",
                "setup": "需要API密鑰"
            },
            "OCR.space": {
                "description": "免費的OCR API服務",
                "cost": "免費（有限制）",
                "accuracy": "中等",
                "setup": "需要API密鑰"
            }
        }
    
    @staticmethod
    def show_service_comparison():
        """顯示服務比較"""
        services = OnlineOCRServices.get_available_services()
        
        st.markdown("### 📊 在線OCR服務比較")
        
        for service_name, details in services.items():
            with st.expander(f"**{service_name}**"):
                st.write(f"**描述**: {details['description']}")
                st.write(f"**費用**: {details['cost']}")
                st.write(f"**準確度**: {details['accuracy']}")
                st.write(f"**設置**: {details['setup']}")

def main():
    """主函數 - 用於測試"""
    st.title("Google Drive OCR 設置")
    
    # 檢查Google Drive API是否可用
    if not GOOGLE_DRIVE_AVAILABLE:
        st.error("❌ Google Drive API未安裝")
        st.code("pip install google-api-python-client google-auth-httplib2 google-auth-oauthlib")
        return
    
    # 顯示設置指南
    create_google_drive_setup_guide()
    
    # 顯示替代方案
    OnlineOCRServices.show_service_comparison()

if __name__ == "__main__":
    main()
