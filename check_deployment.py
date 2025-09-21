#!/usr/bin/env python3
"""
部署前檢查腳本 - 確保所有依賴都正確安裝
"""

def check_imports():
    """檢查所有必要的導入"""
    print("🔍 檢查依賴庫導入...")
    
    required_modules = [
        'streamlit',
        'cv2',
        'numpy', 
        'PIL',
        'pdf2image',
        'paddleocr',
        'pytesseract',
        'pandas',
        'tqdm'
    ]
    
    failed_imports = []
    
    for module in required_modules:
        try:
            if module == 'cv2':
                import cv2
            elif module == 'PIL':
                from PIL import Image
            elif module == 'pdf2image':
                import pdf2image
            elif module == 'paddleocr':
                from paddleocr import PaddleOCR
            elif module == 'pytesseract':
                import pytesseract
            else:
                __import__(module)
            print(f"✅ {module}")
        except ImportError as e:
            print(f"❌ {module}: {e}")
            failed_imports.append(module)
    
    if failed_imports:
        print(f"\n❌ 失敗的導入: {', '.join(failed_imports)}")
        return False
    else:
        print("\n✅ 所有依賴庫導入成功！")
        return True

def check_tesseract():
    """檢查Tesseract是否可用"""
    print("\n🔍 檢查Tesseract...")
    try:
        import pytesseract
        import shutil
        
        tesseract_path = shutil.which('tesseract')
        if tesseract_path:
            print(f"✅ Tesseract 找到: {tesseract_path}")
            return True
        else:
            print("❌ Tesseract 未找到")
            return False
    except Exception as e:
        print(f"❌ Tesseract 檢查失敗: {e}")
        return False

def main():
    """主檢查函數"""
    print("🚀 OCR應用部署檢查\n")
    
    imports_ok = check_imports()
    tesseract_ok = check_tesseract()
    
    if imports_ok and tesseract_ok:
        print("\n🎉 所有檢查通過！可以部署到Streamlit Cloud")
        return True
    else:
        print("\n⚠️ 檢查失敗，請修復問題後再部署")
        return False

if __name__ == "__main__":
    main()
