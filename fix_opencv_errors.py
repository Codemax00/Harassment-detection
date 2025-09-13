#!/usr/bin/env python3
"""
Fix OpenCV Camera Errors - Safety Detector
This script applies environment variables and settings to eliminate OpenCV camera errors
"""

import os
import sys

def apply_opencv_fixes():
    """Apply OpenCV environment fixes"""
    print("🔧 Applying OpenCV Camera Error Fixes")
    print("=" * 40)
    
    # Set environment variables to suppress OpenCV warnings
    fixes = {
        'OPENCV_LOG_LEVEL': 'ERROR',
        'OPENCV_VIDEOIO_DEBUG': '0',
        'OPENCV_VIDEOIO_PRIORITY_DSHOW': '1',
        'OPENCV_VIDEOIO_PRIORITY_MSMF': '0'
    }
    
    for key, value in fixes.items():
        os.environ[key] = value
        print(f"✅ Set {key} = {value}")
    
    print("\n🎥 Testing Camera with Fixes Applied")
    print("-" * 30)
    
    try:
        import cv2
        
        # Suppress OpenCV logging
        cv2.setLogLevel(0)
        
        print("Testing camera 0 with DirectShow...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret:
                print("✅ Camera 0 working perfectly!")
                print(f"   Frame shape: {frame.shape}")
            else:
                print("❌ Cannot read frames")
        else:
            print("❌ Cannot open camera")
        
        cap.release()
        
    except Exception as e:
        print(f"❌ Error: {e}")
    
    print("\n💡 Environment fixes applied!")
    print("Now start the web interface:")
    print("python start_web_clean.py")

if __name__ == "__main__":
    apply_opencv_fixes()
