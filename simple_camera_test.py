#!/usr/bin/env python3
"""
Simple Camera Test - Direct API Test
Tests camera functionality without complex validation
"""

import sys
import os

# Add paths
sys.path.insert(0, 'web_interface')
sys.path.insert(0, 'helpers')

# Suppress OpenCV warnings
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
import cv2
cv2.setLogLevel(0)

def test_camera_direct():
    """Test camera directly without web interface"""
    print("🧪 Direct Camera Test")
    print("=" * 30)
    
    try:
        print("Testing camera 0 with DirectShow...")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        
        if cap.isOpened():
            print("✅ Camera opened")
            
            ret, frame = cap.read()
            if ret:
                print("✅ Frame read successful")
                print(f"   Shape: {frame.shape}")
                
                # Test a few frames
                for i in range(5):
                    ret, frame = cap.read()
                    if ret:
                        print(f"   Frame {i+1}: OK")
                    else:
                        print(f"   Frame {i+1}: Failed")
                        break
                        
                result = True
            else:
                print("❌ Cannot read frames")
                result = False
        else:
            print("❌ Cannot open camera")
            result = False
            
        cap.release()
        return result
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def test_flask_app_direct():
    """Test Flask app import"""
    print("\n🌐 Flask App Test")
    print("=" * 20)
    
    try:
        os.chdir('web_interface')
        from app import app, get_available_cameras
        
        print("✅ Flask app imported")
        
        # Test camera detection
        cameras = get_available_cameras()
        print(f"✅ Found {len(cameras)} cameras")
        
        for cam in cameras:
            print(f"   Camera {cam['id']}: {cam['name']} ({cam['status']})")
        
        return True
        
    except Exception as e:
        print(f"❌ Flask app error: {e}")
        return False

if __name__ == "__main__":
    print("Simple Camera Test - Bypassing Web Interface")
    print("=" * 50)
    
    # Test camera directly
    camera_ok = test_camera_direct()
    
    # Test Flask app
    flask_ok = test_flask_app_direct()
    
    print("\n" + "=" * 50)
    print("📊 Test Results:")
    print(f"   Camera: {'✅ Working' if camera_ok else '❌ Failed'}")
    print(f"   Flask:  {'✅ Working' if flask_ok else '❌ Failed'}")
    
    if camera_ok and flask_ok:
        print("\n🎉 Both camera and Flask are working!")
        print("The issue is likely in the API request validation.")
        print("Try accessing the web interface and selecting the camera.")
    else:
        print("\n⚠️ Basic functionality issues detected.")
        print("Check the error messages above.")
