#!/usr/bin/env python3
"""
Camera Diagnostics Tool for Safety Detector
Helps troubleshoot camera detection and connection issues
"""

import cv2
import os
import time
import sys

def test_camera_backends():
    """Test different camera backends for better compatibility"""
    print("🔍 Testing Camera Backends")
    print("=" * 40)
    
    backends = []
    
    # Windows-specific backends
    if os.name == 'nt':
        backends.extend([
            ('DirectShow', cv2.CAP_DSHOW),
            ('Media Foundation', cv2.CAP_MSMF),
            ('Default', cv2.CAP_ANY)
        ])
    else:
        backends.extend([
            ('V4L2', cv2.CAP_V4L2),
            ('Default', cv2.CAP_ANY)
        ])
    
    working_backends = []
    
    for name, backend in backends:
        print(f"\n📹 Testing {name} backend...")
        try:
            cap = cv2.VideoCapture(0, backend)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    print(f"  ✅ {name}: Working")
                    working_backends.append((name, backend))
                else:
                    print(f"  ❌ {name}: Cannot read frames")
            else:
                print(f"  ❌ {name}: Cannot open camera")
            cap.release()
            time.sleep(0.2)  # Brief pause between tests
        except Exception as e:
            print(f"  ❌ {name}: Error - {e}")
    
    return working_backends

def scan_camera_indices():
    """Scan for available camera indices"""
    print("\n🔍 Scanning Camera Indices")
    print("=" * 40)
    
    available_cameras = []
    
    # Test camera indices 0-10
    for i in range(11):
        cap = None
        try:
            print(f"Testing camera index {i}...", end=" ")
            
            # Use the best backend available
            if os.name == 'nt':
                cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            else:
                cap = cv2.VideoCapture(i)
            
            if cap.isOpened():
                ret, frame = cap.read()
                if ret and frame is not None:
                    # Get camera properties
                    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    
                    available_cameras.append({
                        'index': i,
                        'width': width,
                        'height': height,
                        'fps': fps
                    })
                    
                    print(f"✅ Found! {width}x{height} @ {fps:.1f}fps")
                else:
                    print("❌ Cannot read")
            else:
                print("❌ Cannot open")
                
        except Exception as e:
            print(f"❌ Error: {e}")
        finally:
            if cap is not None:
                cap.release()
            time.sleep(0.1)  # Prevent resource conflicts
    
    return available_cameras

def test_camera_permissions():
    """Test camera permissions and access"""
    print("\n🔒 Testing Camera Permissions")
    print("=" * 40)
    
    try:
        # Try to access camera with different methods
        print("Testing default camera access...")
        cap = cv2.VideoCapture(0)
        
        if cap.isOpened():
            print("✅ Camera opened successfully")
            
            # Test frame reading
            ret, frame = cap.read()
            if ret:
                print("✅ Can read frames")
                print(f"   Frame shape: {frame.shape}")
                print(f"   Frame type: {frame.dtype}")
            else:
                print("❌ Cannot read frames - camera may be in use")
                
        else:
            print("❌ Cannot open camera")
            print("   Possible causes:")
            print("   - Camera is being used by another application")
            print("   - Camera drivers not installed")
            print("   - Insufficient permissions")
            
        cap.release()
        
    except Exception as e:
        print(f"❌ Permission test error: {e}")

def check_system_info():
    """Check system information relevant to camera operation"""
    print("\n💻 System Information")
    print("=" * 40)
    
    print(f"Operating System: {os.name}")
    print(f"Python Version: {sys.version}")
    print(f"OpenCV Version: {cv2.__version__}")
    
    # Check available camera backends
    print(f"\nAvailable OpenCV backends:")
    try:
        backends = cv2.videoio_registry.getCameraBackends()
        for backend in backends:
            backend_name = cv2.videoio_registry.getBackendName(backend)
            print(f"  - {backend_name} ({backend})")
    except:
        print("  Backend information not available")

def run_camera_diagnostics():
    """Run complete camera diagnostics"""
    print("🛡️ Safety Detector - Camera Diagnostics")
    print("=" * 50)
    
    # System info
    check_system_info()
    
    # Test backends
    working_backends = test_camera_backends()
    
    # Scan camera indices
    available_cameras = scan_camera_indices()
    
    # Test permissions
    test_camera_permissions()
    
    # Summary
    print("\n📊 DIAGNOSTIC SUMMARY")
    print("=" * 40)
    
    print(f"Working Backends: {len(working_backends)}")
    for name, backend in working_backends:
        print(f"  ✅ {name}")
    
    print(f"\nAvailable Cameras: {len(available_cameras)}")
    for cam in available_cameras:
        print(f"  📹 Camera {cam['index']}: {cam['width']}x{cam['height']} @ {cam['fps']:.1f}fps")
    
    # Recommendations
    print("\n💡 RECOMMENDATIONS")
    print("=" * 40)
    
    if not available_cameras:
        print("❌ No cameras detected. Try:")
        print("   1. Close all other camera applications (Zoom, Skype, etc.)")
        print("   2. Reconnect your camera")
        print("   3. Check Device Manager for camera drivers")
        print("   4. Run as administrator")
        print("   5. Try a different USB port")
    elif len(available_cameras) == 1:
        print("✅ One camera detected - should work fine")
        print(f"   Use camera index: {available_cameras[0]['index']}")
    else:
        print(f"✅ Multiple cameras detected ({len(available_cameras)})")
        print("   Choose the camera index that works best for you")
    
    if not working_backends:
        print("\n❌ No working backends found")
        print("   OpenCV installation may be corrupted")
        print("   Try: pip install --upgrade opencv-python")
    else:
        best_backend = working_backends[0]
        print(f"\n✅ Recommended backend: {best_backend[0]}")
    
    return len(available_cameras) > 0

if __name__ == "__main__":
    success = run_camera_diagnostics()
    
    if success:
        print("\n🎉 Camera diagnostics completed successfully!")
        print("Your cameras should work with the Safety Detector.")
    else:
        print("\n⚠️ Camera issues detected.")
        print("Please follow the recommendations above.")
    
    print("\nPress Enter to exit...")
    input()
