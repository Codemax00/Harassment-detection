#!/usr/bin/env python3
"""
Final Camera Test - Complete Verification
Tests all camera functionality to ensure everything works
"""

import requests
import time
import json

def test_web_server():
    """Test if web server is running"""
    try:
        response = requests.get('http://localhost:5000', timeout=5)
        return response.status_code == 200
    except:
        return False

def test_camera_page():
    """Test camera page loads"""
    try:
        response = requests.get('http://localhost:5000/camera', timeout=10)
        return response.status_code == 200 and 'Available Cameras' in response.text
    except:
        return False

def test_camera_monitoring_api():
    """Test camera monitoring API with multiple approaches"""
    print("🧪 Testing Camera Monitoring API")
    print("-" * 30)
    
    # Test 1: Original endpoint
    try:
        print("1. Testing /start_camera_monitoring...")
        response = requests.post(
            'http://localhost:5000/start_camera_monitoring',
            json={'camera_id': 0, 'camera_type': 'local'},
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ Original endpoint working!")
            return True
        else:
            print(f"   ❌ Response: {response.text[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 2: Simple endpoint
    try:
        print("2. Testing /camera/start...")
        response = requests.post(
            'http://localhost:5000/camera/start',
            json={'camera_id': 0},
            timeout=10
        )
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   ✅ Simple endpoint working! {data}")
            return True
        else:
            print(f"   ❌ Response: {response.text[:100]}...")
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    # Test 3: GET request
    try:
        print("3. Testing GET /camera/start...")
        response = requests.get('http://localhost:5000/camera/start?camera_id=0', timeout=10)
        
        print(f"   Status: {response.status_code}")
        if response.status_code == 200:
            print("   ✅ GET endpoint working!")
            return True
            
    except Exception as e:
        print(f"   ❌ Error: {e}")
    
    return False

def test_camera_status():
    """Test camera status endpoint"""
    try:
        response = requests.get('http://localhost:5000/camera_status', timeout=5)
        if response.status_code == 200:
            data = response.json()
            print(f"✅ Camera status: {data.get('status', 'unknown')}")
            return True
    except Exception as e:
        print(f"❌ Camera status error: {e}")
    return False

def run_complete_test():
    """Run complete camera test suite"""
    print("🛡️ Final Camera Test Suite")
    print("=" * 50)
    
    # Check web server
    print("1. Checking web server...")
    if not test_web_server():
        print("❌ Web server not running!")
        print("   Start with: python run_web_server.py")
        return False
    print("✅ Web server running")
    
    # Check camera page
    print("\n2. Testing camera page...")
    if not test_camera_page():
        print("❌ Camera page not loading!")
        return False
    print("✅ Camera page loads correctly")
    
    # Test camera monitoring
    print("\n3. Testing camera monitoring...")
    if not test_camera_monitoring_api():
        print("❌ Camera monitoring API failed!")
        return False
    print("✅ Camera monitoring API working")
    
    # Test camera status
    print("\n4. Testing camera status...")
    if not test_camera_status():
        print("❌ Camera status API failed!")
        return False
    print("✅ Camera status API working")
    
    print("\n" + "=" * 50)
    print("🎉 ALL TESTS PASSED!")
    print("Your camera functionality is working correctly.")
    print("\n📱 Next steps:")
    print("1. Go to: http://localhost:5000/camera")
    print("2. Camera 0 should be auto-selected")
    print("3. Click 'Start' to begin monitoring")
    print("4. Live video feed should appear")
    
    return True

if __name__ == "__main__":
    success = run_complete_test()
    
    if not success:
        print("\n⚠️ Some tests failed. Check the error messages above.")
        print("Try restarting the web server and running this test again.")
    
    print("\nPress Enter to exit...")
    input()
