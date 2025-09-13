#!/usr/bin/env python3
"""
Clean startup script for Safety Detector Web Interface
Suppresses OpenCV warnings and provides better error handling
"""

import os
import sys
import cv2

# Suppress OpenCV warnings before any other imports
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
cv2.setLogLevel(0)

# Suppress other warnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

def main():
    print("🛡️ Safety Detector Web Interface (Clean Start)")
    print("=" * 50)
    
    try:
        # Change to web_interface directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        web_interface_dir = os.path.join(script_dir, 'web_interface')
        
        if os.path.exists(web_interface_dir):
            os.chdir(web_interface_dir)
            print(f"📁 Working directory: {web_interface_dir}")
        else:
            print(f"❌ Web interface directory not found: {web_interface_dir}")
            return False
        
        # Import and run the app
        print("🚀 Starting Flask application...")
        
        # Suppress OpenCV messages
        cv2.setLogLevel(0)
        
        from app import app, init_models
        
        print("🤖 Initializing AI models...")
        if init_models():
            print("✅ AI models initialized successfully")
        else:
            print("⚠️ Some AI models failed to initialize (will work with available models)")
        
        print("🌐 Starting web server...")
        print("📱 Access at: http://localhost:5000")
        print("🔗 Network access: http://0.0.0.0:5000")
        print()
        print("Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Start the Flask app
        app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
        
    except KeyboardInterrupt:
        print("\n👋 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Startup error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        print("\n⚠️ Server failed to start. Check the error messages above.")
        input("Press Enter to exit...")
