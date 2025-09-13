# Simple and reliable web server launcher
# This script starts the Safety Detector web interface

# CRITICAL: Suppress OpenCV warnings BEFORE any other imports
import os
import sys
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

from pathlib import Path

# Suppress OpenCV warnings as early as possible
try:
    import cv2
    cv2.setLogLevel(0)
except ImportError:
    pass  # OpenCV will be imported later

def main():
    """Start the web interface"""
    print("🛡️ Safety Detector Web Interface")
    print("=" * 40)
    print("✅ OpenCV warnings suppressed")
    print("✅ Camera error fixes applied")
    
    # Get the script directory
    script_dir = Path(__file__).parent.resolve()
    web_dir = script_dir / 'web_interface'
    
    # Check if web interface exists
    if not web_dir.exists():
        print(f"❌ Error: web_interface directory not found")
        print(f"Expected location: {web_dir}")
        return 1
    
    # Add paths to Python path
    sys.path.insert(0, str(script_dir))
    sys.path.insert(0, str(script_dir / 'helpers'))
    sys.path.insert(0, str(web_dir))
    
    # Change to web interface directory
    original_dir = os.getcwd()
    
    try:
        os.chdir(web_dir)
        print(f"📁 Starting from: {os.getcwd()}")
        
        # Import Flask app
        from app import app
        
        print("🌐 Web interface starting...")
        print("📱 Open your browser to: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop")
        print("=" * 40)
        
        # Start the Flask app with threading enabled
        app.run(host='0.0.0.0', port=5000, debug=False, use_reloader=False, threaded=True)
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("💡 Make sure Flask is installed: pip install flask")
        return 1
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped")
        return 0
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return 1
        
    finally:
        # Restore original directory
        os.chdir(original_dir)

if __name__ == "__main__":
    sys.exit(main())
