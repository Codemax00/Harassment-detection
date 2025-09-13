# Safety Detector Web Interface Launcher
# Easy launcher for the modern web interface

import os
import sys
import subprocess
import time
import webbrowser
from pathlib import Path

def check_dependencies():
    """Check if required packages are installed"""
    print("🔍 Checking dependencies...")
    
    required_packages = [
        'flask',
        'opencv-python',
        'ultralytics',
        'google-generativeai',
        'openai',
        'requests'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print("\n⚠️  Missing packages found!")
        print("Run this command to install missing packages:")
        print(f"pip install {' '.join(missing_packages)}")
        
        choice = input("\nInstall missing packages now? (y/n): ").lower()
        if choice in ['y', 'yes']:
            install_dependencies(missing_packages)
        else:
            print("❌ Cannot start without dependencies")
            return False
    
    return True

def install_dependencies(packages):
    """Install missing packages"""
    print("\n📦 Installing packages...")
    
    try:
        cmd = [sys.executable, "-m", "pip", "install"] + packages
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print("✅ All packages installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Installation failed: {e}")
        print("Please install manually using:")
        print(f"pip install {' '.join(packages)}")
        return False

def check_config():
    """Check if configuration files exist"""
    print("\n🔧 Checking configuration...")
    
    config_files = [
        'ai_config.json',
        '.env'
    ]
    
    missing_configs = []
    
    for config_file in config_files:
        if os.path.exists(config_file):
            print(f"✓ {config_file}")
        else:
            print(f"⚠️  {config_file} not found")
            missing_configs.append(config_file)
    
    if missing_configs:
        print("\n📝 Configuration setup needed:")
        for config in missing_configs:
            if config == '.env':
                print("  - Create .env file with your API keys")
                print("    GEMINI_API_KEY=your_gemini_key_here")
                print("    OPENAI_API_KEY=your_openai_key_here")
            elif config == 'ai_config.json':
                print("  - Run select_ai_model.py to create configuration")
        
        print("\n💡 The web interface will work with basic functionality,")
        print("   but you'll need API keys for AI analysis.")
    
    return True

def start_web_server():
    """Start the Flask web server"""
    print("\n🚀 Starting Safety Detector Web Interface...")
    print("=" * 50)
    
    # Get the current directory
    current_dir = Path(__file__).parent.resolve()
    web_dir = current_dir / 'web_interface'
    
    # Verify web interface directory exists
    if not web_dir.exists():
        print(f"❌ Error: Web interface directory not found at {web_dir}")
        return False
    
    # Add directories to Python path
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir / 'helpers'))
    sys.path.insert(0, str(web_dir))
    
    # Change to web interface directory
    original_cwd = os.getcwd()
    os.chdir(web_dir)
    
    try:
        # Import and run the Flask app
        print(f"📁 Working directory: {os.getcwd()}")
        from app import app
        
        print("🌐 Web interface starting...")
        print("📱 Access the interface at: http://localhost:5000")
        print("🛑 Press Ctrl+C to stop the server")
        print("=" * 50)
        
        # Open browser after a short delay
        def open_browser():
            time.sleep(2)
            try:
                webbrowser.open('http://localhost:5000')
                print("🌐 Opening browser...")
            except Exception as e:
                print(f"⚠️  Could not open browser automatically: {e}")
        
        import threading
        browser_thread = threading.Thread(target=open_browser)
        browser_thread.daemon = True
        browser_thread.start()
        
        # Start Flask app
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except ImportError as e:
        print(f"❌ Error importing Flask app: {e}")
        print("Make sure all dependencies are installed")
        return False
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        return True
    except Exception as e:
        print(f"❌ Error starting server: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Restore original working directory
        try:
            os.chdir(original_cwd)
        except:
            pass

def show_banner():
    """Show welcome banner"""
    banner = """
╔══════════════════════════════════════════════════════════════╗
║                                                              ║
║  🛡️  SAFETY DETECTOR - WEB INTERFACE LAUNCHER  🛡️            ║
║                                                              ║
║     Modern Web Interface for Harassment Detection            ║
║                                                              ║
╚══════════════════════════════════════════════════════════════╝
"""
    print(banner)

def show_features():
    """Show interface features"""
    print("🎯 Features:")
    print("   📹 Video Upload & Analysis")
    print("   📷 Live Camera Monitoring")
    print("   📡 ESP32 Camera Support")
    print("   ⚙️  Configuration Management")
    print("   🤖 Multiple AI Models (Gemini/OpenAI)")
    print("   💰 Cost-Saving Auto-Stop")
    print("   📊 Real-time Analysis Dashboard")
    print("   📱 Mobile-Responsive Design")

def main():
    """Main launcher function"""
    show_banner()
    show_features()
    
    print("\n🔄 Initializing...")
    
    # Check dependencies
    if not check_dependencies():
        return
    
    # Check configuration
    check_config()
    
    # Start web server
    if not start_web_server():
        print("\n❌ Failed to start web interface")
        print("Check the error messages above and try again")
    
    print("\n👋 Thank you for using Safety Detector!")

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        print("Please check your setup and try again")
