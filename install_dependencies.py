# Install Dependencies for Safety Detector with Multi-AI Support
# This script installs all required packages for both Gemini and OpenAI models

import subprocess
import sys
import os

def install_package(package):
    """Install a single package using pip"""
    try:
        print(f"Installing {package}...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        print(f"✓ {package} installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Failed to install {package}: {e}")
        return False

def main():
    """Install all required dependencies"""
    print("=== Safety Detector - Dependency Installation ===")
    print("Installing packages for multi-AI harassment detection system")
    print()
    
    # List of required packages
    packages = [
        "opencv-python==4.8.1.78",
        "ultralytics==8.0.196", 
        "numpy==1.24.3",
        "pillow==10.0.1",
        "google-generativeai==0.3.2",
        "openai==1.3.0",
        "python-dotenv==1.0.0",
        "python-docx==0.8.11"
    ]
    
    print(f"Installing {len(packages)} packages...")
    print()
    
    failed_packages = []
    
    for package in packages:
        if not install_package(package):
            failed_packages.append(package)
    
    print()
    print("=== Installation Complete ===")
    
    if not failed_packages:
        print("🎉 All packages installed successfully!")
        print()
        print("Next steps:")
        print("1. Set up your API keys in .env file:")
        print("   - Copy config_example.env to .env")
        print("   - Add your Gemini API key")
        print("   - Add your OpenAI API key (optional)")
        print()
        print("2. Test your setup:")
        print("   python test_setup.py")
        print()
        print("3. Choose your AI model:")
        print("   python select_ai_model.py")
        print()
        print("4. Start using the system:")
        print("   python start_here.py")
        
    else:
        print("❌ Some packages failed to install:")
        for package in failed_packages:
            print(f"  - {package}")
        print()
        print("Please check your internet connection and try again.")
        print("You may need to run: pip install --upgrade pip")

if __name__ == "__main__":
    main()
