# Simple starter script for Safety Detector
# This helps you choose what to do

import os
import sys

def show_menu():
    """Show the main menu"""
    print("=== Safety Detector - Women Harassment Detection ===")
    print()
    print("What would you like to do?")
    print()
    print("1. 🌐 Web Interface (Modern UI - Recommended)")
    print("2. Test setup (check if everything works)")
    print("3. Start live camera monitoring")
    print("4. Analyze a video file")
    print("5. Select AI model (Gemini/OpenAI)")
    print("6. Show help")
    print("7. Exit")
    print()

def run_test():
    """Run the setup test"""
    print("Running setup test...")
    os.system("python test_setup.py")

def run_live_camera():
    """Start live camera monitoring"""
    print("Starting live camera monitor...")
    print("(Press 'q' in the camera window to stop)")
    print()
    os.chdir("live_camera")
    os.system("python camera_monitor.py")
    os.chdir("..")

def run_video_analyzer():
    """Start video analyzer"""
    print("Starting video analyzer...")
    print()
    os.chdir("video_files")
    os.system("python video_analyzer.py")
    os.chdir("..")

def run_web_interface():
    """Start the web interface"""
    print("Starting Safety Detector Web Interface...")
    print("This will open a modern web interface in your browser.")
    print()
    os.system("python start_web_interface.py")

def run_ai_selector():
    """Run the AI model selector"""
    print("Starting AI model selector...")
    os.system("python select_ai_model.py")

def show_help():
    """Show help information"""
    print()
    print("=== Help ===")
    print()
    print("🌐 Web Interface (RECOMMENDED):")
    print("- Modern web-based interface with beautiful UI")
    print("- Upload videos through drag & drop")
    print("- Real-time camera monitoring dashboard")
    print("- ESP32 camera support with IP connection")
    print("- Configuration management through web UI")
    print("- Mobile-responsive design")
    print("- Access at: http://localhost:5000")
    print()
    print("Live Camera Monitoring:")
    print("- Uses your webcam to watch for harassment in real-time")
    print("- Shows people detection with green boxes")
    print("- Uses AI to analyze behavior when 2+ people are present")
    print("- Press 'q' to quit, 's' to save frame, 'c' to continue after alert")
    print()
    print("Video File Analysis:")
    print("- Analyzes existing video files for harassment")
    print("- Put video files in the 'test_videos' folder")
    print("- Saves images of any alerts found")
    print("- Supports .mp4, .avi, .mov, .mkv, .wmv files")
    print("- Auto-stops on alert to save costs")
    print()
    print("AI Model Selection:")
    print("- Choose between Google Gemini and OpenAI GPT-4 Vision")
    print("- Gemini: Faster response, good for real-time")
    print("- OpenAI: More detailed analysis, higher accuracy")
    print("- Configure API keys in .env file")
    print()
    print("ESP32 Camera Support:")
    print("- Connect ESP32-CAM modules via WiFi")
    print("- Enter IP address to connect")
    print("- Supports multiple camera endpoints")
    print("- Works with both web interface and command line")
    print()
    print("Setup Requirements:")
    print("- Python packages: pip install -r requirements.txt")
    print("- AI API keys in .env file (Gemini and/or OpenAI)")
    print("- Working camera for live monitoring")
    print()
    print("For detailed instructions, see README.md")
    print()

def main():
    """Main menu loop"""
    while True:
        try:
            show_menu()
            choice = input("Enter your choice (1-7): ").strip()
            print()
            
            if choice == "1":
                run_web_interface()
            elif choice == "2":
                run_test()
            elif choice == "3":
                run_live_camera()
            elif choice == "4":
                run_video_analyzer()
            elif choice == "5":
                run_ai_selector()
            elif choice == "6":
                show_help()
            elif choice == "7":
                print("Thank you for using Safety Detector!")
                break
            else:
                print("Invalid choice. Please enter 1, 2, 3, 4, 5, 6, or 7.")
            
            print()
            input("Press Enter to continue...")
            print()
            
        except KeyboardInterrupt:
            print("\n\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")
            print("Please try again.")

if __name__ == "__main__":
    main()
