# Safety Detector - Women Harassment Detection System

A simple Python project that uses AI to detect harassment in live camera feeds and video files.

## What it does

- **Web Interface**: Modern web UI with dashboard, live camera, and analysis
- **AI Detection**: Real-time harassment analysis with Gemini/OpenAI
- **Person Detection**: YOLOv8 for accurate person tracking
- **Parallel Processing**: Ultra-fast analysis with video chunking
- **Cost-Saving**: Auto-stop on alert, configurable performance

## Setup Instructions

### 1. Install Python packages
```bash
pip install -r requirements.txt
```

### 2. Get Google Gemini API Key
1. Go to [Google AI Studio](https://aistudio.google.com/)
2. Create a free account
3. Get your API key
4. Copy `config_example.env` to `.env`
5. Put your API key in the `.env` file

### 3. Test your camera
Make sure your computer has a working camera (webcam).

## How to Use

### 🚀 **Web Interface (Recommended)**
```bash
python run_web_server.py
```
- **Modern Web UI**: Access at `http://localhost:5000`
- **Real-time Camera**: Live monitoring with AI analysis
- **Video Analysis**: Drag-and-drop upload with parallel processing
- **History & Logs**: Comprehensive analysis history with evidence
- **Configuration**: All settings managed through web UI

### **Quick Start Menu**
```bash
python start_here.py
```
- Interactive menu with all options including web interface.

### Live Camera Monitoring (Command Line)
```bash
cd live_camera
python camera_monitor.py
```
- Press 'q' to quit
- Press 's' to save current frame
- Press 'c' to continue after alert (cost-saving)
- Press 'l' to save session log

### Video File Analysis (Command Line)
```bash
cd video_files
python video_analyzer.py
```
- Put video files in the `test_videos` folder
- Choose which video to analyze
- Check results and saved alert images
- **Note**: Web interface is recommended for better experience

## What the AI looks for

- Someone following another person
- Someone blocking another person's path
- Aggressive body language
- Someone cornering another person
- Any threatening behavior

## Files Explained

- `helpers/person_detector.py` - Finds people in images
- `helpers/simple_ai.py` - Uses AI to detect harassment
- `live_camera/camera_monitor.py` - Live camera monitoring
- `video_files/video_analyzer.py` - Analyze video files
- `test_videos/` - Put your test videos here

## Important Notes

⚠️ **This is a prototype for educational purposes**
- May have false positives (wrong alerts)
- Needs good lighting to work well
- Requires internet for AI analysis
- Always verify alerts manually

## Troubleshooting

**Camera not working**: 
- Check if another app is using your camera
- Run the camera diagnostics tool: `python camera_diagnostics.py`
- On the web interface, use the "Test Camera" button to verify
- Make sure you have granted camera permissions to your browser

**No AI responses**: Check your Gemini API key in `.env` file
**Slow performance**: Try smaller video files first
**No people detected**: Make sure people are clearly visible in the frame
