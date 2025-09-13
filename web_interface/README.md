# Safety Detector Web Interface

A modern, responsive web interface for the Safety Detector harassment detection system.

## Features

### 🎯 **Core Functionality**
- **Video Upload & Analysis**: Drag-and-drop video upload with real-time analysis
- **Live Camera Monitoring**: Real-time camera feed monitoring with instant alerts
- **ESP32 Camera Support**: Connect ESP32-CAM modules via WiFi IP address
- **Configuration Management**: Web-based AI model and system configuration
- **Multi-AI Support**: Switch between Gemini and OpenAI models seamlessly

### 🛡️ **Smart Features**
- **Cost-Saving Auto-Stop**: Automatically pauses analysis when harassment detected
- **Real-time Dashboard**: Live monitoring with FPS, people count, and alert tracking
- **Evidence Capture**: Automatic screenshot saving with timestamps
- **Mobile Responsive**: Works perfectly on phones, tablets, and desktops
- **Progress Tracking**: Real-time analysis progress with estimated completion time

### 🎨 **Modern UI/UX**
- **Bootstrap 5**: Modern, professional design
- **Font Awesome Icons**: Beautiful iconography throughout
- **Dark Mode Ready**: Automatic dark mode detection
- **Smooth Animations**: Fade-in effects and smooth transitions
- **Notifications**: Toast notifications for system feedback

## Quick Start

### 1. **Launch Web Interface**
```bash
python start_web_interface.py
```

### 2. **Access Interface**
Open your browser to: `http://localhost:5000`

### 3. **Upload Video or Start Camera**
- **Video Analysis**: Drag video files to upload area
- **Live Monitoring**: Select camera and click "Start Monitoring"
- **ESP32 Camera**: Enter IP address and test connection

## File Structure

```
web_interface/
├── app.py                 # Flask application
├── static/
│   ├── css/
│   │   └── style.css     # Custom styles
│   └── js/
│       └── main.js       # JavaScript functionality
├── templates/
│   ├── base.html         # Base template
│   ├── index.html        # Dashboard
│   ├── upload.html       # Video upload
│   ├── camera.html       # Live camera
│   ├── config.html       # Configuration
│   └── analyze.html      # Analysis results
└── uploads/              # Uploaded video files
```

## Key Pages

### 🏠 **Dashboard (`/`)**
- System status overview
- Quick action buttons
- Available cameras list
- Current AI model status
- Feature highlights

### 📤 **Video Upload (`/upload`)**
- Drag-and-drop file upload
- File validation and progress
- Supported formats display
- Recent uploads history

### 📹 **Live Camera (`/camera`)**
- Available cameras list
- ESP32 camera connection
- Real-time monitoring controls
- Session statistics tracking
- Alert management with auto-pause

### ⚙️ **Configuration (`/config`)**
- AI model selection
- Confidence threshold adjustment
- Cost-saving feature toggles
- Behavior monitoring settings
- Configuration export/import

### 📊 **Analysis (`/analyze/<filename>`)**
- Real-time analysis progress
- Cost estimation and tracking
- Alert detection and display
- Evidence screenshot gallery
- Auto-stop management

## ESP32 Camera Integration

### **Setup Steps**
1. Flash ESP32 with camera firmware
2. Connect to WiFi network
3. Find the ESP32 IP address
4. Enter IP in web interface
5. Test connection and start monitoring

### **Supported Endpoints**
- `/cam-hi.jpg` - High quality image capture
- `/capture` - Standard capture endpoint
- `:81/stream` - Video stream port
- `/stream` - Alternative stream endpoint

### **Connection Testing**
The interface automatically tests multiple endpoints to find the best connection method for your ESP32 camera.

## API Endpoints

### **Video Analysis**
- `POST /upload_video` - Upload video file
- `POST /start_analysis` - Start video analysis
- `GET /analysis_status` - Get analysis progress

### **Camera Management**
- `GET /camera` - Camera selection page
- `POST /test_esp32` - Test ESP32 connection

### **Configuration**
- `GET /config` - Configuration page
- `POST /update_config` - Update system settings

## Advanced Features

### **Cost Management**
- **Auto-Stop on Alert**: Stops analysis when harassment detected
- **User Prompts**: Choice to continue or stop after alerts
- **Cost Estimation**: Real-time cost tracking and savings display
- **Progress Monitoring**: Frame-by-frame analysis tracking

### **Evidence Collection**
- **Automatic Screenshots**: Saves evidence on alert detection
- **Timestamp Logging**: Precise time tracking for incidents
- **Session Reports**: Comprehensive analysis summaries
- **Download Options**: Export evidence and reports

### **Real-time Monitoring**
- **Live FPS Display**: Real-time frame rate monitoring
- **People Counting**: Active person detection display
- **Alert Tracking**: Running alert counter
- **Session Duration**: Elapsed time tracking

## Technical Architecture

### **Backend (Flask)**
- **Model Integration**: Seamless AI model switching
- **Camera Abstraction**: Unified camera handling
- **Progress Tracking**: Real-time analysis status
- **Error Handling**: Graceful error management

### **Frontend (JavaScript)**
- **Real-time Updates**: WebSocket-like status polling
- **File Management**: Drag-drop upload handling
- **Form Validation**: Client-side input validation
- **Responsive Design**: Mobile-first approach

### **Security Features**
- **File Validation**: Secure file type checking
- **Input Sanitization**: XSS protection
- **Error Boundaries**: Safe error handling
- **Local Processing**: Privacy-conscious design

## Customization

### **Styling**
Modify `static/css/style.css` to customize:
- Color schemes and themes
- Layout and spacing
- Animation effects
- Mobile responsiveness

### **Functionality**
Extend `static/js/main.js` to add:
- Custom notifications
- Additional form validation
- Enhanced animations
- New interactive features

### **Templates**
Customize HTML templates in `templates/` for:
- Layout modifications
- Content additions
- UI component changes
- Branding updates

## Performance Optimization

### **Client-side**
- **Lazy Loading**: Images and components load on demand
- **Caching**: Static assets cached for performance
- **Compression**: Minified CSS and JavaScript
- **Responsive Images**: Optimized for different screen sizes

### **Server-side**
- **Background Processing**: Analysis runs in separate threads
- **Status Polling**: Efficient progress updates
- **Memory Management**: Optimized for large video files
- **Connection Pooling**: Efficient database connections

## Troubleshooting

### **Common Issues**

**Port Already in Use**
```bash
# Kill process using port 5000
netstat -ano | findstr :5000
taskkill /PID <process_id> /F
```

**Dependencies Missing**
```bash
pip install -r requirements.txt
```

**Camera Not Detected**
- Check camera permissions
- Try different camera indices
- Restart browser and clear cache
- Run `python camera_diagnostics.py` for advanced troubleshooting

**ESP32 Connection Failed**
- Verify IP address is correct
- Check ESP32 WiFi connection
- Try different endpoints manually

### **Debug Mode**
Run with debug enabled:
```python
app.run(host='0.0.0.0', port=5000, debug=True)
```

## Browser Compatibility

### **Supported Browsers**
- ✅ Chrome 90+
- ✅ Firefox 88+
- ✅ Safari 14+
- ✅ Edge 90+

### **Required Features**
- WebRTC for camera access
- File API for drag-drop uploads
- Fetch API for AJAX requests
- CSS Grid and Flexbox

## Contributing

### **Development Setup**
1. Clone repository
2. Install dependencies: `pip install -r requirements.txt`
3. Set environment variables in `.env`
4. Run: `python start_web_interface.py`

### **Code Style**
- Python: Follow PEP 8
- JavaScript: ES6+ features
- HTML: Semantic markup
- CSS: BEM methodology

## License

This web interface is part of the Safety Detector project and follows the same licensing terms.

---

**🛡️ Safety Detector Web Interface - Making AI-powered safety accessible through modern web technology**
