# Safety Detector Web Interface
# Modern web frontend for harassment detection system

# CRITICAL: Suppress OpenCV warnings BEFORE any imports
import os
import sys
os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'
os.environ['OPENCV_VIDEOIO_DEBUG'] = '0'

import cv2
# Suppress OpenCV logging immediately
cv2.setLogLevel(0)

from flask import Flask, render_template, request, jsonify, redirect, url_for, flash, send_file

# CSRF protection intentionally disabled for API compatibility
CSRF_AVAILABLE = False
CSRFProtect = None
exempt = lambda f: f  # No-op decorator

import json
import threading
import time
from datetime import datetime
import logging

# Additional logging suppression
logging.getLogger('opencv').setLevel(logging.ERROR)
logging.getLogger('ultralytics').setLevel(logging.ERROR)
from werkzeug.utils import secure_filename
import requests
from history_manager import UploadHistoryManager
import base64
from concurrent.futures import ThreadPoolExecutor
import queue
import re
import html

# Add parent directories to path
current_dir = os.path.dirname(__file__)
parent_dir = os.path.join(current_dir, '..')
helpers_dir = os.path.join(parent_dir, 'helpers')

sys.path.insert(0, parent_dir)
sys.path.insert(0, helpers_dir)

try:
    from helpers.multi_ai import setup_ai_models, get_ai_model_info, switch_ai_model
    from helpers.person_detector import load_person_detector
except ImportError:
    # Fallback imports for development
    try:
        import multi_ai as multi_ai_module
        setup_ai_models = multi_ai_module.setup_ai_models
        get_ai_model_info = multi_ai_module.get_ai_model_info
        switch_ai_model = multi_ai_module.switch_ai_model
        
        import person_detector as person_detector_module
        load_person_detector = person_detector_module.load_person_detector
    except ImportError:
        print("⚠️ Warning: AI modules not available. Some features will be limited.")
        setup_ai_models = lambda: None
        get_ai_model_info = lambda: None
        switch_ai_model = lambda x: False
        load_person_detector = lambda: None

try:
    from helpers.guardian_matrix_adapter import GuardianMatrixLiveStreamManager
    from src.pipeline import GuardianMatrixPipeline
    from src.policy.safety_policy import PolicyState
    GUARDIAN_MATRIX_AVAILABLE = True
except Exception as gm_err:
    print(f"⚠️ Guardian Matrix initialization notice: {gm_err}")
    GUARDIAN_MATRIX_AVAILABLE = False
    GuardianMatrixLiveStreamManager = None
    GuardianMatrixPipeline = None
    PolicyState = None

app = Flask(__name__)
# Use environment variable for secret key in production
app.secret_key = os.environ.get('FLASK_SECRET_KEY', 'dev_key_change_in_production_' + str(int(time.time())))
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max file size

# CSRF protection disabled for API compatibility
csrf = None
print("✅ API endpoints ready (CSRF disabled for compatibility)")

# Global variables with thread safety
import threading

ai_manager = None
person_model = None
current_analysis = None
camera_stream = None
history_manager = UploadHistoryManager()
active_camera = None
camera_monitoring_active = False
parallel_analyzer = None
video_chunker = None
selected_camera_id = 0  # Track which camera is being used

# Thread-safe global state
analysis_lock = threading.Lock()
camera_lock = threading.Lock()

# Rate limiting for expensive operations
from collections import defaultdict
rate_limit_data = defaultdict(list)
ANALYSIS_RATE_LIMIT = 3  # Max 3 analysis requests per minute per IP
RATE_LIMIT_WINDOW = 60  # 60 seconds

camera_analysis_data = {
    'frame_count': 0,
    'people_detected': 0,
    'ai_checks': 0,
    'alerts': 0,
    'session_start': None,
    'performance_metrics': {
        'avg_frame_time': 0,
        'avg_detection_time': 0,
        'avg_ai_time': 0,
        'dropped_frames': 0
    }
}

# Allowed extensions
ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'wmv'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def check_rate_limit(ip_address, operation='analysis'):
    """Check if IP has exceeded rate limit for expensive operations"""
    current_time = time.time()
    
    # Clean old entries
    rate_limit_data[ip_address] = [
        timestamp for timestamp in rate_limit_data[ip_address]
        if current_time - timestamp < RATE_LIMIT_WINDOW
    ]
    
    # Check if under limit
    if len(rate_limit_data[ip_address]) >= ANALYSIS_RATE_LIMIT:
        return False
    
    # Add current request
    rate_limit_data[ip_address].append(current_time)
    return True

def sanitize_input(value, max_length=255):
    """Sanitize user input to prevent XSS and other attacks"""
    if not value:
        return value
    
    # Convert to string and limit length
    value = str(value)[:max_length]
    
    # HTML escape to prevent XSS
    value = html.escape(value)
    
    # Remove potentially dangerous characters
    value = re.sub(r'[<>"\'\&\$\`]', '', value)
    
    return value.strip()

def validate_ip_address(ip):
    """Validate IP address format"""
    ip_pattern = r'^(?:(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)\.){3}(?:25[0-5]|2[0-4][0-9]|[01]?[0-9][0-9]?)$'
    return re.match(ip_pattern, ip) is not None

def validate_config_structure(config):
    """Validate configuration structure"""
    try:
        # Check required top-level keys
        if not isinstance(config, dict):
            return False
            
        if 'ai_model_settings' not in config:
            return False
            
        ai_settings = config['ai_model_settings']
        if not isinstance(ai_settings, dict):
            return False
            
        # Check required nested keys
        required_keys = ['active_model', 'harassment_detection_settings']
        for key in required_keys:
            if key not in ai_settings:
                return False
                
        # Validate harassment detection settings structure
        hds = ai_settings['harassment_detection_settings']
        if not isinstance(hds, dict):
            return False
            
        # Check for required nested objects
        required_objects = ['cost_saving_features', 'parallel_processing', 'video_chunking']
        for obj in required_objects:
            if obj not in hds or not isinstance(hds[obj], dict):
                return False
                
        return True
        
    except Exception as e:
        print(f"Config validation error: {e}")
        return False

def init_models():
    """Initialize AI models, parallel analyzer, and video chunker"""
    global ai_manager, person_model, parallel_analyzer, video_chunker
    try:
        ai_manager = setup_ai_models()
        person_model = load_person_detector()
        
        # Initialize parallel analyzer
        sys.path.append(helpers_dir)
        from parallel_analyzer import ParallelFrameAnalyzer
        from video_chunker import VideoChunker
        
        parallel_analyzer = ParallelFrameAnalyzer()
        parallel_analyzer.set_models(person_model, ai_manager)
        
        video_chunker = VideoChunker()
        
        print("✅ AI models, parallel analyzer, and video chunker initialized")
        return True
    except Exception as e:
        print(f"Error initializing models: {e}")
        return False

def generate_camera_frames(camera_id=0):
    """Generate real-time camera frames with AI analysis"""
    global active_camera, camera_monitoring_active, camera_analysis_data
    
    print(f"🎥 Video stream requested for camera {camera_id}")
    
    # Wait for monitoring to be active
    max_wait = 10  # 10 seconds max wait
    wait_count = 0
    while not camera_monitoring_active and wait_count < max_wait:
        print(f"⏳ Waiting for camera monitoring to activate... ({wait_count}/10)")
        time.sleep(1)
        wait_count += 1
    
    if not camera_monitoring_active:
        print("❌ Camera monitoring not active, cannot start video stream")
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera monitoring not active\r\n'
        return
    
    try:
        # Initialize camera with optimized settings
        print(f"🎥 Initializing camera {camera_id} for video stream...")
        
        # Use DirectShow backend on Windows for better compatibility
        if os.name == 'nt':  # Windows
            camera = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            camera = cv2.VideoCapture(camera_id)
            
        # Wait for camera to initialize
        time.sleep(0.5)
        
        if not camera.isOpened():
            print(f"❌ Camera {camera_id} could not be opened for streaming")
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera not available for streaming\r\n'
            return
        
        # Test frame reading
        test_ret, test_frame = camera.read()
        if not test_ret or test_frame is None:
            print(f"❌ Camera {camera_id} cannot read frames for streaming")
            camera.release()
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera cannot read frames\r\n'
            return
        
        # Set optimal camera settings for streaming
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        camera.set(cv2.CAP_PROP_FPS, 15)
        camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        
        print(f"✅ Camera {camera_id} initialized successfully for streaming")
        
        active_camera = camera
        camera_monitoring_active = True
        camera_analysis_data['session_start'] = time.time()
        
        print("📹 Camera optimized: 640x480 @ 15 FPS, buffer=1")
        
        # Initialize AI models if not already done
        if not ai_manager or not person_model:
            init_models()
        
        gm_pipeline = None
        if GUARDIAN_MATRIX_AVAILABLE and GuardianMatrixPipeline is not None:
            try:
                gm_pipeline = GuardianMatrixPipeline(video_id=f"LIVE_CAM_{camera_id}")
                print(f"🚀 Guardian Matrix pipeline active for camera {camera_id} stream")
            except Exception as gm_err:
                print(f"⚠️ Guardian Matrix initialization error: {gm_err}")
                gm_pipeline = None

        frame_count = 0
        last_ai_check = 0
        stream_start_t = time.time()
        
        while camera_monitoring_active:
            success, frame = camera.read()
            if not success:
                break
            
            frame_count += 1
            camera_analysis_data['frame_count'] = frame_count

            # If Guardian Matrix is available, process through research pipeline
            if gm_pipeline is not None:
                current_time = time.time() - stream_start_t
                try:
                    res = gm_pipeline.process_frame(
                        frame,
                        timestamp=current_time,
                        frame_id=frame_count,
                        annotate=True
                    )
                    if res.annotated_frame is not None:
                        frame = res.annotated_frame

                    with camera_lock:
                        camera_analysis_data['people_detected'] = len(res.tracked_people)
                        camera_analysis_data['ai_checks'] = frame_count
                        camera_analysis_data['latest_ai_analysis'] = {
                            'frame': frame_count,
                            'people_count': len(res.tracked_people),
                            'pattern': res.classified_event.value,
                            'policy_state': res.policy_action.state.value,
                            'confidence': res.system1_decision.calibrated_confidence,
                            'ai_message': f"Pattern: {res.classified_event.value} | State: {res.policy_action.state.value}",
                            'is_harassment': (res.policy_action.state == PolicyState.ALERT if PolicyState else False),
                            'timestamp': time.time()
                        }

                        if PolicyState and res.policy_action.state in (PolicyState.ALERT, PolicyState.REVIEW):
                            if 'alerts_list' not in camera_analysis_data:
                                camera_analysis_data['alerts_list'] = []
                            camera_analysis_data['alerts'] = len(camera_analysis_data['alerts_list']) + 1
                            timestamp = time.strftime("%Y%m%d_%H%M%S")
                            evidence_filename = f"camera_alert_{timestamp}.jpg"
                            try:
                                cv2.imwrite(evidence_filename, frame)
                            except Exception:
                                pass
                            camera_analysis_data['alerts_list'].append({
                                'alert_id': camera_analysis_data['alerts'],
                                'frame': frame_count,
                                'timestamp': timestamp,
                                'ai_message': res.policy_action.rationale,
                                'people_count': len(res.tracked_people),
                                'evidence_file': evidence_filename,
                                'created_at': time.time()
                            })
                except Exception as gm_proc_err:
                    print(f"Guardian Matrix frame processing error: {gm_proc_err}")
            else:
                # Legacy fallback processing
                if frame_count % 15 == 0:
                    try:
                        if frame_count == 15:
                            sys.path.append(os.path.join(parent_dir, 'helpers'))
                            from person_detector import find_people_in_frame, draw_person_boxes
                            globals()['find_people_in_frame'] = find_people_in_frame
                            globals()['draw_person_boxes'] = draw_person_boxes
                        people_found = globals()['find_people_in_frame'](person_model, frame)
                        with camera_lock:
                            camera_analysis_data['people_detected'] = len(people_found)
                        if len(people_found) > 0:
                            frame = globals()['draw_person_boxes'](frame, people_found)
                    except Exception as e:
                        print(f"Analysis error: {e}")

                    print(f"Analysis error: {e}")
            
            # Encode frame to JPEG with proper compression
            encode_params = [cv2.IMWRITE_JPEG_QUALITY, 80]  # Good quality for streaming
            ret, buffer = cv2.imencode('.jpg', frame, encode_params)
            
            if ret:
                frame_bytes = buffer.tobytes()
                # Proper MJPEG boundary format for browser compatibility
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n'
                       b'Content-Length: ' + str(len(frame_bytes)).encode() + b'\r\n'
                       b'\r\n' + frame_bytes + b'\r\n')
            else:
                print("❌ Failed to encode frame")
            
            # Control frame rate for smooth streaming
            time.sleep(0.066)  # ~15 FPS for smooth performance
        
        camera.release()
        active_camera = None
        
    except Exception as e:
        print(f"Camera stream error: {e}")
        yield b'--frame\r\nContent-Type: text/plain\r\n\r\nCamera error\r\n'

def get_available_cameras():
    """Detect available cameras with robust error handling"""
    cameras = []
    
    try:
        print("🔍 Scanning for available cameras (suppressing OpenCV warnings)...")
        
        # Suppress OpenCV errors during camera scanning
        old_log_level = cv2.getLogLevel()
        cv2.setLogLevel(0)
        
        # Only check camera 0 and 1 to avoid the range errors
        for i in range(2):
            cap = None
            try:
                # Use DirectShow backend on Windows for better compatibility
                if os.name == 'nt':  # Windows
                    cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
                else:
                    cap = cv2.VideoCapture(i)
                
                # Brief initialization delay (removed invalid timeout property)
                time.sleep(0.1)
                
                if cap.isOpened():
                    # Try to read a frame to verify camera works
                    ret, frame = cap.read()
                    if ret and frame is not None:
                        # Get camera properties
                        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 640)
                        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 480)
                        fps = cap.get(cv2.CAP_PROP_FPS)
                        
                        # Handle FPS detection issues (common on Windows)
                        if fps <= 0 or fps > 120:
                            fps = 30  # Default to 30 FPS
                        
                        cameras.append({
                            'id': i,
                            'name': f'Camera {i}' + (' (Built-in)' if i == 0 else ''),
                            'type': 'DirectShow' if os.name == 'nt' else 'USB/Built-in',
                            'status': 'Available',
                            'resolution': f'{width}x{height}',
                            'fps': int(fps),
                            'backend': 'DirectShow' if os.name == 'nt' else 'Default'
                        })
                        print(f"  ✅ Camera {i} detected: {width}x{height} @ {int(fps)}fps")
                        
            except Exception:
                # Silently skip cameras that cause errors
                pass
            finally:
                # Always release the camera
                if cap is not None:
                    try:
                        cap.release()
                    except:
                        pass
                    time.sleep(0.05)  # Shorter delay
        
        # Restore OpenCV log level
        cv2.setLogLevel(old_log_level)
        
        print(f"📹 Found {len(cameras)} available cameras")
        
        # Always ensure at least one camera option is available
        if not cameras:
            print("⚠️ No cameras auto-detected, adding default camera option")
            cameras.append({
                'id': 0,
                'name': 'Camera 0 (Default)',
                'type': 'System Default',
                'status': 'Available',
                'resolution': '640x480',
                'fps': 30,
                'backend': 'DirectShow'
            })
        
        return cameras
        
    except Exception as e:
        print(f"❌ Camera detection error: {e}")
        # Return default camera as fallback
        return [{
            'id': 0,
            'name': 'Camera 0 (Fallback)',
            'type': 'System Default',
            'status': 'Available',
            'resolution': '640x480',
            'fps': 30,
            'backend': 'DirectShow'
        }]

def test_esp32_camera(ip_address):
    """Test ESP32 camera connection"""
    try:
        # Try common ESP32 camera endpoints
        endpoints = [
            f"http://{ip_address}/cam-hi.jpg",
            f"http://{ip_address}/capture",
            f"http://{ip_address}:81/stream",
            f"http://{ip_address}/stream"
        ]
        
        for endpoint in endpoints:
            try:
                response = requests.get(endpoint, timeout=5)
                if response.status_code == 200:
                    return True, endpoint
            except:
                continue
        
        return False, None
    except Exception as e:
        return False, str(e)

@app.route('/')
def index():
    """Main dashboard"""
    try:
        # Get system status
        model_info = get_ai_model_info() if ai_manager else None
        cameras = get_available_cameras()
        
        # Load configuration
        config = load_config()
        
        # Get history statistics
        history_stats = history_manager.get_statistics()
        recent_uploads = history_manager.get_all_uploads()[-3:]  # Last 3 uploads
        recent_uploads.reverse()  # Show newest first
        
        return render_template('index.html', 
                             model_info=model_info,
                             cameras=cameras,
                             config=config,
                             history_stats=history_stats,
                             recent_uploads=recent_uploads)
    except Exception as e:
        print(f"Dashboard error: {e}")
        flash('Error loading dashboard. Please refresh the page.', 'error')
        return render_template('error.html',
                             error_code=500,
                             error_message="Dashboard Error",
                             error_description="Could not load dashboard data."), 500

@app.route('/upload')
def upload_page():
    """Video upload page"""
    try:
        # Get recent uploads for display
        recent_uploads = history_manager.get_all_uploads()[-5:]  # Last 5 uploads
        recent_uploads.reverse()  # Show newest first
        return render_template('upload.html', recent_uploads=recent_uploads)
    except Exception as e:
        print(f"Upload page error: {e}")
        flash('Error loading upload page. Please refresh.', 'error')
        return render_template('error.html',
                             error_code=500,
                             error_message="Upload Page Error",
                             error_description="Could not load upload interface."), 500

@app.route('/upload_video', methods=['POST'])
def upload_video():
    """Handle video upload"""
    try:
        if 'video' not in request.files:
            return jsonify({'success': False, 'error': 'No video file selected'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No video file selected'}), 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            new_filename = f"{timestamp}_{filename}"
            
            # Create uploads directory if it doesn't exist
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Read file data
            file_data = file.read()
            file_size = len(file_data)
            
            # Validate file size
            if file_size > app.config['MAX_CONTENT_LENGTH']:
                return jsonify({'success': False, 'error': 'File too large. Maximum size is 500MB.'}), 413
            
            # Add to history with file data for organized storage
            upload_id = history_manager.add_upload(new_filename, file_size, file_data)
            history_manager.add_log_entry(upload_id, "info", f"Video uploaded and organized: {filename} ({file_size/1024/1024:.2f} MB)")
            
            # Return JSON response for JavaScript frontend
            return jsonify({
                'success': True,
                'message': f'Video uploaded successfully: {filename}',
                'filename': new_filename,
                'analysis_url': url_for('analyze_video', filename=new_filename)
            })
            
        else:
            # Return JSON error for invalid file type
            return jsonify({
                'success': False,
                'error': 'Invalid file type. Please upload MP4, AVI, MOV, MKV, or WMV files.'
            }), 400
            
    except Exception as e:
        print(f"Upload error: {e}")
        return jsonify({
            'success': False,
            'error': f'Upload failed: {str(e)}'
        }), 500

@app.route('/analyze/<filename>')
def analyze_video(filename):
    """Video analysis page"""
    # Find upload by filename
    upload = None
    for u in history_manager.get_all_uploads():
        if u['filename'] == filename:
            upload = u
            break
    
    if not upload:
        flash('Upload record not found', 'error')
        return redirect(url_for('upload_page'))
    
    # Get the actual video file path from organized storage
    file_path = upload.get('video_path')
    if not file_path or not os.path.exists(file_path):
        # Try backup location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            flash('Video file not found in storage', 'error')
            return redirect(url_for('upload_page'))
    
    # Get video info
    cap = cv2.VideoCapture(file_path)
    if cap.isOpened():
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        duration = frame_count / fps if fps > 0 else 0
        
        video_info = {
            'filename': filename,
            'upload_id': upload['id'],
            'frame_count': frame_count,
            'fps': fps,
            'duration': duration,
            'file_path': file_path
        }
        cap.release()
    else:
        flash('Could not read video file', 'error')
        return redirect(url_for('upload_page'))
    
    return render_template('analyze.html', video_info=video_info)

@app.route('/start_analysis', methods=['POST'])
def start_analysis():
    """Start video analysis"""
    global current_analysis
    
    # Validate JSON request
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    if not request.json:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    
    # Check rate limit
    client_ip = request.environ.get('HTTP_X_FORWARDED_FOR', request.remote_addr)
    if not check_rate_limit(client_ip):
        return jsonify({'error': 'Rate limit exceeded. Please wait before starting another analysis.'}), 429
    
    filename = sanitize_input(request.json.get('filename'))
    if not filename or not isinstance(filename, str):
        return jsonify({'error': 'Valid filename is required'}), 400
    
    # Find upload by filename
    upload = None
    for u in history_manager.get_all_uploads():
        if u['filename'] == filename:
            upload = u
            break
    
    if not upload:
        return jsonify({'error': 'Upload record not found'}), 404
    
    # Get the actual video file path
    file_path = upload.get('video_path')
    if not file_path or not os.path.exists(file_path):
        # Try backup location
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        if not os.path.exists(file_path):
            return jsonify({'error': 'Video file not found in storage'}), 404
    
    # Get analysis settings from request
    settings = request.json.get('settings', {})
    
    # Update analysis status
    history_manager.update_upload_status(upload['id'], 'running')
    history_manager.add_log_entry(upload['id'], 'info', f'Analysis started from web interface with mode: {settings.get("mode", "standard")}')
    
    # Store settings in current_analysis for the background thread
    current_analysis = {
        'status': 'starting',
        'filename': filename,
        'upload_id': upload['id'],
        'settings': settings,
        'start_time': datetime.now().isoformat()
    }
    
    # Start analysis in background thread
    analysis_thread = threading.Thread(target=run_video_analysis, args=(file_path, filename, upload['id']))
    analysis_thread.daemon = True
    analysis_thread.start()
    
    return jsonify({'message': 'Analysis started', 'status': 'running'})

@app.route('/analysis_status')
def analysis_status():
    """Get current analysis status"""
    global current_analysis
    
    with analysis_lock:
        if current_analysis is None:
            return jsonify({'status': 'idle'})
        
        # Make sure alerts are included in the response
        status_data = current_analysis.copy()
        
        # Ensure alerts array is properly included
        if 'alerts' not in status_data:
            status_data['alerts'] = []
        
        # Add debug info
        status_data['debug_info'] = {
            'alerts_count': len(status_data.get('alerts', [])),
            'has_alerts': len(status_data.get('alerts', [])) > 0,
            'timestamp': datetime.now().isoformat()
        }
        
        return jsonify(status_data)

@app.route('/camera')
def camera_page():
    """Camera selection and monitoring page"""
    try:
        print("📹 Loading camera page...")
        
        # Always suppress OpenCV warnings during camera detection
        cv2.setLogLevel(0)
        
        cameras = get_available_cameras()
        
        print(f"📹 Camera page loaded with {len(cameras)} cameras")
        
        if not cameras:
            flash('No cameras detected. Camera 0 will be available for testing.', 'info')
        else:
            flash(f'Found {len(cameras)} camera(s). Use "Test Camera" to verify functionality.', 'success')
        
        return render_template('camera.html', cameras=cameras)
        
    except Exception as e:
        print(f"❌ Camera page error: {e}")
        
        # Create a fallback camera list
        fallback_cameras = [{
            'id': 0,
            'name': 'Camera 0 (Fallback)',
            'type': 'System Default',
            'status': 'Testing Required',
            'resolution': '640x480',
            'fps': 30,
            'backend': 'DirectShow'
        }]
        
        flash('Camera detection had issues, but Camera 0 is available for testing.', 'warning')
        
        return render_template('camera.html', 
                             cameras=fallback_cameras,
                             error_message=f"Camera detection warning: {str(e)}")

@app.route('/start_camera_monitoring', methods=['POST'])
def start_camera_monitoring():
    """Start live camera monitoring - simplified version"""
    try:
        print(f"📥 Camera monitoring request received")
        
        # Get camera parameters with defaults
        camera_id = 0  # Default to camera 0
        camera_type = 'local'
        
        # Try to get parameters from JSON if available
        try:
            if request.is_json and request.json:
                camera_id = int(request.json.get('camera_id', 0))
                camera_type = request.json.get('camera_type', 'local')
                print(f"   Using JSON params: camera_id={camera_id}, camera_type={camera_type}")
            else:
                print(f"   Using defaults: camera_id={camera_id}, camera_type={camera_type}")
        except:
            print(f"   JSON parse failed, using defaults")
        
        # Simple camera validation
        if camera_id < 0 or camera_id > 5:
            camera_id = 0  # Reset to safe default
        
        print(f"🚀 Starting camera monitoring with camera {camera_id}")
        
        # Store the selected camera globally for video stream
        global selected_camera_id, camera_monitoring_active
        with camera_lock:
            selected_camera_id = camera_id
            camera_monitoring_active = True
            # Reset camera analysis data for new session
            camera_analysis_data['frame_count'] = 0
            camera_analysis_data['people_detected'] = 0
            camera_analysis_data['ai_checks'] = 0
            camera_analysis_data['alerts'] = 0
            camera_analysis_data['session_start'] = time.time()
            
            # Clear any existing alerts
            if 'alerts_list' in camera_analysis_data:
                camera_analysis_data['alerts_list'] = []
            
        print(f"✅ Camera monitoring state set: camera_id={camera_id}, active={camera_monitoring_active}")
        print(f"📊 Session data reset for new monitoring session")
        
        return jsonify({
            'message': f'Camera {camera_id} monitoring started',
            'status': 'running',
            'camera_id': camera_id,
            'camera_type': camera_type,
            'success': True
        })
        
    except Exception as e:
        print(f"❌ Camera monitoring error: {e}")
        return jsonify({
            'error': f'Camera monitoring failed: {str(e)}',
            'success': False
        }), 500

# Add a simple camera start endpoint without validation
@app.route('/camera/start', methods=['GET', 'POST'])
def simple_camera_start():
    """Simple camera start endpoint without strict validation"""
    try:
        print("📥 Simple camera start request")
        
        # Default to camera 0
        camera_id = 0
        
        # Try to get camera ID from request
        if request.method == 'POST':
            if request.is_json and request.json:
                camera_id = request.json.get('camera_id', 0)
            elif request.form:
                camera_id = int(request.form.get('camera_id', 0))
        
        # Store selected camera
        global selected_camera_id, camera_monitoring_active
        with camera_lock:
            selected_camera_id = camera_id
            camera_monitoring_active = True
        
        print(f"✅ Camera {camera_id} monitoring started (simple)")
        
        return jsonify({
            'success': True,
            'message': f'Camera {camera_id} started',
            'camera_id': camera_id,
            'status': 'running'
        })
        
    except Exception as e:
        print(f"❌ Simple camera start error: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/stop_camera_monitoring', methods=['POST'])
def stop_camera_monitoring():
    """Stop live camera monitoring"""
    global camera_monitoring_active, active_camera
    try:
        camera_monitoring_active = False
        if active_camera:
            active_camera.release()
            active_camera = None
        
        return jsonify({
            'message': 'Camera monitoring stopped',
            'status': 'stopped'
        })
    except Exception as e:
        return jsonify({'error': f'Failed to stop camera monitoring: {str(e)}'}), 500

@app.route('/video_stream')
def video_stream():
    """Real-time video stream with AI analysis"""
    from flask import Response
    
    print("📹 Video stream endpoint accessed")
    
    def generate():
        try:
            # Use the selected camera ID instead of hardcoded 0
            global selected_camera_id, camera_monitoring_active
            
            with camera_lock:
                camera_id = selected_camera_id
                monitoring_active = camera_monitoring_active
            
            print(f"📹 Video stream: camera_id={camera_id}, monitoring_active={monitoring_active}")
            
            if not monitoring_active:
                print("⚠️ Video stream requested but monitoring not active")
                yield b'--frame\r\nContent-Type: text/plain\r\n\r\nMonitoring not started yet\r\n'
                return
            
            print(f"🎬 Starting video stream generator for camera {camera_id}")
            frame_count = 0
            for frame in generate_camera_frames(camera_id):
                frame_count += 1
                if frame_count % 30 == 0:  # Log every 30 frames
                    print(f"📺 Video stream: {frame_count} frames streamed")
                yield frame
                
        except Exception as e:
            print(f"❌ Video stream generator error: {e}")
            import traceback
            traceback.print_exc()
            yield b'--frame\r\nContent-Type: text/plain\r\n\r\nStream generator error\r\n'
    
    try:
        print("🌐 Creating video stream response...")
        return Response(generate(),
                       mimetype='multipart/x-mixed-replace; boundary=frame',
                       headers={
                           'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0',
                           'Connection': 'close'
                       })
    except Exception as e:
        print(f"❌ Video stream setup error: {e}")
        return jsonify({'error': 'Failed to initialize video stream'}), 500

@app.route('/camera_status')
def camera_status():
    """Get real-time camera monitoring status"""
    global camera_analysis_data, camera_monitoring_active
    
    with camera_lock:  # Thread safety for camera data
        try:
            if camera_monitoring_active and camera_analysis_data.get('session_start'):
                session_duration = time.time() - camera_analysis_data['session_start']
                
                status_data = {
                    'status': 'monitoring',
                    'frame_count': camera_analysis_data.get('frame_count', 0),
                    'people_detected': camera_analysis_data.get('people_detected', 0),
                    'ai_checks': camera_analysis_data.get('ai_checks', 0),
                    'alerts': camera_analysis_data.get('alerts', 0),
                    'session_duration': int(session_duration),
                    'fps': round(camera_analysis_data.get('frame_count', 0) / session_duration) if session_duration > 0 else 0
                }
                
                # Add latest AI analysis if available
                if 'latest_ai_analysis' in camera_analysis_data:
                    latest = camera_analysis_data['latest_ai_analysis']
                    status_data['latest_analysis'] = {
                        'frame': latest.get('frame', 0),
                        'people_count': latest.get('people_count', 0),
                        'ai_message': latest.get('ai_message', ''),
                        'is_harassment': latest.get('is_harassment', False),
                        'timestamp': latest.get('timestamp', 0)
                    }
                
                # Add alerts list if available (ensure it's a list)
                alerts_list = camera_analysis_data.get('alerts_list', [])
                if alerts_list and isinstance(alerts_list, list):
                    status_data['alerts_list'] = alerts_list
                else:
                    status_data['alerts_list'] = []
                
                # Add debug info
                status_data['debug'] = {
                    'has_latest_analysis': 'latest_ai_analysis' in camera_analysis_data,
                    'alerts_list_length': len(status_data['alerts_list']),
                    'timestamp': datetime.now().isoformat()
                }
                
                return jsonify(status_data)
            else:
                return jsonify({
                    'status': 'stopped',
                    'frame_count': 0,
                    'people_detected': 0,
                    'ai_checks': 0,
                    'alerts': 0,
                    'alerts_list': []
                })
                
        except Exception as e:
            print(f"Camera status error: {e}")
            return jsonify({
                'status': 'error',
                'error': str(e),
                'alerts_list': []
            }), 500

@app.route('/test_esp32', methods=['POST'])
def test_esp32():
    """Test ESP32 camera connection"""
    # Validate JSON request
    if not request.is_json:
        return jsonify({'error': 'Content-Type must be application/json'}), 400
    
    if not request.json:
        return jsonify({'error': 'Invalid JSON payload'}), 400
    
    ip_address = sanitize_input(request.json.get('ip_address'))
    
    if not ip_address or not isinstance(ip_address, str) or not validate_ip_address(ip_address):
        return jsonify({'error': 'Valid IP address is required'}), 400
    
    success, endpoint = test_esp32_camera(ip_address)
    
    if success:
        return jsonify({
            'success': True,
            'message': f'ESP32 camera connected successfully',
            'endpoint': endpoint
        })
    else:
        return jsonify({
            'success': False,
            'message': 'Could not connect to ESP32 camera',
            'error': endpoint
        })

@app.route('/config')
def config_page():
    """Configuration management page"""
    config = load_config()
    model_info = get_ai_model_info() if ai_manager else None
    return render_template('config.html', config=config, model_info=model_info)

@app.route('/debug')
def debug_page():
    """Debug and testing page"""
    return render_template('debug.html')

@app.route('/test_camera/<int:camera_id>')
def test_camera(camera_id):
    """Test specific camera functionality"""
    try:
        print(f"🧪 Testing camera {camera_id}...")
        
        # Test camera with DirectShow
        if os.name == 'nt':
            cap = cv2.VideoCapture(camera_id, cv2.CAP_DSHOW)
        else:
            cap = cv2.VideoCapture(camera_id)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                # Get properties
                width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fps = cap.get(cv2.CAP_PROP_FPS)
                
                result = {
                    'success': True,
                    'camera_id': camera_id,
                    'width': width,
                    'height': height,
                    'fps': fps if fps > 0 else 30,
                    'frame_shape': frame.shape,
                    'message': f'Camera {camera_id} is working correctly'
                }
            else:
                result = {
                    'success': False,
                    'camera_id': camera_id,
                    'error': 'Cannot read frames from camera'
                }
        else:
            result = {
                'success': False,
                'camera_id': camera_id,
                'error': 'Cannot open camera'
            }
        
        cap.release()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'camera_id': camera_id,
            'error': str(e)
        }), 500

@app.route('/history')
def history_page():
    """Upload history page"""
    uploads = history_manager.get_all_uploads()
    stats = history_manager.get_statistics()
    return render_template('history.html', uploads=uploads, stats=stats)

@app.route('/history/<int:upload_id>')
def history_detail(upload_id):
    """Upload history detail page"""
    upload = history_manager.get_upload_by_id(upload_id)
    if not upload:
        flash('Upload not found', 'error')
        return redirect(url_for('history_page'))
    return render_template('history_detail.html', upload=upload)

@app.route('/delete_upload/<int:upload_id>', methods=['POST'])
def delete_upload(upload_id):
    """Delete upload and associated files"""
    success, message = history_manager.delete_upload(upload_id)
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    return redirect(url_for('history_page'))

@app.route('/clear_history', methods=['POST'])
def clear_history():
    """Clear all upload history"""
    success, message = history_manager.clear_all_history()
    if success:
        flash(message, 'success')
    else:
        flash(message, 'error')
    return redirect(url_for('history_page'))

@app.route('/download_evidence/<int:upload_id>/<filename>')
def download_evidence(upload_id, filename):
    """Download evidence file"""
    try:
        # Get upload info to find evidence files
        upload = history_manager.get_upload_by_id(upload_id)
        if not upload:
            flash('Upload not found', 'error')
            return redirect(url_for('history_page'))
        
        # Get evidence files from organized storage
        evidence_files = history_manager.get_evidence_files(upload_id)
        
        # Find evidence files
        if evidence_files:
            # If specific filename requested
            if not '*' in filename:
                for file_path in evidence_files:
                    if filename in file_path or os.path.basename(file_path) == filename:
                        if os.path.exists(file_path):
                            return send_file(file_path, as_attachment=True)
            else:
                # Handle pattern matching for alert files
                alert_pattern = filename.replace('*', '')
                for file_path in evidence_files:
                    if alert_pattern in file_path and os.path.exists(file_path):
                        return send_file(file_path, as_attachment=True)
        
        # If no specific file found, return the first available evidence file for this upload
        if evidence_files:
            for file_path in evidence_files:
                if os.path.exists(file_path):
                    return send_file(file_path, as_attachment=True)
        
        # If no file found, return error
        return jsonify({'error': 'Evidence file not found'}), 404
        
    except Exception as e:
        print(f"Evidence download error: {e}")
        return jsonify({'error': f'Error downloading evidence: {str(e)}'}), 500

@app.route('/download_report/<int:upload_id>')
def download_report(upload_id):
    """Download analysis report"""
    try:
        upload = history_manager.get_upload_by_id(upload_id)
        if not upload:
            flash('Upload not found', 'error')
            return redirect(url_for('history_page'))
        
        # Generate comprehensive report
        report_data = {
            'upload_id': upload_id,
            'filename': upload['original_filename'],
            'analysis_date': upload['upload_date'],
            'file_size_mb': upload['file_size'] / (1024 * 1024),
            'total_frames': upload['analysis_results']['total_frames'],
            'frames_processed': upload['analysis_results']['frames_processed'],
            'ai_checks': upload['analysis_results']['ai_checks'],
            'analysis_duration': upload['analysis_results']['analysis_duration'],
            'completion_percentage': upload['analysis_results']['completion_percentage'],
            'status': upload['analysis_status'],
            'alerts': upload['alerts'],
            'logs': upload['logs']
        }
        
        # Save report using file manager
        report_files = history_manager.save_analysis_report(upload_id, report_data)
        
        if report_files and 'text_report' in report_files:
            return send_file(report_files['text_report'], as_attachment=True)
        else:
            flash('Error generating report', 'error')
            return redirect(url_for('history_detail', upload_id=upload_id))
            
    except Exception as e:
        flash(f'Error downloading report: {str(e)}', 'error')
        return redirect(url_for('history_detail', upload_id=upload_id))

@app.route('/view_evidence/<int:upload_id>/<filename>')
def view_evidence(upload_id, filename):
    """View evidence file in browser"""
    try:
        evidence_files = history_manager.get_evidence_files(upload_id)
        for file_path in evidence_files:
            if filename in file_path:
                return send_file(file_path)
        
        flash('Evidence file not found', 'error')
        return redirect(url_for('history_detail', upload_id=upload_id))
    except Exception as e:
        flash(f'Error viewing evidence: {str(e)}', 'error')
        return redirect(url_for('history_detail', upload_id=upload_id))

@app.route('/get_video_path/<filename>')
def get_video_path(filename):
    """Get actual video file path for analysis"""
    try:
        # Find upload by filename
        upload = None
        for u in history_manager.get_all_uploads():
            if u['filename'] == filename:
                upload = u
                break
        
        if not upload:
            return jsonify({'error': 'Upload not found'}), 404
        
        file_path = upload.get('video_path')
        if not file_path or not os.path.exists(file_path):
            return jsonify({'error': 'Video file not found in storage'}), 404
        
        return jsonify({
            'file_path': file_path,
            'exists': True,
            'size': os.path.getsize(file_path)
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/update_config', methods=['POST'])
def update_config():
    """Update configuration"""
    try:
        # Validate JSON request
        if not request.is_json:
            return jsonify({'error': 'Content-Type must be application/json'}), 400
        
        if not request.json:
            return jsonify({'error': 'Invalid JSON payload'}), 400
            
        new_config = request.json
        
        # Update AI model if changed (check nested structure)
        if 'ai_model_settings' in new_config and 'active_model' in new_config['ai_model_settings']:
            active_model = new_config['ai_model_settings']['active_model']
            success = switch_ai_model(active_model)
            if not success:
                return jsonify({'error': f'Failed to switch to AI model: {active_model}'}), 400
        
        # Validate config structure before saving
        if not validate_config_structure(new_config):
            return jsonify({'error': 'Invalid configuration structure'}), 400
        
        # Save configuration
        if not save_config(new_config):
            return jsonify({'error': 'Failed to save configuration to file'}), 500
        
        print(f"✅ Configuration saved successfully: {len(str(new_config))} characters")
        return jsonify({'message': 'Configuration updated successfully'})
        
    except Exception as e:
        print(f"❌ Config update error: {e}")
        return jsonify({'error': f'Configuration update failed: {str(e)}'}), 500

def load_config():
    """Load current configuration"""
    try:
        config_path = os.path.join(parent_dir, 'ai_config.json')
        with open(config_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return {}

def save_config(config):
    """Save configuration"""
    try:
        config_path = os.path.join(parent_dir, 'ai_config.json')
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def run_video_analysis(file_path, filename, upload_id):
    """Run PARALLEL AI-powered video analysis in background"""
    global current_analysis, parallel_analyzer
    
    try:
        current_analysis = {
            'status': 'running',
            'filename': filename,
            'upload_id': upload_id,
            'progress': 0,
            'frame_count': 0,
            'ai_checks': 0,
            'alerts': [],
            'start_time': datetime.now().isoformat(),
            'parallel_enabled': parallel_analyzer.enabled if parallel_analyzer else False
        }
        
        # Log analysis start
        history_manager.add_log_entry(upload_id, 'info', f'CHUNKED PARALLEL AI analysis started for {filename}')
        
        # Check if video chunking is enabled
        config = load_config()
        chunking_enabled = config.get('ai_model_settings', {}).get('harassment_detection_settings', {}).get('video_chunking', {}).get('enabled', True)
        
        if chunking_enabled and video_chunker:
            history_manager.add_log_entry(upload_id, 'info', f'Using video chunking: {video_chunker.chunk_duration}s chunks, {video_chunker.max_parallel_chunks} parallel workers')
            
            # Use chunked parallel analysis
            try:
                # Define progress callback for chunks
                def chunk_progress_callback(completed_chunks, total_chunks, progress, chunk_result):
                    current_analysis.update({
                        'progress': progress,
                        'chunks_completed': completed_chunks,
                        'total_chunks': total_chunks,
                        'frame_count': sum(r.get('frames_processed', 0) for r in video_chunker.chunk_results),
                        'ai_checks': sum(r.get('ai_checks', 0) for r in video_chunker.chunk_results)
                    })
                    
                    elapsed_time = time.time() - time.mktime(datetime.fromisoformat(current_analysis['start_time']).timetuple())
                    history_manager.update_upload_status(upload_id, 'running', {
                        'chunks_completed': completed_chunks,
                        'total_chunks': total_chunks,
                        'frames_processed': current_analysis['frame_count'],
                        'ai_checks': current_analysis['ai_checks'],
                        'completion_percentage': progress,
                        'analysis_duration': f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                    })
                    
                    print(f"📊 Chunk Progress: {completed_chunks}/{total_chunks} chunks ({progress}%)")
                
                # Run chunked analysis
                results = video_chunker.analyze_video_chunks_parallel(
                    file_path, 
                    person_model, 
                    ai_manager,
                    progress_callback=chunk_progress_callback
                )
                
                # Process aggregated results
                current_analysis.update({
                    'progress': 100,
                    'frame_count': results['total_frames'],
                    'ai_checks': results['total_ai_checks'],
                    'status': 'completed'
                })
                
                # Add all alerts to current_analysis
                for alert in results['all_alerts']:
                    alert_for_frontend = {
                        'alert_id': len(current_analysis['alerts']) + 1,
                        'frame': alert['frame_in_original'],
                        'time': alert['time_formatted'],
                        'message': alert['ai_message'],
                        'confidence': alert['confidence'],
                        'people_count': alert['people_count'],
                        'chunk_number': alert['chunk_number']
                    }
                    current_analysis['alerts'].append(alert_for_frontend)
                    
                    # Also save to history
                    alert_data = {
                        'frame': alert['frame_in_original'],
                        'time': alert['time_formatted'],
                        'confidence': alert['confidence'],
                        'message': alert['ai_message'],
                        'ai_analysis': f'Chunked analysis detected harassment at {alert["time_formatted"]} (Chunk {alert["chunk_number"]})',
                        'people_count': alert['people_count']
                    }
                    history_manager.add_alert(upload_id, alert_data)
                
                history_manager.add_log_entry(upload_id, 'info', f'CHUNKED analysis completed: {results["parallel_efficiency"]:.1f}x speedup, {results["total_alerts"]} alerts found')
                
            except Exception as e:
                history_manager.add_log_entry(upload_id, 'error', f'Chunked analysis failed: {str(e)}')
                raise e
        
        else:
            # Fall back to sequential processing
            history_manager.add_log_entry(upload_id, 'info', 'Using sequential processing (chunking disabled)')
        
        # Open video file
        cap = cv2.VideoCapture(file_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video file: {file_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        history_manager.add_log_entry(upload_id, 'info', f'Video info: {total_frames} frames, {fps} FPS')
        
        # Initialize AI models (REAL MODELS)
        try:
            # Import the real AI analysis functions
            sys.path.append(os.path.join(parent_dir, 'helpers'))
            from person_detector import load_person_detector, find_people_in_frame, draw_person_boxes
            from multi_ai import setup_ai_models, analyze_for_harassment, count_people_for_analysis, get_ai_model_info
            
            person_model = load_person_detector()
            ai_manager = setup_ai_models()
            model_info = get_ai_model_info()
            
            history_manager.add_log_entry(upload_id, 'info', f'AI models loaded: {model_info["active_model"].upper()}')
            
        except Exception as e:
            history_manager.add_log_entry(upload_id, 'error', f'Failed to load AI models: {str(e)}')
            raise Exception(f"AI model initialization failed: {str(e)}")
        
        # Update analysis results
        history_manager.update_upload_status(upload_id, 'running', {
            'total_frames': total_frames
        })
        
        frame_number = 0
        alert_count = 0
        ai_check_count = 0
        
        # Get analysis mode from current_analysis if available
        analysis_settings = current_analysis.get('settings', {}) if current_analysis else {}
        analysis_mode = analysis_settings.get('mode', 'standard')
        
        # Set frame checking frequency based on mode
        if analysis_mode == 'fast':
            check_every_n_frames = 30  # Every 30 frames (~1 second)
        elif analysis_mode == 'thorough':
            check_every_n_frames = 1   # Every frame
        else:  # standard
            check_every_n_frames = 1   # Every frame, but only analyze when 2+ people
        
        history_manager.add_log_entry(upload_id, 'info', f'Analysis mode: {analysis_mode} (checking every {check_every_n_frames} frames)')
        
        # Process video frames with REAL AI ANALYSIS
        print(f"🎬 Starting REAL analysis of {total_frames} frames...")
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            current_analysis['frame_count'] = frame_number
            current_analysis['progress'] = int((frame_number / total_frames) * 100)
            current_video_time = frame_number / fps if fps > 0 else 0
            
            # Skip frames for performance (analyze every frame, but can be adjusted)
            if frame_number % check_every_n_frames != 0:
                continue
            
            # Save every 60th frame as sample
            if frame_number % 60 == 0:
                history_manager.save_frame(upload_id, frame_number, frame, "sample")
            
            # REAL PERSON DETECTION
            frame_start_time = time.time()
            people_found = find_people_in_frame(person_model, frame)
            people_detection_time = time.time() - frame_start_time
            
            # REAL AI ANALYSIS - Only when 2+ people present
            if len(people_found) >= 2:
                ai_check_count += 1
                current_analysis['ai_checks'] = ai_check_count
                
                print(f"🧠 AI analyzing frame {frame_number}/{total_frames} - {len(people_found)} people detected")
                history_manager.add_log_entry(upload_id, 'info', f'AI analyzing frame {frame_number} - {len(people_found)} people detected')
                
                # REAL AI HARASSMENT DETECTION
                ai_start_time = time.time()
                is_harassment, ai_message = analyze_for_harassment(ai_manager, frame)
                ai_analysis_time = time.time() - ai_start_time
                
                print(f"🤖 AI Response: {ai_message} (Harassment: {is_harassment})")
                
                if is_harassment:
                    alert_count += 1
                    time_in_video = f"{int(current_video_time//60):02d}:{int(current_video_time%60):02d}"
                    
                    print(f"🚨 HARASSMENT ALERT #{alert_count} at {time_in_video}: {ai_message}")
                    history_manager.add_log_entry(upload_id, 'alert', f'HARASSMENT DETECTED at frame {frame_number}: {ai_message}')
                    
                    alert_data = {
                        'frame': frame_number,
                        'time': time_in_video,
                        'confidence': 0.92,  # You can extract confidence from AI response if available
                        'message': ai_message,
                        'ai_analysis': f'Real AI analysis detected harassment behavior at {time_in_video}',
                        'people_count': len(people_found),
                        'analysis_time': ai_analysis_time
                    }
                    
                    # Draw detection boxes on frame for evidence
                    evidence_frame = draw_person_boxes(frame.copy(), people_found)
                    cv2.putText(evidence_frame, f"ALERT at {time_in_video}", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    cv2.putText(evidence_frame, ai_message[:60], (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    
                    # Save alert with evidence
                    alert_id = history_manager.add_alert(upload_id, alert_data, evidence_frame)
                    
                    # Add alert to current_analysis for frontend
                    alert_for_frontend = {
                        'alert_id': alert_id,
                        'frame': frame_number,
                        'time': time_in_video,
                        'message': ai_message,
                        'confidence': alert_data['confidence'],
                        'people_count': len(people_found),
                        'ai_analysis': alert_data['ai_analysis']
                    }
                    
                    current_analysis['alerts'].append(alert_for_frontend)
                    
                    print(f"📊 Alert added to current_analysis. Total alerts: {len(current_analysis['alerts'])}")
                    
                    # Check cost-saving auto-stop
                    config = load_config()
                    if config.get('ai_model_settings', {}).get('harassment_detection_settings', {}).get('cost_saving_features', {}).get('auto_stop_on_alert', True):
                        history_manager.add_log_entry(upload_id, 'info', 'Auto-stopping analysis due to REAL alert detection (cost-saving)')
                        break
                else:
                    # Log normal behavior for transparency
                    history_manager.add_log_entry(upload_id, 'info', f'Frame {frame_number}: Normal behavior detected - {ai_message}')
            
            # Update progress every 100 frames for smoother progress bar
            if frame_number % 100 == 0:
                elapsed_time = time.time() - time.mktime(datetime.fromisoformat(current_analysis['start_time']).timetuple())
                progress_percentage = int((frame_number / total_frames) * 100)
                
                # Update current_analysis for status endpoint
                current_analysis.update({
                    'progress': progress_percentage,
                    'frame_count': frame_number,
                    'ai_checks': ai_check_count,
                    'alerts': current_analysis.get('alerts', [])
                })
                
                history_manager.update_upload_status(upload_id, 'running', {
                    'frames_processed': frame_number,
                    'ai_checks': ai_check_count,
                    'completion_percentage': progress_percentage,
                    'analysis_duration': f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
                })
                
                print(f"📊 Progress: {progress_percentage}% ({frame_number}/{total_frames} frames, {ai_check_count} AI checks)")
                
                # Log progress less frequently to avoid spam
                if frame_number % 1000 == 0:  # Log every 1000 frames
                    history_manager.add_log_entry(upload_id, 'info', f'Progress: {frame_number}/{total_frames} frames ({progress_percentage}%)')
        
        cap.release()
        print(f"✅ Video analysis completed: {frame_number} frames processed, {ai_check_count} AI checks, {alert_count} alerts")
        
        # Final status update
        elapsed_time = time.time() - time.mktime(datetime.fromisoformat(current_analysis['start_time']).timetuple())
        final_frame_count = current_analysis.get('frame_count', 0)
        final_ai_checks = current_analysis.get('ai_checks', 0)
        final_progress = current_analysis.get('progress', 100)
        
        history_manager.update_upload_status(upload_id, 'completed', {
            'frames_processed': final_frame_count,
            'ai_checks': final_ai_checks,
            'alerts_found': len(current_analysis['alerts']),
            'completion_percentage': final_progress,
            'analysis_duration': f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}"
        })
        
        current_analysis['status'] = 'completed'
        current_analysis['end_time'] = datetime.now().isoformat()
        
        # Generate final report with REAL results
        report_data = {
            'upload_id': upload_id,
            'filename': filename,
            'analysis_date': current_analysis['start_time'],
            'file_size_mb': os.path.getsize(file_path) / (1024 * 1024),
            'total_frames': final_frame_count,
            'frames_processed': final_frame_count,
            'ai_checks': final_ai_checks,
            'analysis_duration': f"{int(elapsed_time//60):02d}:{int(elapsed_time%60):02d}",
            'completion_percentage': final_progress,
            'status': 'completed',
            'alerts': current_analysis['alerts'],
            'ai_model_used': 'gemini',  # Default to gemini
            'parallel_enabled': current_analysis.get('parallel_enabled', False),
            'cost_info': {
                'api_calls': final_ai_checks,
                'estimated_cost': final_ai_checks * 0.001,  # Approximate cost per API call
                'auto_stopped': final_progress < 100
            }
        }
        
        history_manager.save_analysis_report(upload_id, report_data)
        
        if len(current_analysis['alerts']) > 0:
            history_manager.add_log_entry(upload_id, 'alert', f'Analysis completed with {len(current_analysis["alerts"])} REAL harassment alerts detected')
        else:
            history_manager.add_log_entry(upload_id, 'info', 'Analysis completed - No harassment detected')
        
    except Exception as e:
        current_analysis = {
            'status': 'error',
            'error': str(e),
            'filename': filename,
            'upload_id': upload_id
        }
        
        # Log error
        history_manager.update_upload_status(upload_id, 'error')
        history_manager.add_log_entry(upload_id, 'error', f'REAL AI analysis failed: {str(e)}')

# Error handlers
@app.errorhandler(404)
def not_found_error(error):
    """Handle 404 errors"""
    return render_template('error.html', 
                         error_code=404,
                         error_message="Page not found",
                         error_description="The requested page could not be found."), 404

@app.errorhandler(500)
def internal_error(error):
    """Handle 500 errors"""
    return render_template('error.html',
                         error_code=500,
                         error_message="Internal server error",
                         error_description="An unexpected error occurred. Please try again."), 500

@app.errorhandler(400)
def bad_request_error(error):
    """Handle 400 errors"""
    # Return JSON for API endpoints
    if request.path.startswith('/start_') or request.path.startswith('/test_') or request.path.startswith('/update_'):
        return jsonify({
            'error': 'Bad request - invalid input',
            'message': 'The request was invalid. Please check your input.',
            'status_code': 400
        }), 400
    
    return render_template('error.html',
                         error_code=400,
                         error_message="Bad request",
                         error_description="The request was invalid. Please check your input."), 400

@app.errorhandler(413)
def file_too_large_error(error):
    """Handle file too large errors"""
    return render_template('error.html',
                         error_code=413,
                         error_message="File too large",
                         error_description="The uploaded file is too large. Maximum size is 500MB."), 413

@app.errorhandler(429)
def rate_limit_error(error):
    """Handle rate limit errors"""
    return render_template('error.html',
                         error_code=429,
                         error_message="Rate limit exceeded",
                         error_description="Too many requests. Please wait before trying again."), 429

if __name__ == '__main__':
    print("🛡️ Safety Detector Web Interface")
    print("=" * 40)
    print("✅ OpenCV warnings suppressed")
    print("✅ Camera error fixes applied")
    print("✅ Thread safety enabled")
    print("=" * 40)
    
    print("🤖 Initializing AI models...")
    if init_models():
        print("✅ Models initialized successfully")
    else:
        print("⚠️ Warning: Some models failed to initialize")
    
    print("🌐 Starting web server...")
    print("📱 Local access: http://localhost:5000")
    print("🔗 Network access: http://0.0.0.0:5000")
    print("=" * 40)
    print("Press Ctrl+C to stop the server")
    
    # Use debug=False for production-like environment
    app.run(host='0.0.0.0', port=5000, debug=False, threaded=True)
