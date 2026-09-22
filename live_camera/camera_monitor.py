# Live Camera Monitor for Safety Detection
# This script watches live camera feed and detects harassment

import cv2
import time
import sys
import os
import json
from datetime import datetime

# Add parent directories to path so we can import our modules
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, parent_dir)
sys.path.insert(0, os.path.join(parent_dir, 'helpers'))

from person_detector import load_person_detector, find_people_in_frame, draw_person_boxes
from multi_ai import setup_ai_models, analyze_for_harassment, count_people_for_analysis, get_ai_model_info

try:
    from src.pipeline import GuardianMatrixPipeline
    from src.policy.safety_policy import PolicyState
    GUARDIAN_MATRIX_AVAILABLE = True
except Exception:
    GUARDIAN_MATRIX_AVAILABLE = False

def start_camera():
    """Start the camera and return camera object"""
    print("Starting camera...")
    camera = cv2.VideoCapture(0)  # 0 means first camera
    
    if not camera.isOpened():
        print("ERROR: Could not open camera!")
        return None
    
    print("Camera started successfully!")
    return camera

def main():
    """Main function that runs the live monitoring"""
    print("=== Safety Detector - Live Camera Monitor ===")
    print("Controls:")
    print("  'q' - Quit monitoring")
    print("  's' - Save current frame")
    print("  'l' - Save current session log")
    print("  'c' - Continue after alert (cost-saving feature)")
    print("💰 Note: Monitoring pauses after alerts to save AI costs")
    print()
    
    # Setup everything
    camera = start_camera()
    if camera is None:
        return
    
    person_model = load_person_detector()
    ai_manager = setup_ai_models()
    
    # Show which AI model is being used
    model_info = get_ai_model_info()
    print(f"Using AI model: {model_info['active_model'].upper()}")
    print(f"Available models: {', '.join(model_info['available_models'])}")
    
    # Settings
    frame_count = 0
    last_ai_check = 0
    ai_check_interval = 3  # Check with AI every 30 frames (about every 1 second)
    
    # Create session tracking
    session_data = {
        'session_type': 'live_camera',
        'start_time': datetime.now().isoformat(),
        'frame_analysis': [],
        'alerts': [],
        'summary': {}
    }
    
    # Performance tracking
    total_people_detection_time = 0
    total_ai_analysis_time = 0
    total_frames_with_ai = 0
    gm_pipeline = None
    if GUARDIAN_MATRIX_AVAILABLE:
        try:
            gm_pipeline = GuardianMatrixPipeline(video_id="LIVE_CAM_MONITOR")
            print("🚀 Guardian Matrix research pipeline enabled for live monitoring!")
        except Exception as e:
            print(f"⚠️ Guardian Matrix init notice: {e}")
            gm_pipeline = None

    print("Starting live monitoring...")
    print("Looking for people and checking for harassment...")
    
    while True:
        # Start timing for this frame
        frame_start_time = time.time()
        
        # Read frame from camera
        success, frame = camera.read()
        if not success:
            print("Could not read from camera")
            break
        
        frame_count += 1
        
        if gm_pipeline is not None:
            current_t = time.time() - session_start_time
            res = gm_pipeline.process_frame(frame, timestamp=current_t, frame_id=frame_count, annotate=True)
            frame_with_boxes = res.annotated_frame if res.annotated_frame is not None else frame
            people_found = res.tracked_people
            people_detection_time = res.processing_time_ms / 1000.0
            total_people_detection_time += people_detection_time
            people_status = f"People: {len(people_found)} | Pattern: {res.classified_event.value} | State: {res.policy_action.state.value}"
        else:
            # Time people detection
            people_detection_start = time.time()
            people_found = find_people_in_frame(person_model, frame)
            people_detection_time = time.time() - people_detection_start
            total_people_detection_time += people_detection_time
            
            # Draw boxes around people
            frame_with_boxes = draw_person_boxes(frame.copy(), people_found)
            
            # Count people
            people_status = count_people_for_analysis(people_found)
        
        # Calculate performance metrics
        avg_people_detection = total_people_detection_time / frame_count
        avg_ai_analysis = total_ai_analysis_time / total_frames_with_ai if total_frames_with_ai > 0 else 0
        fps_actual = frame_count / (time.time() - session_start_time)
        
        # Add text to show what we found
        cv2.putText(frame_with_boxes, people_status, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(frame_with_boxes, f"Frame: {frame_count}", (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Add performance info to display
        cv2.putText(frame_with_boxes, f"Detection: {people_detection_time:.3f}s", (10, 400), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame_with_boxes, f"Avg AI: {avg_ai_analysis:.3f}s", (10, 420), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        cv2.putText(frame_with_boxes, f"FPS: {fps_actual:.1f}", (10, 440), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        
        # Frame data for logging
        frame_data = {
            'frame_number': frame_count,
            'timestamp': datetime.now().isoformat(),
            'people_count': len(people_found),
            'people_detection_time': people_detection_time,
            'ai_analysis_time': 0,
            'total_frame_time': 0,
            'has_alert': False,
            'ai_message': ''
        }
        
        # Check with AI every few seconds if we have people
        if len(people_found) >= 2 and frame_count - last_ai_check > ai_check_interval:
            print(f"Checking frame {frame_count} with AI...")
            
            # Time AI analysis
            ai_start_time = time.time()
            is_harassment, ai_message = analyze_for_harassment(None, frame)  # multi_ai handles model selection
            ai_analysis_time = time.time() - ai_start_time
            
            total_ai_analysis_time += ai_analysis_time
            total_frames_with_ai += 1
            last_ai_check = frame_count
            
            frame_data['ai_analysis_time'] = ai_analysis_time
            frame_data['ai_message'] = ai_message
            
            print(f"  AI analysis time: {ai_analysis_time:.2f} seconds")
            
            if is_harassment:
                frame_data['has_alert'] = True
                
                print(f"*** ALERT DETECTED *** {ai_message}")
                
                # Save alert evidence
                timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
                alert_filename = f"live_alert_{len(session_data['alerts'])+1}_{timestamp_str}.jpg"
                cv2.imwrite(alert_filename, frame_with_boxes)
                
                # Store alert data
                alert_data = {
                    'alert_number': len(session_data['alerts']) + 1,
                    'frame_number': frame_count,
                    'timestamp': datetime.now().isoformat(),
                    'people_count': len(people_found),
                    'ai_message': ai_message,
                    'evidence_file': alert_filename,
                    'analysis_time': ai_analysis_time
                }
                session_data['alerts'].append(alert_data)
                
                # Add red warning to screen
                cv2.putText(frame_with_boxes, "ALERT DETECTED!", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                cv2.putText(frame_with_boxes, ai_message[:50], (10, 130), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                
                # AUTO-PAUSE FEATURE: Show alert for longer time
                cv2.putText(frame_with_boxes, "Press 'c' to continue, 'q' to quit", (10, 160), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                cv2.putText(frame_with_boxes, "Monitoring paused to save costs", (10, 180), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
            else:
                print(f"Normal behavior: {ai_message}")
                cv2.putText(frame_with_boxes, "Monitoring...", (10, 100), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Complete frame timing
        frame_data['total_frame_time'] = time.time() - frame_start_time
        session_data['frame_analysis'].append(frame_data)
        
        # Keep only last 100 frames in memory (to prevent memory issues)
        if len(session_data['frame_analysis']) > 100:
            session_data['frame_analysis'] = session_data['frame_analysis'][-100:]
        
        # Show the video
        cv2.imshow('Safety Monitor - Live Camera', frame_with_boxes)
        
        # Check for key presses
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('q'):
            print("Stopping camera monitor...")
            break
        elif key == ord('s'):
            filename = f"saved_frame_{int(time.time())}.jpg"
            cv2.imwrite(filename, frame)
            print(f"Saved frame as {filename}")
        elif key == ord('l'):
            # Save current session log
            save_session_log(session_data, frame_count, session_start_time, 
                           total_people_detection_time, total_ai_analysis_time, total_frames_with_ai)
        elif key == ord('c'):
            # Continue monitoring after alert (clear any alert states)
            print("Continuing monitoring...")
            continue
    
    # Session complete - generate final summary
    session_end_time = time.time()
    session_data['end_time'] = datetime.now().isoformat()
    
    # Calculate final statistics
    total_session_time = session_end_time - session_start_time
    avg_people_detection = total_people_detection_time / frame_count if frame_count > 0 else 0
    avg_ai_analysis = total_ai_analysis_time / total_frames_with_ai if total_frames_with_ai > 0 else 0
    actual_fps = frame_count / total_session_time if total_session_time > 0 else 0
    
    session_data['summary'] = {
        'total_session_time_seconds': total_session_time,
        'frames_processed': frame_count,
        'frames_with_ai_analysis': total_frames_with_ai,
        'alerts_found': len(session_data['alerts']),
        'average_people_detection_time': avg_people_detection,
        'average_ai_analysis_time': avg_ai_analysis,
        'actual_fps': actual_fps
    }
    
    print()
    print("=== Live Camera Session Complete ===")
    print(f"Session duration: {total_session_time:.1f} seconds")
    print(f"Frames processed: {frame_count}")
    print(f"AI analyses performed: {total_frames_with_ai}")
    print(f"Alerts detected: {len(session_data['alerts'])}")
    print(f"Average FPS: {actual_fps:.1f}")
    print(f"Average people detection: {avg_people_detection:.3f} seconds")
    print(f"Average AI analysis: {avg_ai_analysis:.3f} seconds")
    
    # Save final session log
    save_session_log(session_data, frame_count, session_start_time, 
                   total_people_detection_time, total_ai_analysis_time, total_frames_with_ai)
    
    # Clean up
    camera.release()
    cv2.destroyAllWindows()
    print("Camera monitor stopped.")

def save_session_log(session_data, frame_count, session_start_time, 
                    total_people_detection_time, total_ai_analysis_time, total_frames_with_ai):
    """Save the current session data to log files"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Calculate current statistics
    current_time = time.time()
    session_duration = current_time - session_start_time
    avg_people_detection = total_people_detection_time / frame_count if frame_count > 0 else 0
    avg_ai_analysis = total_ai_analysis_time / total_frames_with_ai if total_frames_with_ai > 0 else 0
    current_fps = frame_count / session_duration if session_duration > 0 else 0
    
    # Update session data
    session_data['summary'] = {
        'session_duration_seconds': session_duration,
        'frames_processed': frame_count,
        'frames_with_ai_analysis': total_frames_with_ai,
        'alerts_found': len(session_data['alerts']),
        'average_people_detection_time': avg_people_detection,
        'average_ai_analysis_time': avg_ai_analysis,
        'current_fps': current_fps
    }
    
    # Save JSON log
    log_filename = f"live_session_log_{timestamp}.json"
    with open(log_filename, 'w') as f:
        json.dump(session_data, f, indent=2)
    
    # Save human-readable summary
    summary_filename = f"live_session_summary_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Safety Detector - Live Camera Session Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Session Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Session Duration: {session_duration:.1f} seconds\n\n")
        
        f.write(f"Performance Statistics:\n")
        f.write(f"  Frames Processed: {frame_count}\n")
        f.write(f"  Average FPS: {current_fps:.1f}\n")
        f.write(f"  AI Analyses: {total_frames_with_ai}\n")
        f.write(f"  Average People Detection: {avg_people_detection:.3f} seconds\n")
        f.write(f"  Average AI Analysis: {avg_ai_analysis:.3f} seconds\n\n")
        
        f.write(f"Alert Summary:\n")
        f.write(f"  Total Alerts: {len(session_data['alerts'])}\n")
        
        if session_data['alerts']:
            f.write(f"\nAlert Details:\n")
            for alert in session_data['alerts']:
                f.write(f"  Alert {alert['alert_number']}: Frame {alert['frame_number']}\n")
                f.write(f"    Time: {alert['timestamp']}\n")
                f.write(f"    People: {alert['people_count']}\n")
                f.write(f"    Message: {alert['ai_message']}\n")
                f.write(f"    Evidence: {alert['evidence_file']}\n")
                f.write(f"    Analysis Time: {alert['analysis_time']:.2f}s\n\n")
        else:
            f.write(f"  No alerts detected during this session.\n")
    
    print(f"Session log saved: {log_filename}")
    print(f"Session summary saved: {summary_filename}")

if __name__ == "__main__":
    main()
