# Video File Analyzer for Safety Detection
# This script analyzes existing video files for harassment

import cv2
import time
import sys
import os
import json
from datetime import datetime

# Add helpers folder to path so we can import our helper files
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'helpers'))

from person_detector import load_person_detector, find_people_in_frame, draw_person_boxes
from multi_ai import setup_ai_models, analyze_for_harassment, count_people_for_analysis, get_ai_model_info

def analyze_video_file(video_path):
    """Analyze a video file for harassment"""
    print(f"=== Analyzing Video: {video_path} ===")
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"ERROR: Video file not found: {video_path}")
        return
    
    # Open video file
    video = cv2.VideoCapture(video_path)
    if not video.isOpened():
        print(f"ERROR: Could not open video file: {video_path}")
        return
    
    # Get video information
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(video.get(cv2.CAP_PROP_FPS))
    duration = total_frames / fps if fps > 0 else 0
    
    print(f"Video Info:")
    print(f"  - Total frames: {total_frames}")
    print(f"  - FPS: {fps}")
    print(f"  - Duration: {duration:.1f} seconds")
    print()
    
    # Setup detection models
    person_model = load_person_detector()
    ai_manager = setup_ai_models()
    
    # Show which AI model is being used
    model_info = get_ai_model_info()
    print(f"Using AI model: {model_info['active_model'].upper()}")
    print(f"Available models: {', '.join(model_info['available_models'])}")
    print()
    
    # Analysis settings
    frame_count = 0
    ai_checks = 0
    alerts_found = 0
    check_every_n_frames = 1  # Check every frame (analyze 30 frames per second)
    
    # Create log and summary data
    analysis_session = {
        'video_path': video_path,
        'video_name': os.path.basename(video_path),
        'start_time': datetime.now().isoformat(),
        'video_info': {
            'total_frames': total_frames,
            'fps': fps,
            'duration_seconds': duration
        },
        'frame_analysis': [],
        'alerts': [],
        'summary': {}
    }
    
    print("Starting analysis...")
    start_time = time.time()
    
    while True:
        success, frame = video.read()
        if not success:
            break
        
        frame_count += 1
        current_video_time = frame_count / fps if fps > 0 else 0
        
        # Skip frames to speed up analysis
        if frame_count % check_every_n_frames != 0:
            continue
        
        # Start timing for this frame
        frame_start_time = time.time()
        
        # Find people in frame
        people_detection_start = time.time()
        people_found = find_people_in_frame(person_model, frame)
        people_detection_time = time.time() - people_detection_start
        
        frame_data = {
            'frame_number': frame_count,
            'video_time_seconds': current_video_time,
            'people_count': len(people_found),
            'people_detection_time': people_detection_time,
            'ai_analysis_time': 0,
            'total_frame_time': 0,
            'has_alert': False,
            'ai_message': '',
            'alert_confidence': 0
        }
        
        # Only check with AI if we have multiple people
        if len(people_found) >= 2:
            ai_checks += 1
            
            print(f"Checking frame {frame_count} (time: {current_video_time:.1f}s) - {len(people_found)} people found")
            
            # Time the AI analysis
            ai_start_time = time.time()
            is_harassment, ai_message = analyze_for_harassment(None, frame)  # multi_ai handles model selection
            ai_analysis_time = time.time() - ai_start_time
            
            frame_data['ai_analysis_time'] = ai_analysis_time
            frame_data['ai_message'] = ai_message
            
            if is_harassment:
                alerts_found += 1
                frame_data['has_alert'] = True
                
                print(f"  *** ALERT {alerts_found} *** at {current_video_time:.1f}s: {ai_message}")
                print(f"  AI analysis time: {ai_analysis_time:.2f} seconds")
                
                # Save the frame with alert
                frame_with_boxes = draw_person_boxes(frame.copy(), people_found)
                cv2.putText(frame_with_boxes, f"ALERT at {current_video_time:.1f}s", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                
                alert_filename = f"alert_{alerts_found}_at_{current_video_time:.1f}s.jpg"
                cv2.imwrite(alert_filename, frame_with_boxes)
                print(f"  Saved evidence: {alert_filename}")
                
                # Store alert details
                alert_data = {
                    'alert_number': alerts_found,
                    'frame_number': frame_count,
                    'video_time_seconds': current_video_time,
                    'people_count': len(people_found),
                    'ai_message': ai_message,
                    'evidence_file': alert_filename,
                    'analysis_time': ai_analysis_time
                }
                analysis_session['alerts'].append(alert_data)
                
                progress = (frame_count / total_frames * 100) if total_frames > 0 else 0.0
                # AUTO-STOP FEATURE: Ask user whether to continue after alert
                print()
                print("🚨 HARASSMENT ALERT DETECTED! 🚨")
                print("💰 Stopping analysis to save AI costs...")
                print(f"📊 Current progress: {progress:.1f}% ({frame_count}/{total_frames} frames)")
                print(f"⏱️  Analysis time so far: {time.time() - start_time:.1f} seconds")
                print(f"🔍 AI checks performed: {ai_checks}")
                print()
                
                try:
                    user_choice = input("Continue analysis? (y/n/s): [n] ").strip().lower()
                    print()
                    
                    if user_choice in ['y', 'yes']:
                        print("✅ Continuing analysis...")
                        continue
                    elif user_choice in ['s', 'summary']:
                        print("📋 Generating summary and stopping...")
                        break
                    else:  # 'n', 'no', or Enter (default)
                        print("🛑 Analysis stopped by user to save costs.")
                        print("📋 Generating summary of current progress...")
                        break
                        
                except KeyboardInterrupt:
                    print("\n🛑 Analysis stopped by user (Ctrl+C)")
                    break
                
            else:
                print(f"  Normal behavior: {ai_message}")
                print(f"  AI analysis time: {ai_analysis_time:.2f} seconds")
        
        # Calculate total frame processing time
        frame_data['total_frame_time'] = time.time() - frame_start_time
        analysis_session['frame_analysis'].append(frame_data)
        
        # Show progress with timing info
        progress = (frame_count / total_frames) * 100
        if frame_count % 10 == 0:  # Update every 10 frames
            avg_frame_time = sum(f['total_frame_time'] for f in analysis_session['frame_analysis'][-10:]) / min(10, len(analysis_session['frame_analysis']))
            print(f"Progress: {progress:.1f}% ({frame_count}/{total_frames} frames) - Avg frame time: {avg_frame_time:.2f}s")
    
    # Analysis complete
    analysis_time = time.time() - start_time
    analysis_session['end_time'] = datetime.now().isoformat()
    
    # Calculate summary statistics
    frames_with_people = len([f for f in analysis_session['frame_analysis'] if f['people_count'] > 0])
    frames_with_multiple_people = len([f for f in analysis_session['frame_analysis'] if f['people_count'] >= 2])
    ai_frames = len([f for f in analysis_session['frame_analysis'] if f['ai_analysis_time'] > 0])
    
    avg_people_detection_time = sum(f['people_detection_time'] for f in analysis_session['frame_analysis']) / len(analysis_session['frame_analysis']) if analysis_session['frame_analysis'] else 0
    avg_ai_time = sum(f['ai_analysis_time'] for f in analysis_session['frame_analysis'] if f['ai_analysis_time'] > 0) / ai_frames if ai_frames > 0 else 0
    avg_total_frame_time = sum(f['total_frame_time'] for f in analysis_session['frame_analysis']) / len(analysis_session['frame_analysis']) if analysis_session['frame_analysis'] else 0
    
    analysis_session['summary'] = {
        'total_analysis_time_seconds': analysis_time,
        'frames_processed': frame_count,
        'frames_with_people': frames_with_people,
        'frames_with_multiple_people': frames_with_multiple_people,
        'ai_checks_performed': ai_checks,
        'alerts_found': alerts_found,
        'average_people_detection_time': avg_people_detection_time,
        'average_ai_analysis_time': avg_ai_time,
        'average_total_frame_time': avg_total_frame_time,
        'frames_per_second_processed': frame_count / analysis_time if analysis_time > 0 else 0
    }
    
    print()
    print("=== Analysis Complete ===")
    analysis_status = "STOPPED EARLY" if frame_count < total_frames else "COMPLETE"
    print(f"Analysis status: {analysis_status}")
    print(f"Analysis time: {analysis_time:.1f} seconds")
    print(f"Frames processed: {frame_count} of {total_frames} ({progress:.1f}%)")
    print(f"Frames with people: {frames_with_people}")
    print(f"Frames with 2+ people: {frames_with_multiple_people}")
    print(f"AI checks performed: {ai_checks}")
    print(f"Alerts found: {alerts_found}")
    
    if alerts_found > 0:
        print(f"💰 Cost savings: Analysis stopped after first alert detection")
        remaining_frames = total_frames - frame_count
        estimated_remaining_time = remaining_frames * avg_total_frame_time if avg_total_frame_time > 0 else 0
        print(f"💰 Estimated time saved: {estimated_remaining_time:.1f} seconds")
    
    print()
    print("=== Timing Statistics ===")
    print(f"Average people detection time: {avg_people_detection_time:.3f} seconds")
    print(f"Average AI analysis time: {avg_ai_time:.3f} seconds")
    print(f"Average total frame time: {avg_total_frame_time:.3f} seconds")
    print(f"Processing speed: {analysis_session['summary']['frames_per_second_processed']:.2f} frames/second")
    
    # Generate log files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    
    # Save detailed JSON log
    log_filename = f"analysis_log_{video_name}_{timestamp}.json"
    with open(log_filename, 'w') as f:
        json.dump(analysis_session, f, indent=2)
    print(f"\nDetailed log saved: {log_filename}")
    
    # Save human-readable summary
    summary_filename = f"analysis_summary_{video_name}_{timestamp}.txt"
    with open(summary_filename, 'w') as f:
        f.write(f"Safety Detector - Video Analysis Summary\n")
        f.write(f"=" * 50 + "\n\n")
        f.write(f"Video: {video_path}\n")
        f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        f.write(f"Video Information:\n")
        f.write(f"  Total Frames: {total_frames}\n")
        f.write(f"  FPS: {fps}\n")
        f.write(f"  Duration: {duration:.1f} seconds\n\n")
        
        f.write(f"Analysis Results:\n")
        f.write(f"  Frames Processed: {frame_count}\n")
        f.write(f"  Frames with People: {frames_with_people}\n")
        f.write(f"  Frames with 2+ People: {frames_with_multiple_people}\n")
        f.write(f"  AI Checks Performed: {ai_checks}\n")
        f.write(f"  Alerts Found: {alerts_found}\n\n")
        
        f.write(f"Timing Performance:\n")
        f.write(f"  Total Analysis Time: {analysis_time:.1f} seconds\n")
        f.write(f"  Average People Detection: {avg_people_detection_time:.3f} seconds\n")
        f.write(f"  Average AI Analysis: {avg_ai_time:.3f} seconds\n")
        f.write(f"  Average Frame Processing: {avg_total_frame_time:.3f} seconds\n")
        f.write(f"  Processing Speed: {analysis_session['summary']['frames_per_second_processed']:.2f} frames/second\n\n")
        
        if alerts_found > 0:
            f.write(f"ALERTS DETECTED:\n")
            for alert in analysis_session['alerts']:
                f.write(f"  Alert {alert['alert_number']}: Frame {alert['frame_number']} at {alert['video_time_seconds']:.1f}s\n")
                f.write(f"    People: {alert['people_count']}\n")
                f.write(f"    Message: {alert['ai_message']}\n")
                f.write(f"    Evidence: {alert['evidence_file']}\n")
                f.write(f"    Analysis Time: {alert['analysis_time']:.2f}s\n\n")
        else:
            f.write(f"No harassment detected in this video.\n")
    
    print(f"Summary report saved: {summary_filename}")
    
    if alerts_found > 0:
        print(f"\n*** WARNING: {alerts_found} potential harassment incidents detected! ***")
        print("Check the saved alert images and summary report for evidence.")
    else:
        print("\nNo harassment detected in this video.")
    
    video.release()

def list_video_files():
    """Show available video files in test_videos folder"""
    test_videos_path = os.path.join(os.path.dirname(__file__), '..', 'test_videos')
    
    if not os.path.exists(test_videos_path):
        print("test_videos folder not found")
        return []
    
    video_files = []
    for file in os.listdir(test_videos_path):
        if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv', '.wmv')):
            video_files.append(os.path.join(test_videos_path, file))
    
    return video_files

def main():
    """Main function for video analysis"""
    print("=== Safety Detector - Video File Analyzer ===")
    print()
    
    # Check for video files
    available_videos = list_video_files()
    
    if not available_videos:
        print("No video files found in test_videos folder.")
        print("Please add some video files (.mp4, .avi, .mov, etc.) to the test_videos folder.")
        print()
        print("You can also specify a video file path directly:")
        video_path = input("Enter video file path (or press Enter to exit): ").strip()
        if video_path:
            analyze_video_file(video_path)
        return
    
    print("Available video files:")
    for i, video in enumerate(available_videos, 1):
        filename = os.path.basename(video)
        print(f"  {i}. {filename}")
    
    print()
    print("Choose an option:")
    print("  - Enter number to analyze a video")
    print("  - Enter full path to analyze any video file")
    print("  - Press Enter to exit")
    
    choice = input("Your choice: ").strip()
    
    if not choice:
        print("Goodbye!")
        return
    
    # Check if it's a number (selecting from list)
    try:
        video_index = int(choice) - 1
        if 0 <= video_index < len(available_videos):
            analyze_video_file(available_videos[video_index])
        else:
            print("Invalid number!")
    except ValueError:
        # It's a file path
        analyze_video_file(choice)

if __name__ == "__main__":
    main()
