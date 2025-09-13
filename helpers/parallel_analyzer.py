# Parallel Image Analysis for Safety Detector
# Processes multiple frames simultaneously for faster analysis

import cv2
import threading
import queue
import time
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

class ParallelFrameAnalyzer:
    def __init__(self, config_file=None):
        if config_file is None:
            # Try different config file locations
            possible_paths = [
                "../ai_config.json",
                "ai_config.json", 
                os.path.join(os.path.dirname(__file__), "..", "ai_config.json")
            ]
            config_file = None
            for path in possible_paths:
                if os.path.exists(path):
                    config_file = path
                    break
            
            if config_file is None:
                print("Warning: ai_config.json not found, using default settings")
        
        self.config = self._load_config(config_file)
        self.parallel_config = self.config.get('ai_model_settings', {}).get('harassment_detection_settings', {}).get('parallel_processing', {})
        
        # Parallel processing settings
        self.enabled = self.parallel_config.get('enabled', True)
        self.max_parallel_frames = self.parallel_config.get('max_parallel_frames', 4)
        self.batch_size = self.parallel_config.get('batch_size', 8)
        self.thread_pool_size = self.parallel_config.get('thread_pool_size', 4)
        self.queue_buffer_size = self.parallel_config.get('queue_buffer_size', 16)
        
        # Processing queues
        self.frame_queue = queue.Queue(maxsize=self.queue_buffer_size)
        self.result_queue = queue.Queue()
        
        # Thread pool for parallel processing
        self.executor = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        
        # Analysis models (will be set externally)
        self.person_model = None
        self.ai_manager = None
        
        print(f"🚀 Parallel Analyzer initialized:")
        print(f"   Max parallel frames: {self.max_parallel_frames}")
        print(f"   Batch size: {self.batch_size}")
        print(f"   Thread pool size: {self.thread_pool_size}")
        print(f"   Enabled: {self.enabled}")
    
    def _load_config(self, config_file):
        """Load configuration"""
        try:
            with open(config_file, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return {}
    
    def set_models(self, person_model, ai_manager):
        """Set the AI models for analysis"""
        self.person_model = person_model
        self.ai_manager = ai_manager
    
    def analyze_single_frame(self, frame_data):
        """Analyze a single frame (for parallel processing)"""
        frame_number, frame, timestamp = frame_data
        
        try:
            # Import analysis functions
            import sys
            sys.path.append(os.path.dirname(__file__))
            from person_detector import find_people_in_frame, draw_person_boxes
            from multi_ai import analyze_for_harassment
            
            analysis_start = time.time()
            
            # Person detection
            people_found = find_people_in_frame(self.person_model, frame)
            people_detection_time = time.time() - analysis_start
            
            result = {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'people_count': len(people_found),
                'people_detection_time': people_detection_time,
                'ai_analysis_time': 0,
                'ai_message': '',
                'is_harassment': False,
                'processed_frame': None,
                'evidence_frame': None
            }
            
            # AI analysis if 2+ people
            if len(people_found) >= 2:
                ai_start = time.time()
                is_harassment, ai_message = analyze_for_harassment(self.ai_manager, frame)
                ai_analysis_time = time.time() - ai_start
                
                result.update({
                    'ai_analysis_time': ai_analysis_time,
                    'ai_message': ai_message,
                    'is_harassment': is_harassment
                })
                
                # Create processed frame with detection boxes
                processed_frame = draw_person_boxes(frame.copy(), people_found)
                
                if is_harassment:
                    # Add alert overlay
                    cv2.putText(processed_frame, "HARASSMENT ALERT!", (10, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
                    cv2.putText(processed_frame, ai_message[:50], (10, 70), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    result['evidence_frame'] = processed_frame.copy()
                
                result['processed_frame'] = processed_frame
            else:
                # Just add detection boxes for people
                result['processed_frame'] = draw_person_boxes(frame.copy(), people_found)
            
            return result
            
        except Exception as e:
            return {
                'frame_number': frame_number,
                'timestamp': timestamp,
                'error': str(e),
                'people_count': 0,
                'is_harassment': False
            }
    
    def analyze_frames_parallel(self, frames_batch):
        """Analyze multiple frames in parallel"""
        if not self.enabled:
            # Fall back to sequential processing
            return [self.analyze_single_frame(frame_data) for frame_data in frames_batch]
        
        # Submit all frames for parallel processing
        futures = []
        for frame_data in frames_batch:
            future = self.executor.submit(self.analyze_single_frame, frame_data)
            futures.append(future)
        
        # Collect results as they complete
        results = []
        for future in as_completed(futures):
            try:
                result = future.result(timeout=30)  # 30 second timeout per frame
                results.append(result)
            except Exception as e:
                print(f"Parallel analysis error: {e}")
                results.append({
                    'frame_number': -1,
                    'error': str(e),
                    'people_count': 0,
                    'is_harassment': False
                })
        
        # Sort results by frame number
        results.sort(key=lambda x: x.get('frame_number', 0))
        return results
    
    def process_video_parallel(self, video_path, progress_callback=None, alert_callback=None):
        """Process entire video with parallel analysis"""
        print(f"🎬 Starting parallel video analysis: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        print(f"📊 Video info: {total_frames} frames, {fps} FPS")
        print(f"🚀 Processing with {self.max_parallel_frames} parallel threads")
        
        frame_number = 0
        frames_batch = []
        all_results = []
        total_ai_checks = 0
        total_alerts = 0
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_number += 1
            
            # Add frame to batch
            frames_batch.append((frame_number, frame.copy(), time.time()))
            
            # Process batch when it reaches batch_size
            if len(frames_batch) >= self.batch_size:
                print(f"🔄 Processing batch: frames {frame_number - self.batch_size + 1} to {frame_number}")
                
                # Analyze batch in parallel
                batch_results = self.analyze_frames_parallel(frames_batch)
                all_results.extend(batch_results)
                
                # Process results
                for result in batch_results:
                    if result.get('people_count', 0) >= 2:
                        total_ai_checks += 1
                    
                    if result.get('is_harassment', False):
                        total_alerts += 1
                        print(f"🚨 PARALLEL ALERT #{total_alerts} at frame {result['frame_number']}: {result.get('ai_message', 'Alert detected')}")
                        
                        # Call alert callback if provided
                        if alert_callback:
                            alert_callback(result)
                
                # Call progress callback if provided
                if progress_callback:
                    progress = int((frame_number / total_frames) * 100)
                    progress_callback(frame_number, total_frames, progress, total_ai_checks, total_alerts)
                
                # Clear batch
                frames_batch = []
        
        # Process remaining frames
        if frames_batch:
            print(f"🔄 Processing final batch: {len(frames_batch)} frames")
            batch_results = self.analyze_frames_parallel(frames_batch)
            all_results.extend(batch_results)
            
            # Process final results
            for result in batch_results:
                if result.get('people_count', 0) >= 2:
                    total_ai_checks += 1
                
                if result.get('is_harassment', False):
                    total_alerts += 1
                    if alert_callback:
                        alert_callback(result)
        
        cap.release()
        
        elapsed_time = time.time() - start_time
        avg_time_per_frame = elapsed_time / frame_number if frame_number > 0 else 0
        
        print(f"✅ Parallel analysis complete!")
        print(f"   Total time: {elapsed_time:.2f} seconds")
        print(f"   Frames processed: {frame_number}")
        print(f"   AI checks: {total_ai_checks}")
        print(f"   Alerts found: {total_alerts}")
        print(f"   Avg time per frame: {avg_time_per_frame:.3f} seconds")
        print(f"   Performance: {frame_number / elapsed_time:.1f} FPS")
        
        return {
            'total_frames': frame_number,
            'ai_checks': total_ai_checks,
            'alerts_found': total_alerts,
            'analysis_duration': elapsed_time,
            'avg_time_per_frame': avg_time_per_frame,
            'processing_fps': frame_number / elapsed_time,
            'results': all_results
        }
    
    def process_live_camera_parallel(self, camera_id=0, duration_seconds=None):
        """Process live camera with parallel analysis"""
        print(f"📹 Starting parallel live camera analysis")
        
        # Open camera
        camera = cv2.VideoCapture(camera_id)
        if not camera.isOpened():
            raise Exception(f"Could not open camera {camera_id}")
        
        print(f"🚀 Live processing with {self.max_parallel_frames} parallel threads")
        
        frame_number = 0
        frames_batch = []
        start_time = time.time()
        last_batch_time = start_time
        
        try:
            while True:
                # Check duration limit
                if duration_seconds and (time.time() - start_time) > duration_seconds:
                    break
                
                ret, frame = camera.read()
                if not ret:
                    break
                
                frame_number += 1
                
                # Add frame to batch
                frames_batch.append((frame_number, frame.copy(), time.time()))
                
                # Process batch when ready or every 2 seconds
                current_time = time.time()
                if len(frames_batch) >= self.batch_size or (current_time - last_batch_time) > 2.0:
                    if frames_batch:  # Only process if we have frames
                        print(f"🔄 Live batch: {len(frames_batch)} frames")
                        
                        # Analyze batch in parallel
                        batch_results = self.analyze_frames_parallel(frames_batch)
                        
                        # Process results immediately
                        for result in batch_results:
                            if result.get('is_harassment', False):
                                print(f"🚨 LIVE ALERT at frame {result['frame_number']}: {result.get('ai_message', 'Alert')}")
                        
                        frames_batch = []
                        last_batch_time = current_time
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.01)  # 10ms delay
        
        finally:
            camera.release()
            print(f"📹 Live camera analysis stopped after {frame_number} frames")
    
    def get_performance_stats(self):
        """Get performance statistics"""
        return {
            'parallel_enabled': self.enabled,
            'max_parallel_frames': self.max_parallel_frames,
            'batch_size': self.batch_size,
            'thread_pool_size': self.thread_pool_size,
            'queue_buffer_size': self.queue_buffer_size
        }
    
    def update_parallel_config(self, new_config):
        """Update parallel processing configuration"""
        self.parallel_config.update(new_config)
        
        # Update instance variables
        self.enabled = self.parallel_config.get('enabled', True)
        self.max_parallel_frames = self.parallel_config.get('max_parallel_frames', 4)
        self.batch_size = self.parallel_config.get('batch_size', 8)
        self.thread_pool_size = self.parallel_config.get('thread_pool_size', 4)
        
        # Recreate thread pool with new size
        self.executor.shutdown(wait=False)
        self.executor = ThreadPoolExecutor(max_workers=self.thread_pool_size)
        
        print(f"🔄 Parallel config updated: {self.max_parallel_frames} parallel frames, {self.thread_pool_size} threads")
    
    def cleanup(self):
        """Clean up resources"""
        self.executor.shutdown(wait=True)
        print("🧹 Parallel analyzer cleaned up")
