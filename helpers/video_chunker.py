# Video Chunking Module for Safety Detector
# Splits videos into smaller chunks for parallel processing

import cv2
import os
import json
import time
import threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

class VideoChunker:
    def __init__(self, config_file="../ai_config.json"):
        self.config = self._load_config(config_file)
        self.chunk_config = self.config.get('ai_model_settings', {}).get('harassment_detection_settings', {}).get('video_chunking', {})
        
        # Chunking settings
        self.enabled = self.chunk_config.get('enabled', True)
        self.chunk_duration = self.chunk_config.get('chunk_duration_seconds', 15)
        self.max_parallel_chunks = self.chunk_config.get('max_parallel_chunks', 3)
        self.overlap_seconds = self.chunk_config.get('overlap_seconds', 2)
        self.auto_cleanup = self.chunk_config.get('auto_cleanup_chunks', True)
        self.preserve_on_alert = self.chunk_config.get('preserve_chunks_on_alert', True)
        
        # Processing state
        self.chunk_results = []
        self.total_alerts = 0
        
        print(f"🎬 Video Chunker initialized:")
        print(f"   Chunk duration: {self.chunk_duration} seconds")
        print(f"   Max parallel chunks: {self.max_parallel_chunks}")
        print(f"   Overlap: {self.overlap_seconds} seconds")
        print(f"   Enabled: {self.enabled}")
    
    def _load_config(self, config_file):
        """Load configuration"""
        try:
            # Handle different possible paths
            possible_paths = [
                config_file,
                "ai_config.json",
                "../ai_config.json",
                os.path.join(os.path.dirname(__file__), "..", "ai_config.json")
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r') as f:
                        return json.load(f)
            
            print("Warning: Config file not found, using defaults")
            return {}
        except Exception as e:
            print(f"Warning: Could not load config: {e}")
            return {}
    
    def split_video_into_chunks(self, video_path, output_dir):
        """Split video into smaller chunks"""
        print(f"✂️ Splitting video into {self.chunk_duration}-second chunks...")
        
        # Create output directory
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise Exception(f"Could not open video: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"📊 Video info: {duration:.1f}s, {total_frames} frames, {fps} FPS")
        
        # Calculate chunks
        frames_per_chunk = fps * self.chunk_duration
        overlap_frames = fps * self.overlap_seconds
        
        chunks_info = []
        chunk_number = 0
        
        for start_time in range(0, int(duration), self.chunk_duration - self.overlap_seconds):
            chunk_number += 1
            end_time = min(start_time + self.chunk_duration, duration)
            
            start_frame = start_time * fps
            end_frame = min(end_time * fps, total_frames)
            
            chunk_filename = f"chunk_{chunk_number:03d}_{int(start_time):04d}s-{int(end_time):04d}s.mp4"
            chunk_path = output_path / chunk_filename
            
            # Create video writer for this chunk
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            chunk_writer = cv2.VideoWriter(str(chunk_path), fourcc, fps, (width, height))
            
            # Extract frames for this chunk
            cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            frames_written = 0
            for frame_num in range(int(start_frame), int(end_frame)):
                ret, frame = cap.read()
                if not ret:
                    break
                
                chunk_writer.write(frame)
                frames_written += 1
            
            chunk_writer.release()
            
            chunk_info = {
                'chunk_number': chunk_number,
                'filename': chunk_filename,
                'path': str(chunk_path),
                'start_time': start_time,
                'end_time': end_time,
                'start_frame': int(start_frame),
                'end_frame': int(end_frame),
                'frames_count': frames_written,
                'duration': end_time - start_time
            }
            
            chunks_info.append(chunk_info)
            print(f"📦 Created chunk {chunk_number}: {start_time}s-{end_time}s ({frames_written} frames)")
            
            if end_time >= duration:
                break
        
        cap.release()
        
        # Save chunks metadata
        metadata = {
            'original_video': video_path,
            'total_duration': duration,
            'total_frames': total_frames,
            'fps': fps,
            'chunks_count': len(chunks_info),
            'chunk_duration': self.chunk_duration,
            'overlap_seconds': self.overlap_seconds,
            'chunks': chunks_info,
            'created_at': datetime.now().isoformat()
        }
        
        metadata_file = output_path / "chunks_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Video split into {len(chunks_info)} chunks")
        return chunks_info, str(metadata_file)
    
    def analyze_chunk(self, chunk_info, person_model, ai_manager):
        """Analyze a single video chunk"""
        chunk_path = chunk_info['path']
        chunk_number = chunk_info['chunk_number']
        
        print(f"🧠 Analyzing chunk {chunk_number}: {chunk_info['filename']}")
        
        try:
            # Import analysis functions
            import sys
            sys.path.append(os.path.dirname(__file__))
            from person_detector import find_people_in_frame, draw_person_boxes
            from multi_ai import analyze_for_harassment
            
            # Open chunk video
            cap = cv2.VideoCapture(chunk_path)
            if not cap.isOpened():
                return {'chunk_number': chunk_number, 'error': 'Could not open chunk'}
            
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            chunk_results = {
                'chunk_number': chunk_number,
                'filename': chunk_info['filename'],
                'start_time': chunk_info['start_time'],
                'end_time': chunk_info['end_time'],
                'frames_processed': 0,
                'ai_checks': 0,
                'alerts': [],
                'processing_time': 0,
                'analysis_start': time.time()
            }
            
            frame_number = 0
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_number += 1
                chunk_results['frames_processed'] = frame_number
                
                # Analyze every frame (since chunks are small)
                people_found = find_people_in_frame(person_model, frame)
                
                if len(people_found) >= 2:
                    chunk_results['ai_checks'] += 1
                    
                    # Real AI analysis
                    is_harassment, ai_message = analyze_for_harassment(ai_manager, frame)
                    
                    if is_harassment:
                        # Calculate time in original video
                        time_in_chunk = frame_number / fps
                        time_in_original = chunk_info['start_time'] + time_in_chunk
                        
                        alert = {
                            'chunk_number': chunk_number,
                            'frame_in_chunk': frame_number,
                            'frame_in_original': chunk_info['start_frame'] + frame_number,
                            'time_in_original': time_in_original,
                            'time_formatted': f"{int(time_in_original//60):02d}:{int(time_in_original%60):02d}",
                            'people_count': len(people_found),
                            'ai_message': ai_message,
                            'confidence': 0.92
                        }
                        
                        chunk_results['alerts'].append(alert)
                        print(f"🚨 CHUNK {chunk_number} ALERT: {ai_message[:50]}... at {alert['time_formatted']}")
            
            cap.release()
            
            chunk_results['processing_time'] = time.time() - chunk_results['analysis_start']
            
            print(f"✅ Chunk {chunk_number} complete: {frame_number} frames, {chunk_results['ai_checks']} AI checks, {len(chunk_results['alerts'])} alerts")
            
            return chunk_results
            
        except Exception as e:
            print(f"❌ Chunk {chunk_number} analysis failed: {e}")
            return {
                'chunk_number': chunk_number,
                'error': str(e),
                'alerts': [],
                'ai_checks': 0,
                'frames_processed': 0
            }
    
    def analyze_video_chunks_parallel(self, video_path, person_model, ai_manager, progress_callback=None):
        """Split video and analyze chunks in parallel"""
        print(f"🚀 Starting parallel chunk analysis for: {os.path.basename(video_path)}")
        
        # Create chunks directory
        video_name = Path(video_path).stem
        chunks_dir = Path(video_path).parent / f"{video_name}_chunks"
        
        try:
            # Split video into chunks
            chunks_info, metadata_file = self.split_video_into_chunks(video_path, chunks_dir)
            
            # Analyze chunks in parallel
            print(f"🔄 Analyzing {len(chunks_info)} chunks with {self.max_parallel_chunks} parallel workers...")
            
            all_results = []
            completed_chunks = 0
            
            # Use ThreadPoolExecutor for parallel chunk analysis
            with ThreadPoolExecutor(max_workers=self.max_parallel_chunks) as executor:
                # Submit all chunk analysis tasks
                future_to_chunk = {
                    executor.submit(self.analyze_chunk, chunk_info, person_model, ai_manager): chunk_info
                    for chunk_info in chunks_info
                }
                
                # Collect results as they complete
                for future in as_completed(future_to_chunk):
                    chunk_info = future_to_chunk[future]
                    
                    try:
                        result = future.result()
                        all_results.append(result)
                        completed_chunks += 1
                        
                        # Progress callback
                        if progress_callback:
                            progress = int((completed_chunks / len(chunks_info)) * 100)
                            progress_callback(completed_chunks, len(chunks_info), progress, result)
                        
                        print(f"📊 Chunk progress: {completed_chunks}/{len(chunks_info)} ({int((completed_chunks/len(chunks_info))*100)}%)")
                        
                    except Exception as e:
                        print(f"❌ Chunk analysis failed: {e}")
                        all_results.append({
                            'chunk_number': chunk_info['chunk_number'],
                            'error': str(e),
                            'alerts': []
                        })
            
            # Aggregate results
            aggregated_results = self._aggregate_chunk_results(all_results, chunks_info)
            
            # Cleanup chunks if configured
            if self.auto_cleanup and not (self.preserve_on_alert and aggregated_results['total_alerts'] > 0):
                self._cleanup_chunks(chunks_dir)
            
            return aggregated_results
            
        except Exception as e:
            print(f"❌ Parallel chunk analysis failed: {e}")
            raise e
    
    def _aggregate_chunk_results(self, chunk_results, chunks_info):
        """Aggregate results from all chunks"""
        print(f"📊 Aggregating results from {len(chunk_results)} chunks...")
        
        total_frames = sum(r.get('frames_processed', 0) for r in chunk_results)
        total_ai_checks = sum(r.get('ai_checks', 0) for r in chunk_results)
        total_alerts = sum(len(r.get('alerts', [])) for r in chunk_results)
        total_processing_time = sum(r.get('processing_time', 0) for r in chunk_results)
        
        # Collect all alerts and sort by time
        all_alerts = []
        for result in chunk_results:
            for alert in result.get('alerts', []):
                all_alerts.append(alert)
        
        # Sort alerts by time in original video
        all_alerts.sort(key=lambda x: x.get('time_in_original', 0))
        
        # Calculate performance metrics
        avg_processing_time = total_processing_time / len(chunk_results) if chunk_results else 0
        processing_fps = total_frames / total_processing_time if total_processing_time > 0 else 0
        
        aggregated = {
            'total_chunks': len(chunk_results),
            'total_frames': total_frames,
            'total_ai_checks': total_ai_checks,
            'total_alerts': total_alerts,
            'all_alerts': all_alerts,
            'total_processing_time': total_processing_time,
            'avg_chunk_time': avg_processing_time,
            'processing_fps': processing_fps,
            'chunk_results': chunk_results,
            'chunks_info': chunks_info,
            'parallel_efficiency': len(chunk_results) / (total_processing_time / self.chunk_duration) if total_processing_time > 0 else 1
        }
        
        print(f"✅ Aggregation complete:")
        print(f"   Total frames: {total_frames}")
        print(f"   Total AI checks: {total_ai_checks}")
        print(f"   Total alerts: {total_alerts}")
        print(f"   Processing time: {total_processing_time:.2f}s")
        print(f"   Processing FPS: {processing_fps:.1f}")
        print(f"   Parallel efficiency: {aggregated['parallel_efficiency']:.1f}x")
        
        return aggregated
    
    def _cleanup_chunks(self, chunks_dir):
        """Clean up temporary chunk files"""
        try:
            import shutil
            if chunks_dir.exists():
                shutil.rmtree(chunks_dir)
                print(f"🧹 Cleaned up chunks directory: {chunks_dir}")
        except Exception as e:
            print(f"⚠️ Could not cleanup chunks: {e}")
    
    def get_chunk_analysis_summary(self, aggregated_results):
        """Generate summary of chunk analysis"""
        return {
            'method': 'Parallel Chunk Analysis',
            'total_chunks': aggregated_results['total_chunks'],
            'chunk_duration': self.chunk_duration,
            'parallel_workers': self.max_parallel_chunks,
            'total_frames': aggregated_results['total_frames'],
            'ai_checks': aggregated_results['total_ai_checks'],
            'alerts_found': aggregated_results['total_alerts'],
            'processing_time': aggregated_results['total_processing_time'],
            'processing_fps': aggregated_results['processing_fps'],
            'parallel_efficiency': aggregated_results['parallel_efficiency'],
            'performance_gain': f"{aggregated_results['parallel_efficiency']:.1f}x faster than sequential"
        }
    
    def estimate_chunk_performance(self, video_duration):
        """Estimate performance improvement with chunking"""
        chunks_needed = max(1, int(video_duration / self.chunk_duration))
        sequential_time = video_duration * 2  # Estimate 2x real-time for sequential
        parallel_time = (chunks_needed / self.max_parallel_chunks) * self.chunk_duration * 2
        
        return {
            'chunks_needed': chunks_needed,
            'estimated_sequential_time': sequential_time,
            'estimated_parallel_time': parallel_time,
            'estimated_speedup': sequential_time / parallel_time if parallel_time > 0 else 1,
            'time_saved': sequential_time - parallel_time
        }
