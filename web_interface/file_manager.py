# File Management System for Safety Detector
# Organizes uploads, frames, evidence, and reports in structured folders

import os
import json
import shutil
from datetime import datetime
from pathlib import Path
import cv2

class SafetyDetectorFileManager:
    def __init__(self, base_dir="storage"):
        self.base_dir = Path(base_dir)
        self.uploads_dir = self.base_dir / "uploads"
        self.analysis_dir = self.base_dir / "analysis"
        self.reports_dir = self.base_dir / "reports"
        self.evidence_dir = self.base_dir / "evidence"
        
        # Create directory structure
        self._create_directories()
    
    def _create_directories(self):
        """Create all necessary directories"""
        directories = [
            self.base_dir,
            self.uploads_dir,
            self.analysis_dir,
            self.reports_dir,
            self.evidence_dir
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
    
    def create_upload_folder(self, upload_id, filename):
        """Create organized folder structure for an upload"""
        upload_folder = self.analysis_dir / f"upload_{upload_id:04d}_{filename.split('.')[0]}"
        
        # Create subfolders
        subfolders = {
            'frames': upload_folder / "frames",
            'alerts': upload_folder / "alerts", 
            'evidence': upload_folder / "evidence",
            'logs': upload_folder / "logs",
            'reports': upload_folder / "reports"
        }
        
        for subfolder in subfolders.values():
            subfolder.mkdir(parents=True, exist_ok=True)
        
        # Create metadata file
        metadata = {
            'upload_id': upload_id,
            'filename': filename,
            'created': datetime.now().isoformat(),
            'upload_folder': str(upload_folder),
            'subfolders': {k: str(v) for k, v in subfolders.items()}
        }
        
        metadata_file = upload_folder / "metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        return {
            'upload_folder': str(upload_folder),
            'subfolders': {k: str(v) for k, v in subfolders.items()},
            'metadata_file': str(metadata_file)
        }
    
    def save_video_file(self, upload_id, filename, file_data):
        """Save uploaded video file"""
        video_path = self.uploads_dir / f"{upload_id:04d}_{filename}"
        
        with open(video_path, 'wb') as f:
            f.write(file_data)
        
        return str(video_path)
    
    def save_frame(self, upload_id, frame_number, frame_data, frame_type="regular"):
        """Save individual frame"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return None
        
        frames_dir = Path(upload_info['subfolders']['frames'])
        
        # Create frame filename
        frame_filename = f"frame_{frame_number:06d}_{frame_type}.jpg"
        frame_path = frames_dir / frame_filename
        
        # Save frame
        cv2.imwrite(str(frame_path), frame_data)
        
        return str(frame_path)
    
    def save_alert_evidence(self, upload_id, alert_id, frame_data, alert_info):
        """Save alert evidence with metadata"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return None
        
        evidence_dir = Path(upload_info['subfolders']['evidence'])
        
        # Save evidence image
        evidence_filename = f"alert_{alert_id:03d}_frame_{alert_info.get('frame', 0):06d}.jpg"
        evidence_path = evidence_dir / evidence_filename
        cv2.imwrite(str(evidence_path), frame_data)
        
        # Save alert metadata
        alert_metadata = {
            'alert_id': alert_id,
            'frame_number': alert_info.get('frame', 0),
            'timestamp': alert_info.get('timestamp', datetime.now().isoformat()),
            'time_in_video': alert_info.get('time_in_video', '00:00:00'),
            'confidence': alert_info.get('confidence', 0.0),
            'message': alert_info.get('message', 'Alert detected'),
            'evidence_file': evidence_filename,
            'ai_analysis': alert_info.get('ai_analysis', '')
        }
        
        metadata_file = evidence_dir / f"alert_{alert_id:03d}_metadata.json"
        with open(metadata_file, 'w') as f:
            json.dump(alert_metadata, f, indent=2)
        
        return {
            'evidence_path': str(evidence_path),
            'metadata_path': str(metadata_file),
            'filename': evidence_filename
        }
    
    def save_analysis_log(self, upload_id, log_entry):
        """Save analysis log entry"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return False
        
        logs_dir = Path(upload_info['subfolders']['logs'])
        log_file = logs_dir / "analysis.log"
        
        # Format log entry
        timestamp = datetime.now().isoformat()
        log_line = f"[{timestamp}] {log_entry['type'].upper()}: {log_entry['message']}\n"
        
        # Append to log file
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(log_line)
        
        # Also save as JSON for structured access
        json_log_file = logs_dir / "analysis.json"
        
        # Load existing logs
        logs = []
        if json_log_file.exists():
            with open(json_log_file, 'r') as f:
                logs = json.load(f)
        
        # Add new entry
        log_entry['timestamp'] = timestamp
        logs.append(log_entry)
        
        # Save updated logs
        with open(json_log_file, 'w') as f:
            json.dump(logs, f, indent=2)
        
        return True
    
    def save_analysis_report(self, upload_id, report_data):
        """Save comprehensive analysis report"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return None
        
        reports_dir = Path(upload_info['subfolders']['reports'])
        
        # Save JSON report
        json_report_file = reports_dir / "analysis_report.json"
        with open(json_report_file, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        # Save text report
        text_report_file = reports_dir / "analysis_report.txt"
        text_report = self._generate_text_report(report_data)
        with open(text_report_file, 'w', encoding='utf-8') as f:
            f.write(text_report)
        
        return {
            'json_report': str(json_report_file),
            'text_report': str(text_report_file)
        }
    
    def _generate_text_report(self, report_data):
        """Generate human-readable text report"""
        report = f"""
SAFETY DETECTOR ANALYSIS REPORT
===============================

Upload Information:
- Upload ID: #{report_data.get('upload_id', 'N/A')}
- Filename: {report_data.get('filename', 'N/A')}
- Analysis Date: {report_data.get('analysis_date', 'N/A')}
- File Size: {report_data.get('file_size_mb', 0):.2f} MB

Analysis Results:
- Total Frames: {report_data.get('total_frames', 0)}
- Frames Processed: {report_data.get('frames_processed', 0)}
- AI Checks Performed: {report_data.get('ai_checks', 0)}
- Analysis Duration: {report_data.get('analysis_duration', '00:00:00')}
- Completion: {report_data.get('completion_percentage', 0)}%

Alert Summary:
- Total Alerts: {len(report_data.get('alerts', []))}
"""
        
        # Add alert details
        alerts = report_data.get('alerts', [])
        if alerts:
            report += "\nAlert Details:\n"
            for i, alert in enumerate(alerts, 1):
                report += f"""
Alert #{i}:
- Frame: {alert.get('frame', 0)}
- Time: {alert.get('time_in_video', '00:00:00')}
- Confidence: {alert.get('confidence', 0)*100:.1f}%
- Message: {alert.get('message', 'Alert detected')}
- Evidence: {alert.get('evidence_file', 'N/A')}
"""
        else:
            report += "\nNo alerts detected in this video.\n"
        
        # Add cost information
        if 'cost_info' in report_data:
            cost_info = report_data['cost_info']
            report += f"""
Cost Information:
- API Calls Made: {cost_info.get('api_calls', 0)}
- Estimated Cost: ${cost_info.get('estimated_cost', 0):.3f}
- Cost Savings: {cost_info.get('savings_percentage', 0)}%
"""
        
        report += f"""
Analysis Status: {report_data.get('status', 'Unknown')}
Generated by Safety Detector v1.0
Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        return report.strip()
    
    def get_upload_info(self, upload_id):
        """Get upload folder information"""
        # Find upload folder
        for folder in self.analysis_dir.iterdir():
            if folder.is_dir() and folder.name.startswith(f"upload_{upload_id:04d}_"):
                metadata_file = folder / "metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        return json.load(f)
        return None
    
    def get_upload_files(self, upload_id, file_type="all"):
        """Get files for an upload"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return []
        
        files = {}
        
        if file_type in ["all", "frames"]:
            frames_dir = Path(upload_info['subfolders']['frames'])
            if frames_dir.exists():
                files['frames'] = [str(f) for f in frames_dir.glob("*.jpg")]
        
        if file_type in ["all", "evidence"]:
            evidence_dir = Path(upload_info['subfolders']['evidence'])
            if evidence_dir.exists():
                files['evidence'] = [str(f) for f in evidence_dir.glob("*.jpg")]
        
        if file_type in ["all", "reports"]:
            reports_dir = Path(upload_info['subfolders']['reports'])
            if reports_dir.exists():
                files['reports'] = [str(f) for f in reports_dir.glob("*")]
        
        if file_type in ["all", "logs"]:
            logs_dir = Path(upload_info['subfolders']['logs'])
            if logs_dir.exists():
                files['logs'] = [str(f) for f in logs_dir.glob("*")]
        
        return files
    
    def delete_upload_files(self, upload_id):
        """Delete all files for an upload"""
        upload_info = self.get_upload_info(upload_id)
        if not upload_info:
            return False
        
        try:
            # Delete upload folder and all contents
            upload_folder = Path(upload_info['upload_folder'])
            if upload_folder.exists():
                shutil.rmtree(upload_folder)
            
            # Delete original video file
            video_files = list(self.uploads_dir.glob(f"{upload_id:04d}_*"))
            for video_file in video_files:
                video_file.unlink()
            
            return True
        except Exception as e:
            print(f"Error deleting upload files: {e}")
            return False
    
    def get_storage_stats(self):
        """Get storage usage statistics"""
        def get_dir_size(directory):
            total = 0
            try:
                for dirpath, dirnames, filenames in os.walk(directory):
                    for filename in filenames:
                        filepath = os.path.join(dirpath, filename)
                        if os.path.exists(filepath):
                            total += os.path.getsize(filepath)
            except:
                pass
            return total
        
        stats = {
            'total_size_bytes': get_dir_size(self.base_dir),
            'uploads_size_bytes': get_dir_size(self.uploads_dir),
            'analysis_size_bytes': get_dir_size(self.analysis_dir),
            'reports_size_bytes': get_dir_size(self.reports_dir),
            'evidence_size_bytes': get_dir_size(self.evidence_dir),
        }
        
        # Convert to MB
        for key in stats:
            stats[key.replace('_bytes', '_mb')] = stats[key] / (1024 * 1024)
        
        return stats
    
    def cleanup_old_files(self, days_old=30):
        """Clean up files older than specified days"""
        import time
        cutoff_time = time.time() - (days_old * 24 * 60 * 60)
        cleaned_files = []
        
        try:
            for folder in self.analysis_dir.iterdir():
                if folder.is_dir():
                    # Check folder modification time
                    if folder.stat().st_mtime < cutoff_time:
                        cleaned_files.append(folder.name)
                        shutil.rmtree(folder)
            
            # Clean old video files
            for video_file in self.uploads_dir.iterdir():
                if video_file.stat().st_mtime < cutoff_time:
                    cleaned_files.append(video_file.name)
                    video_file.unlink()
                    
        except Exception as e:
            print(f"Error during cleanup: {e}")
        
        return cleaned_files
