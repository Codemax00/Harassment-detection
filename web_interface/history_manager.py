# Upload History Manager for Safety Detector
# Manages upload history, logs, and file cleanup

import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from file_manager import SafetyDetectorFileManager

class UploadHistoryManager:
    def __init__(self, history_file="upload_history.json", uploads_dir="uploads"):
        self.history_file = history_file
        self.uploads_dir = uploads_dir
        self.file_manager = SafetyDetectorFileManager()
        self.history_data = self._load_history()
    
    def _load_history(self):
        """Load history from JSON file"""
        try:
            if os.path.exists(self.history_file):
                with open(self.history_file, 'r') as f:
                    return json.load(f)
            return {"uploads": []}
        except Exception as e:
            print(f"Error loading history: {e}")
            return {"uploads": []}
    
    def _save_history(self):
        """Save history to JSON file"""
        try:
            with open(self.history_file, 'w') as f:
                json.dump(self.history_data, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving history: {e}")
            return False
    
    def add_upload(self, filename, file_size, file_data=None, analysis_status="pending"):
        """Add new upload to history"""
        upload_id = len(self.history_data["uploads"]) + 1
        
        # Create organized folder structure
        folder_info = self.file_manager.create_upload_folder(upload_id, filename)
        
        # Save video file if provided
        video_path = None
        if file_data:
            video_path = self.file_manager.save_video_file(upload_id, filename, file_data)
        
        upload_record = {
            "id": upload_id,
            "filename": filename,
            "original_filename": filename,
            "file_size": file_size,
            "upload_date": datetime.now().isoformat(),
            "analysis_status": analysis_status,
            "analysis_results": {
                "total_frames": 0,
                "frames_processed": 0,
                "ai_checks": 0,
                "alerts_found": 0,
                "analysis_duration": "00:00:00",
                "completion_percentage": 0
            },
            "logs": [],
            "alerts": [],
            "evidence_files": [],
            "folder_info": folder_info,
            "video_path": video_path
        }
        
        self.history_data["uploads"].append(upload_record)
        self._save_history()
        
        # Log the upload
        self.add_log_entry(upload_id, "info", f"Upload created with organized folder structure")
        
        return upload_id
    
    def update_upload_status(self, upload_id, status, analysis_results=None):
        """Update upload analysis status and results"""
        for upload in self.history_data["uploads"]:
            if upload["id"] == upload_id:
                upload["analysis_status"] = status
                if analysis_results:
                    upload["analysis_results"].update(analysis_results)
                self._save_history()
                return True
        return False
    
    def add_log_entry(self, upload_id, log_type, message):
        """Add log entry to upload record"""
        log_entry = {
            "timestamp": datetime.now().isoformat(),
            "type": log_type,  # info, warning, error, alert
            "message": message
        }
        
        # Save to file manager
        self.file_manager.save_analysis_log(upload_id, log_entry)
        
        # Also save to history record
        for upload in self.history_data["uploads"]:
            if upload["id"] == upload_id:
                upload["logs"].append(log_entry)
                self._save_history()
                return True
        return False
    
    def add_alert(self, upload_id, alert_data, frame_image=None):
        """Add alert to upload record"""
        alert_id = len([u for u in self.history_data["uploads"] if u["id"] == upload_id][0].get("alerts", [])) + 1
        
        # Save evidence if frame image provided
        evidence_info = None
        if frame_image is not None:
            evidence_info = self.file_manager.save_alert_evidence(upload_id, alert_id, frame_image, alert_data)
        
        alert = {
            "alert_id": alert_id,
            "timestamp": datetime.now().isoformat(),
            "frame": alert_data.get("frame", 0),
            "time_in_video": alert_data.get("time", "00:00:00"),
            "confidence": alert_data.get("confidence", 0.0),
            "message": alert_data.get("message", "Alert detected"),
            "evidence_file": evidence_info["filename"] if evidence_info else "",
            "evidence_path": evidence_info["evidence_path"] if evidence_info else "",
            "ai_analysis": alert_data.get("ai_analysis", "")
        }
        
        for upload in self.history_data["uploads"]:
            if upload["id"] == upload_id:
                upload["alerts"].append(alert)
                if alert["evidence_file"]:
                    upload["evidence_files"].append(alert["evidence_file"])
                
                # Update alert count in analysis results
                upload["analysis_results"]["alerts_found"] = len(upload["alerts"])
                
                self._save_history()
                
                # Log the alert
                self.add_log_entry(upload_id, "alert", f"Alert #{alert_id} detected: {alert['message']}")
                
                return alert_id
        return False
    
    def get_all_uploads(self):
        """Get all upload records"""
        return self.history_data["uploads"]
    
    def get_upload_by_id(self, upload_id):
        """Get specific upload record by ID"""
        for upload in self.history_data["uploads"]:
            if upload["id"] == upload_id:
                return upload
        return None
    
    def delete_upload(self, upload_id):
        """Delete upload record and associated files"""
        upload = self.get_upload_by_id(upload_id)
        if not upload:
            return False, "Upload not found"
        
        try:
            # Delete all files using file manager
            success = self.file_manager.delete_upload_files(upload_id)
            if not success:
                return False, "Error deleting upload files"
            
            # Remove from history
            self.history_data["uploads"] = [
                u for u in self.history_data["uploads"] 
                if u["id"] != upload_id
            ]
            
            self._save_history()
            return True, "Upload deleted successfully"
            
        except Exception as e:
            return False, f"Error deleting upload: {str(e)}"
    
    def clear_all_history(self):
        """Clear all history and delete all files"""
        try:
            # Delete all uploaded files
            if os.path.exists(self.uploads_dir):
                for filename in os.listdir(self.uploads_dir):
                    file_path = os.path.join(self.uploads_dir, filename)
                    if os.path.isfile(file_path):
                        os.remove(file_path)
            
            # Clear history
            self.history_data = {"uploads": []}
            self._save_history()
            return True, "All history cleared successfully"
            
        except Exception as e:
            return False, f"Error clearing history: {str(e)}"
    
    def get_statistics(self):
        """Get overall statistics"""
        uploads = self.history_data["uploads"]
        total_uploads = len(uploads)
        completed_analyses = len([u for u in uploads if u["analysis_status"] == "completed"])
        total_alerts = sum(len(u.get("alerts", [])) for u in uploads)
        total_size = sum(u.get("file_size", 0) for u in uploads)
        
        return {
            "total_uploads": total_uploads,
            "completed_analyses": completed_analyses,
            "pending_analyses": total_uploads - completed_analyses,
            "total_alerts": total_alerts,
            "total_size_mb": round(total_size / (1024 * 1024), 2),
            "uploads_with_alerts": len([u for u in uploads if len(u.get("alerts", [])) > 0])
        }
    
    def cleanup_old_files(self, days_old=30):
        """Clean up files older than specified days"""
        cutoff_date = datetime.now().timestamp() - (days_old * 24 * 60 * 60)
        cleaned_files = []
        
        for upload in self.history_data["uploads"][:]:  # Use slice to avoid modification during iteration
            upload_date = datetime.fromisoformat(upload["upload_date"]).timestamp()
            
            if upload_date < cutoff_date:
                success, message = self.delete_upload(upload["id"])
                if success:
                    cleaned_files.append(upload["filename"])
        
        return cleaned_files
    
    def save_frame(self, upload_id, frame_number, frame_image, frame_type="regular"):
        """Save frame image"""
        return self.file_manager.save_frame(upload_id, frame_number, frame_image, frame_type)
    
    def save_analysis_report(self, upload_id, report_data):
        """Save comprehensive analysis report"""
        return self.file_manager.save_analysis_report(upload_id, report_data)
    
    def get_upload_files(self, upload_id, file_type="all"):
        """Get files for an upload"""
        return self.file_manager.get_upload_files(upload_id, file_type)
    
    def get_evidence_files(self, upload_id):
        """Get evidence files for an upload"""
        files = self.file_manager.get_upload_files(upload_id, "evidence")
        return files.get("evidence", [])
    
    def get_report_files(self, upload_id):
        """Get report files for an upload"""
        files = self.file_manager.get_upload_files(upload_id, "reports")
        return files.get("reports", [])
