# Comprehensive Bug Hunt for Safety Detector Web Interface
# Tests every button, feature, and functionality

import requests
import time
import json
import os
from datetime import datetime

class SafetyDetectorBugHunt:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.bugs_found = []
        self.tests_passed = 0
        self.tests_failed = 0
        
    def log_bug(self, page, feature, error, severity="medium"):
        """Log a bug found during testing"""
        bug = {
            'page': page,
            'feature': feature,
            'error': error,
            'severity': severity,
            'timestamp': datetime.now().isoformat()
        }
        self.bugs_found.append(bug)
        print(f"🐛 BUG FOUND [{severity.upper()}]: {page} - {feature}")
        print(f"   Error: {error}")
        
    def log_success(self, page, feature):
        """Log a successful test"""
        self.tests_passed += 1
        print(f"✅ PASS: {page} - {feature}")
        
    def log_failure(self, page, feature, error):
        """Log a failed test"""
        self.tests_failed += 1
        self.log_bug(page, feature, error, "high")
        
    def test_server_running(self):
        """Test if web server is accessible"""
        print("\n🔍 Testing Web Server Accessibility")
        print("-" * 50)
        
        try:
            response = self.session.get(self.base_url, timeout=5)
            if response.status_code == 200:
                self.log_success("Server", "Web server accessibility")
                return True
            else:
                self.log_failure("Server", "Web server response", f"Status code: {response.status_code}")
                return False
        except Exception as e:
            self.log_failure("Server", "Web server connection", str(e))
            return False
    
    def test_dashboard_page(self):
        """Test dashboard page and all its features"""
        print("\n🏠 Testing Dashboard Page")
        print("-" * 50)
        
        try:
            # Test main dashboard load
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                self.log_success("Dashboard", "Page load")
                
                # Check for key elements in HTML
                html = response.text
                
                # Test navigation elements
                if 'navbar' in html:
                    self.log_success("Dashboard", "Navigation bar")
                else:
                    self.log_bug("Dashboard", "Navigation bar", "Navigation not found in HTML")
                
                # Test quick action buttons
                if 'Upload Video' in html:
                    self.log_success("Dashboard", "Upload Video button")
                else:
                    self.log_bug("Dashboard", "Upload Video button", "Button not found")
                
                if 'Start Monitoring' in html:
                    self.log_success("Dashboard", "Start Monitoring button")
                else:
                    self.log_bug("Dashboard", "Start Monitoring button", "Button not found")
                
                # Test status cards
                if 'AI Model' in html:
                    self.log_success("Dashboard", "AI Model status card")
                else:
                    self.log_bug("Dashboard", "AI Model status card", "Status card not found")
                
            else:
                self.log_failure("Dashboard", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("Dashboard", "Page load", str(e))
    
    def test_upload_page(self):
        """Test upload page functionality"""
        print("\n📤 Testing Upload Page")
        print("-" * 50)
        
        try:
            # Test upload page load
            response = self.session.get(f"{self.base_url}/upload")
            if response.status_code == 200:
                self.log_success("Upload", "Page load")
                
                html = response.text
                
                # Test form elements
                if 'enctype="multipart/form-data"' in html:
                    self.log_success("Upload", "File upload form")
                else:
                    self.log_bug("Upload", "File upload form", "Multipart form not found")
                
                if 'accept=".mp4,.avi,.mov,.mkv,.wmv"' in html:
                    self.log_success("Upload", "File type restrictions")
                else:
                    self.log_bug("Upload", "File type restrictions", "File type validation missing")
                
                # Test drag-drop area
                if 'dropZone' in html:
                    self.log_success("Upload", "Drag-drop zone")
                else:
                    self.log_bug("Upload", "Drag-drop zone", "Drop zone element not found")
                
                # Test recent uploads section
                if 'Recent Uploads' in html:
                    self.log_success("Upload", "Recent uploads display")
                else:
                    self.log_bug("Upload", "Recent uploads display", "Recent uploads section missing")
                
            else:
                self.log_failure("Upload", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("Upload", "Page load", str(e))
    
    def test_camera_page(self):
        """Test camera page functionality"""
        print("\n📹 Testing Camera Page")
        print("-" * 50)
        
        try:
            # Test camera page load
            response = self.session.get(f"{self.base_url}/camera")
            if response.status_code == 200:
                self.log_success("Camera", "Page load")
                
                html = response.text
                
                # Test camera controls
                if 'startBtn' in html:
                    self.log_success("Camera", "Start button")
                else:
                    self.log_bug("Camera", "Start button", "Start button ID not found")
                
                if 'stopBtn' in html:
                    self.log_success("Camera", "Stop button")
                else:
                    self.log_bug("Camera", "Stop button", "Stop button ID not found")
                
                if 'captureBtn' in html:
                    self.log_success("Camera", "Capture button")
                else:
                    self.log_bug("Camera", "Capture button", "Capture button ID not found")
                
                # Test ESP32 camera section
                if 'ESP32 Camera' in html:
                    self.log_success("Camera", "ESP32 camera section")
                else:
                    self.log_bug("Camera", "ESP32 camera section", "ESP32 section not found")
                
                if 'esp32IP' in html:
                    self.log_success("Camera", "ESP32 IP input")
                else:
                    self.log_bug("Camera", "ESP32 IP input", "ESP32 IP input field not found")
                
                # Test monitoring settings
                if 'analysisInterval' in html:
                    self.log_success("Camera", "Analysis interval setting")
                else:
                    self.log_bug("Camera", "Analysis interval setting", "Analysis interval control missing")
                
                if 'autoStop' in html:
                    self.log_success("Camera", "Auto-stop checkbox")
                else:
                    self.log_bug("Camera", "Auto-stop checkbox", "Auto-stop control missing")
                
            else:
                self.log_failure("Camera", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("Camera", "Page load", str(e))
    
    def test_history_page(self):
        """Test history page functionality"""
        print("\n📚 Testing History Page")
        print("-" * 50)
        
        try:
            # Test history page load
            response = self.session.get(f"{self.base_url}/history")
            if response.status_code == 200:
                self.log_success("History", "Page load")
                
                html = response.text
                
                # Test statistics cards
                if 'Total Uploads' in html:
                    self.log_success("History", "Statistics cards")
                else:
                    self.log_bug("History", "Statistics cards", "Statistics section not found")
                
                # Test action buttons
                if 'Upload New Video' in html:
                    self.log_success("History", "Upload New Video button")
                else:
                    self.log_bug("History", "Upload New Video button", "Upload button not found")
                
                if 'Export History' in html:
                    self.log_success("History", "Export History button")
                else:
                    self.log_bug("History", "Export History button", "Export button not found")
                
                if 'Clear All History' in html:
                    self.log_success("History", "Clear All History button")
                else:
                    self.log_bug("History", "Clear All History button", "Clear button not found")
                
                # Test table structure
                if '<table' in html and 'table-hover' in html:
                    self.log_success("History", "Upload table")
                else:
                    self.log_bug("History", "Upload table", "Upload table not found")
                
                # Test modals
                if 'deleteModal' in html:
                    self.log_success("History", "Delete confirmation modal")
                else:
                    self.log_bug("History", "Delete confirmation modal", "Delete modal not found")
                
            else:
                self.log_failure("History", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("History", "Page load", str(e))
    
    def test_config_page(self):
        """Test configuration page functionality"""
        print("\n⚙️ Testing Configuration Page")
        print("-" * 50)
        
        try:
            # Test config page load
            response = self.session.get(f"{self.base_url}/config")
            if response.status_code == 200:
                self.log_success("Config", "Page load")
                
                html = response.text
                
                # Test AI model selection
                if 'activeModel' in html:
                    self.log_success("Config", "AI model selection")
                else:
                    self.log_bug("Config", "AI model selection", "AI model dropdown not found")
                
                # Test confidence threshold
                if 'confidenceThreshold' in html:
                    self.log_success("Config", "Confidence threshold slider")
                else:
                    self.log_bug("Config", "Confidence threshold slider", "Confidence slider not found")
                
                # Test cost-saving features
                if 'autoStopOnAlert' in html:
                    self.log_success("Config", "Auto-stop checkbox")
                else:
                    self.log_bug("Config", "Auto-stop checkbox", "Auto-stop setting not found")
                
                if 'promptUserToContinue' in html:
                    self.log_success("Config", "Prompt user checkbox")
                else:
                    self.log_bug("Config", "Prompt user checkbox", "Prompt setting not found")
                
                # Test save button
                if 'saveConfigBtn' in html:
                    self.log_success("Config", "Save configuration button")
                else:
                    self.log_bug("Config", "Save configuration button", "Save button not found")
                
                # Test reset button
                if 'resetBtn' in html:
                    self.log_success("Config", "Reset button")
                else:
                    self.log_bug("Config", "Reset button", "Reset button not found")
                
                # Test export/import
                if 'exportConfigBtn' in html:
                    self.log_success("Config", "Export config button")
                else:
                    self.log_bug("Config", "Export config button", "Export button not found")
                
                if 'importConfigBtn' in html:
                    self.log_success("Config", "Import config button")
                else:
                    self.log_bug("Config", "Import config button", "Import button not found")
                
            else:
                self.log_failure("Config", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("Config", "Page load", str(e))
    
    def test_debug_page(self):
        """Test debug page functionality"""
        print("\n🐛 Testing Debug Page")
        print("-" * 50)
        
        try:
            # Test debug page load
            response = self.session.get(f"{self.base_url}/debug")
            if response.status_code == 200:
                self.log_success("Debug", "Page load")
                
                html = response.text
                
                # Test camera test
                if 'testCameraBtn' in html:
                    self.log_success("Debug", "Test camera button")
                else:
                    self.log_bug("Debug", "Test camera button", "Camera test button not found")
                
                # Test download test
                if 'testDownloadBtn' in html:
                    self.log_success("Debug", "Test download button")
                else:
                    self.log_bug("Debug", "Test download button", "Download test button not found")
                
                # Test console output
                if 'consoleOutput' in html:
                    self.log_success("Debug", "Console output area")
                else:
                    self.log_bug("Debug", "Console output area", "Console area not found")
                
                if 'clearConsoleBtn' in html:
                    self.log_success("Debug", "Clear console button")
                else:
                    self.log_bug("Debug", "Clear console button", "Clear button not found")
                
            else:
                self.log_failure("Debug", "Page load", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_failure("Debug", "Page load", str(e))
    
    def test_api_endpoints(self):
        """Test API endpoints functionality"""
        print("\n🔌 Testing API Endpoints")
        print("-" * 50)
        
        # Test analysis status endpoint
        try:
            response = self.session.get(f"{self.base_url}/analysis_status")
            if response.status_code == 200:
                self.log_success("API", "Analysis status endpoint")
                
                # Check JSON response
                try:
                    data = response.json()
                    if 'status' in data:
                        self.log_success("API", "Analysis status JSON format")
                    else:
                        self.log_bug("API", "Analysis status JSON format", "Missing status field")
                except:
                    self.log_bug("API", "Analysis status JSON parsing", "Invalid JSON response")
            else:
                self.log_bug("API", "Analysis status endpoint", f"Status code: {response.status_code}")
        except Exception as e:
            self.log_bug("API", "Analysis status endpoint", str(e))
        
        # Test ESP32 test endpoint (should handle missing IP)
        try:
            response = self.session.post(f"{self.base_url}/test_esp32", 
                                       json={}, 
                                       headers={'Content-Type': 'application/json'})
            if response.status_code == 400:  # Should return 400 for missing IP
                self.log_success("API", "ESP32 test validation")
            else:
                self.log_bug("API", "ESP32 test validation", f"Unexpected status: {response.status_code}")
        except Exception as e:
            self.log_bug("API", "ESP32 test endpoint", str(e))
    
    def test_javascript_errors(self):
        """Test for common JavaScript errors in templates"""
        print("\n📜 Testing JavaScript Functionality")
        print("-" * 50)
        
        pages_to_check = [
            ('/', 'Dashboard'),
            ('/upload', 'Upload'),
            ('/camera', 'Camera'),
            ('/history', 'History'),
            ('/config', 'Config'),
            ('/debug', 'Debug')
        ]
        
        for url, page_name in pages_to_check:
            try:
                response = self.session.get(f"{self.base_url}{url}")
                if response.status_code == 200:
                    html = response.text
                    
                    # Check for common JavaScript patterns
                    if 'addEventListener' in html:
                        self.log_success("JavaScript", f"{page_name} - Event listeners")
                    else:
                        self.log_bug("JavaScript", f"{page_name} - Event listeners", "No event listeners found")
                    
                    # Check for error handling
                    if 'try {' in html or 'catch' in html:
                        self.log_success("JavaScript", f"{page_name} - Error handling")
                    else:
                        self.log_bug("JavaScript", f"{page_name} - Error handling", "No error handling found")
                    
                    # Check for Bootstrap integration
                    if 'bootstrap' in html.lower():
                        self.log_success("JavaScript", f"{page_name} - Bootstrap integration")
                    else:
                        self.log_bug("JavaScript", f"{page_name} - Bootstrap integration", "Bootstrap not found")
                        
            except Exception as e:
                self.log_bug("JavaScript", f"{page_name} - Page check", str(e))
    
    def test_file_upload_simulation(self):
        """Test file upload functionality"""
        print("\n📁 Testing File Upload Functionality")
        print("-" * 50)
        
        try:
            # Create a small test file
            test_content = b"FAKE_VIDEO_DATA_FOR_TESTING" * 100
            
            # Test upload with valid file
            files = {'video': ('test_video.mp4', test_content, 'video/mp4')}
            response = self.session.post(f"{self.base_url}/upload_video", files=files)
            
            if response.status_code == 200 or response.status_code == 302:  # 302 for redirect
                self.log_success("Upload", "File upload functionality")
                
                # Check if redirected to analysis
                if response.status_code == 302 and 'analyze' in response.headers.get('Location', ''):
                    self.log_success("Upload", "Redirect to analysis")
                else:
                    self.log_bug("Upload", "Redirect to analysis", "Not redirected to analysis page")
                    
            else:
                self.log_bug("Upload", "File upload functionality", f"Status code: {response.status_code}")
                
        except Exception as e:
            self.log_bug("Upload", "File upload test", str(e))
        
        # Test upload with invalid file type
        try:
            files = {'video': ('test.txt', b'text file content', 'text/plain')}
            response = self.session.post(f"{self.base_url}/upload_video", files=files)
            
            # Should reject invalid file types
            if response.status_code != 200 or 'Invalid file type' in response.text:
                self.log_success("Upload", "File type validation")
            else:
                self.log_bug("Upload", "File type validation", "Invalid file type accepted")
                
        except Exception as e:
            self.log_bug("Upload", "File type validation test", str(e))
    
    def test_navigation_links(self):
        """Test all navigation links"""
        print("\n🧭 Testing Navigation Links")
        print("-" * 50)
        
        nav_links = [
            ('/', 'Dashboard'),
            ('/upload', 'Upload Video'),
            ('/history', 'History'),
            ('/camera', 'Live Camera'),
            ('/config', 'Configuration'),
            ('/debug', 'Debug')
        ]
        
        for url, name in nav_links:
            try:
                response = self.session.get(f"{self.base_url}{url}")
                if response.status_code == 200:
                    self.log_success("Navigation", f"{name} link")
                else:
                    self.log_bug("Navigation", f"{name} link", f"Status code: {response.status_code}")
            except Exception as e:
                self.log_bug("Navigation", f"{name} link", str(e))
    
    def test_responsive_design_elements(self):
        """Test responsive design elements in HTML"""
        print("\n📱 Testing Responsive Design Elements")
        print("-" * 50)
        
        try:
            response = self.session.get(f"{self.base_url}/")
            if response.status_code == 200:
                html = response.text
                
                # Check for Bootstrap responsive classes
                if 'col-md-' in html:
                    self.log_success("Responsive", "Bootstrap grid system")
                else:
                    self.log_bug("Responsive", "Bootstrap grid system", "No responsive grid found")
                
                # Check for viewport meta tag
                if 'viewport' in html:
                    self.log_success("Responsive", "Viewport meta tag")
                else:
                    self.log_bug("Responsive", "Viewport meta tag", "No viewport meta tag")
                
                # Check for responsive navigation
                if 'navbar-toggler' in html:
                    self.log_success("Responsive", "Mobile navigation toggle")
                else:
                    self.log_bug("Responsive", "Mobile navigation toggle", "No mobile nav toggle")
                    
        except Exception as e:
            self.log_bug("Responsive", "Design elements check", str(e))
    
    def generate_bug_report(self):
        """Generate comprehensive bug report"""
        print("\n" + "="*70)
        print("🔍 COMPREHENSIVE BUG HUNT REPORT")
        print("="*70)
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   ✅ Tests Passed: {self.tests_passed}")
        print(f"   ❌ Tests Failed: {self.tests_failed}")
        print(f"   🐛 Bugs Found: {len(self.bugs_found)}")
        
        if self.bugs_found:
            print(f"\n🐛 BUGS FOUND ({len(self.bugs_found)}):")
            print("-" * 50)
            
            # Group bugs by severity
            high_bugs = [b for b in self.bugs_found if b['severity'] == 'high']
            medium_bugs = [b for b in self.bugs_found if b['severity'] == 'medium']
            low_bugs = [b for b in self.bugs_found if b['severity'] == 'low']
            
            if high_bugs:
                print(f"\n🔴 HIGH SEVERITY ({len(high_bugs)}):")
                for bug in high_bugs:
                    print(f"   • {bug['page']} - {bug['feature']}")
                    print(f"     Error: {bug['error']}")
            
            if medium_bugs:
                print(f"\n🟡 MEDIUM SEVERITY ({len(medium_bugs)}):")
                for bug in medium_bugs:
                    print(f"   • {bug['page']} - {bug['feature']}")
                    print(f"     Error: {bug['error']}")
            
            if low_bugs:
                print(f"\n🟢 LOW SEVERITY ({len(low_bugs)}):")
                for bug in low_bugs:
                    print(f"   • {bug['page']} - {bug['feature']}")
                    print(f"     Error: {bug['error']}")
        else:
            print(f"\n🎉 NO BUGS FOUND! All tests passed successfully!")
        
        # Save detailed report to file
        report_filename = f"bug_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report_data = {
            'timestamp': datetime.now().isoformat(),
            'summary': {
                'tests_passed': self.tests_passed,
                'tests_failed': self.tests_failed,
                'bugs_found': len(self.bugs_found)
            },
            'bugs': self.bugs_found
        }
        
        with open(report_filename, 'w') as f:
            json.dump(report_data, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print("="*70)
    
    def run_comprehensive_bug_hunt(self):
        """Run complete bug hunt on all features"""
        print("🛡️ SAFETY DETECTOR COMPREHENSIVE BUG HUNT")
        print("="*70)
        print("Testing every button, feature, and functionality...")
        
        # Test server accessibility first
        if not self.test_server_running():
            print("\n❌ Cannot continue - web server not accessible!")
            return
        
        # Test all pages and features
        self.test_dashboard_page()
        self.test_upload_page()
        self.test_camera_page()
        self.test_history_page()
        self.test_config_page()
        self.test_debug_page()
        
        # Test functionality
        self.test_api_endpoints()
        self.test_javascript_errors()
        self.test_file_upload_simulation()
        self.test_navigation_links()
        self.test_responsive_design_elements()
        
        # Generate comprehensive report
        self.generate_bug_report()

if __name__ == "__main__":
    bug_hunter = SafetyDetectorBugHunt()
    bug_hunter.run_comprehensive_bug_hunt()
