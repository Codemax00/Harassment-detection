# Automated Frontend Tests for Safety Detector
# Tests that can run while the web server is active

import requests
import json
import time
from datetime import datetime

class AutomatedFrontendTests:
    def __init__(self, base_url="http://localhost:5000"):
        self.base_url = base_url
        self.session = requests.Session()
        self.results = {
            'passed': 0,
            'failed': 0,
            'bugs': [],
            'timestamp': datetime.now().isoformat()
        }
    
    def log_result(self, test_name, passed, error=None):
        """Log test result"""
        if passed:
            self.results['passed'] += 1
            print(f"✅ {test_name}")
        else:
            self.results['failed'] += 1
            self.results['bugs'].append({
                'test': test_name,
                'error': error,
                'timestamp': datetime.now().isoformat()
            })
            print(f"❌ {test_name}: {error}")
    
    def test_server_connectivity(self):
        """Test basic server connectivity"""
        print("\n🔌 Testing Server Connectivity")
        print("-" * 40)
        
        try:
            response = self.session.get(self.base_url, timeout=5)
            self.log_result("Server responds to requests", response.status_code == 200)
            
            if response.status_code == 200:
                # Test response contains HTML
                self.log_result("Response contains HTML", 'html' in response.text.lower())
                
                # Test title contains expected text
                self.log_result("Page title correct", 'Safety Detector' in response.text)
                
        except Exception as e:
            self.log_result("Server connectivity", False, str(e))
    
    def test_all_pages_load(self):
        """Test all main pages load without errors"""
        print("\n📄 Testing Page Loading")
        print("-" * 40)
        
        pages = [
            ('/', 'Dashboard'),
            ('/upload', 'Upload'),
            ('/camera', 'Camera'),
            ('/history', 'History'),
            ('/config', 'Configuration'),
            ('/debug', 'Debug')
        ]
        
        for url, name in pages:
            try:
                response = self.session.get(f"{self.base_url}{url}", timeout=10)
                self.log_result(f"{name} page loads", response.status_code == 200)
                
                if response.status_code == 200:
                    # Test page contains expected elements
                    html = response.text.lower()
                    
                    # All pages should have navigation
                    self.log_result(f"{name} has navigation", 'navbar' in html)
                    
                    # All pages should have Bootstrap
                    self.log_result(f"{name} has Bootstrap", 'bootstrap' in html)
                    
                    # All pages should have custom CSS
                    self.log_result(f"{name} has custom styles", 'style.css' in html)
                    
            except Exception as e:
                self.log_result(f"{name} page load", False, str(e))
    
    def test_api_endpoints(self):
        """Test API endpoints"""
        print("\n🔌 Testing API Endpoints")
        print("-" * 40)
        
        # Test analysis status endpoint
        try:
            response = self.session.get(f"{self.base_url}/analysis_status")
            self.log_result("Analysis status endpoint", response.status_code == 200)
            
            if response.status_code == 200:
                try:
                    data = response.json()
                    self.log_result("Analysis status JSON valid", isinstance(data, dict))
                except:
                    self.log_result("Analysis status JSON valid", False, "Invalid JSON")
                    
        except Exception as e:
            self.log_result("Analysis status endpoint", False, str(e))
        
        # Test ESP32 endpoint with invalid data
        try:
            response = self.session.post(
                f"{self.base_url}/test_esp32",
                json={},
                headers={'Content-Type': 'application/json'}
            )
            # Should return 400 for missing IP
            self.log_result("ESP32 endpoint validation", response.status_code == 400)
            
        except Exception as e:
            self.log_result("ESP32 endpoint", False, str(e))
    
    def test_static_files(self):
        """Test static files are accessible"""
        print("\n📁 Testing Static Files")
        print("-" * 40)
        
        static_files = [
            '/static/css/style.css',
            '/static/js/main.js'
        ]
        
        for file_path in static_files:
            try:
                response = self.session.get(f"{self.base_url}{file_path}")
                self.log_result(f"Static file {file_path}", response.status_code == 200)
            except Exception as e:
                self.log_result(f"Static file {file_path}", False, str(e))
    
    def test_form_validation(self):
        """Test form validation"""
        print("\n📝 Testing Form Validation")
        print("-" * 40)
        
        # Test upload with no file
        try:
            response = self.session.post(f"{self.base_url}/upload_video")
            # Should redirect or show error
            self.log_result("Upload validation (no file)", response.status_code in [302, 400])
            
        except Exception as e:
            self.log_result("Upload validation", False, str(e))
        
        # Test upload with invalid file type
        try:
            files = {'video': ('test.txt', b'text content', 'text/plain')}
            response = self.session.post(f"{self.base_url}/upload_video", files=files)
            # Should reject invalid file type
            self.log_result("Upload file type validation", 
                          response.status_code in [302, 400] or 'Invalid file type' in response.text)
            
        except Exception as e:
            self.log_result("Upload file type validation", False, str(e))
    
    def test_error_handling(self):
        """Test error handling"""
        print("\n🚨 Testing Error Handling")
        print("-" * 40)
        
        # Test 404 pages
        try:
            response = self.session.get(f"{self.base_url}/nonexistent-page")
            self.log_result("404 handling", response.status_code == 404)
            
        except Exception as e:
            self.log_result("404 handling", False, str(e))
        
        # Test invalid upload ID
        try:
            response = self.session.get(f"{self.base_url}/history/99999")
            # Should redirect or show error
            self.log_result("Invalid upload ID handling", response.status_code in [302, 404])
            
        except Exception as e:
            self.log_result("Invalid upload ID handling", False, str(e))
    
    def test_responsive_elements(self):
        """Test responsive design elements"""
        print("\n📱 Testing Responsive Elements")
        print("-" * 40)
        
        try:
            response = self.session.get(self.base_url)
            if response.status_code == 200:
                html = response.text
                
                # Check for viewport meta tag
                self.log_result("Viewport meta tag", 'viewport' in html)
                
                # Check for Bootstrap responsive classes
                self.log_result("Bootstrap grid system", 'col-md-' in html or 'col-lg-' in html)
                
                # Check for responsive navigation
                self.log_result("Mobile navigation toggle", 'navbar-toggler' in html)
                
        except Exception as e:
            self.log_result("Responsive elements check", False, str(e))
    
    def test_javascript_integration(self):
        """Test JavaScript integration"""
        print("\n📜 Testing JavaScript Integration")
        print("-" * 40)
        
        pages_with_js = [
            ('/', 'Dashboard'),
            ('/upload', 'Upload'),
            ('/camera', 'Camera'),
            ('/config', 'Configuration'),
            ('/debug', 'Debug')
        ]
        
        for url, name in pages_with_js:
            try:
                response = self.session.get(f"{self.base_url}{url}")
                if response.status_code == 200:
                    html = response.text
                    
                    # Check for JavaScript
                    self.log_result(f"{name} has JavaScript", '<script' in html)
                    
                    # Check for event listeners
                    self.log_result(f"{name} has event listeners", 'addEventListener' in html)
                    
                    # Check for error handling
                    self.log_result(f"{name} has error handling", 'try {' in html or 'catch' in html)
                    
            except Exception as e:
                self.log_result(f"{name} JavaScript check", False, str(e))
    
    def test_security_headers(self):
        """Test basic security headers"""
        print("\n🔒 Testing Security Headers")
        print("-" * 40)
        
        try:
            response = self.session.get(self.base_url)
            headers = response.headers
            
            # Check for basic security headers (Flask may not set all by default)
            self.log_result("Content-Type header", 'Content-Type' in headers)
            
            # Check for no sensitive information in headers
            sensitive_headers = ['server', 'x-powered-by']
            for header in sensitive_headers:
                if header.lower() in [h.lower() for h in headers.keys()]:
                    self.log_result(f"No {header} header exposure", False, f"{header} header exposed")
                else:
                    self.log_result(f"No {header} header exposure", True)
                    
        except Exception as e:
            self.log_result("Security headers check", False, str(e))
    
    def test_file_upload_functionality(self):
        """Test file upload functionality"""
        print("\n📤 Testing File Upload")
        print("-" * 40)
        
        try:
            # Test with valid video file
            test_video_content = b"FAKE_MP4_HEADER" + b"x" * 1000  # 1KB fake video
            files = {'video': ('test_video.mp4', test_video_content, 'video/mp4')}
            
            response = self.session.post(f"{self.base_url}/upload_video", files=files)
            
            # Should either succeed (200) or redirect (302)
            self.log_result("Video upload functionality", response.status_code in [200, 302])
            
            if response.status_code == 302:
                # Check if redirected to analysis page
                location = response.headers.get('Location', '')
                self.log_result("Upload redirects to analysis", 'analyze' in location)
            
        except Exception as e:
            self.log_result("Video upload test", False, str(e))
    
    def generate_test_report(self):
        """Generate comprehensive test report"""
        print("\n" + "="*60)
        print("🔍 AUTOMATED FRONTEND TEST REPORT")
        print("="*60)
        
        total_tests = self.results['passed'] + self.results['failed']
        success_rate = (self.results['passed'] / total_tests * 100) if total_tests > 0 else 0
        
        print(f"\n📊 TEST SUMMARY:")
        print(f"   ✅ Tests Passed: {self.results['passed']}")
        print(f"   ❌ Tests Failed: {self.results['failed']}")
        print(f"   📈 Success Rate: {success_rate:.1f}%")
        print(f"   🕐 Test Time: {datetime.now().isoformat()}")
        
        if self.results['bugs']:
            print(f"\n🐛 ISSUES FOUND ({len(self.results['bugs'])}):")
            print("-" * 40)
            for i, bug in enumerate(self.results['bugs'], 1):
                print(f"{i}. {bug['test']}")
                print(f"   Error: {bug['error']}")
                print(f"   Time: {bug['timestamp']}")
                print()
        else:
            print("\n🎉 NO ISSUES FOUND! All automated tests passed!")
        
        # Save detailed report
        report_filename = f"automated_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"\n📄 Detailed report saved to: {report_filename}")
        print("="*60)
        
        return success_rate > 80  # Return True if success rate > 80%
    
    def run_all_tests(self):
        """Run all automated tests"""
        print("🤖 AUTOMATED FRONTEND TESTING")
        print("="*60)
        print("Running automated tests while web server is active...")
        
        # Give server time to fully start
        time.sleep(2)
        
        # Run all test suites
        self.test_server_connectivity()
        self.test_all_pages_load()
        self.test_api_endpoints()
        self.test_static_files()
        self.test_form_validation()
        self.test_error_handling()
        self.test_responsive_elements()
        self.test_javascript_integration()
        self.test_security_headers()
        self.test_file_upload_functionality()
        
        # Generate comprehensive report
        return self.generate_test_report()

if __name__ == "__main__":
    print("🛡️ Starting Automated Frontend Tests...")
    print("Make sure the web server is running: python run_web_server.py")
    print()
    
    tester = AutomatedFrontendTests()
    success = tester.run_all_tests()
    
    if success:
        print("\n🎯 OVERALL RESULT: TESTS PASSED ✅")
    else:
        print("\n🚨 OVERALL RESULT: ISSUES FOUND ❌")
        print("Please review the bugs and run manual testing for complete coverage.")
