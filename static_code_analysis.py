# Static Code Analysis for Frontend Issues
# Analyzes HTML templates and JavaScript for potential bugs

import os
import re
from pathlib import Path

class FrontendCodeAnalyzer:
    def __init__(self, web_interface_dir="web_interface"):
        self.web_dir = Path(web_interface_dir)
        self.issues = []
        self.warnings = []
        
    def log_issue(self, file_path, issue_type, message, line_num=None):
        """Log an issue found in code"""
        issue = {
            'file': str(file_path),
            'type': issue_type,
            'message': message,
            'line': line_num
        }
        if issue_type == 'ERROR':
            self.issues.append(issue)
        else:
            self.warnings.append(issue)
        
        line_info = f" (line {line_num})" if line_num else ""
        print(f"{'🔴' if issue_type == 'ERROR' else '🟡'} {issue_type}: {file_path}{line_info}")
        print(f"   {message}")
    
    def analyze_html_templates(self):
        """Analyze HTML templates for issues"""
        print("\n📄 Analyzing HTML Templates")
        print("-" * 40)
        
        template_dir = self.web_dir / "templates"
        if not template_dir.exists():
            self.log_issue("templates/", "ERROR", "Templates directory not found")
            return
        
        for template_file in template_dir.glob("*.html"):
            try:
                with open(template_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                
                self.check_html_structure(template_file, content, lines)
                self.check_bootstrap_usage(template_file, content, lines)
                self.check_javascript_in_templates(template_file, content, lines)
                self.check_form_validation(template_file, content, lines)
                
            except Exception as e:
                self.log_issue(template_file, "ERROR", f"Could not read template: {e}")
    
    def check_html_structure(self, file_path, content, lines):
        """Check HTML structure issues"""
        
        # Check for missing DOCTYPE
        if not content.strip().startswith('<!DOCTYPE html>') and 'extends' not in content:
            self.log_issue(file_path, "WARNING", "Missing DOCTYPE declaration")
        
        # Check for missing viewport meta tag
        if 'viewport' not in content and 'base.html' not in str(file_path):
            self.log_issue(file_path, "WARNING", "Missing viewport meta tag")
        
        # Check for unclosed tags (basic check)
        open_tags = re.findall(r'<(\w+)[^>]*>', content)
        close_tags = re.findall(r'</(\w+)>', content)
        
        # Count self-closing and template tags
        self_closing = ['img', 'input', 'br', 'hr', 'meta', 'link']
        template_tags = [tag for tag in open_tags if tag.startswith('{')]
        
        for tag in set(open_tags):
            if tag not in self_closing and not tag.startswith('{'):
                open_count = open_tags.count(tag)
                close_count = close_tags.count(tag)
                if open_count != close_count and abs(open_count - close_count) > 1:
                    self.log_issue(file_path, "WARNING", f"Possible unclosed {tag} tags")
    
    def check_bootstrap_usage(self, file_path, content, lines):
        """Check Bootstrap usage issues"""
        
        # Check for missing Bootstrap classes
        if 'class=' in content:
            # Check for common Bootstrap issues
            if 'btn' in content and 'btn-' not in content:
                self.log_issue(file_path, "WARNING", "Button without Bootstrap style class")
            
            # Check for responsive grid usage
            if 'col-' in content and 'row' not in content:
                self.log_issue(file_path, "WARNING", "Bootstrap columns without row container")
        
        # Check for inline styles (should use CSS classes)
        inline_styles = re.findall(r'style\s*=\s*["\'][^"\']*["\']', content)
        if len(inline_styles) > 5:  # Allow some inline styles
            self.log_issue(file_path, "WARNING", f"Many inline styles found ({len(inline_styles)})")
    
    def check_javascript_in_templates(self, file_path, content, lines):
        """Check JavaScript in templates"""
        
        # Find JavaScript blocks
        js_blocks = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
        
        for i, js_block in enumerate(js_blocks):
            # Check for missing error handling
            if 'try' not in js_block and 'catch' not in js_block and len(js_block.strip()) > 100:
                self.log_issue(file_path, "WARNING", f"JavaScript block {i+1} missing error handling")
            
            # Check for console.log (should be removed in production)
            if 'console.log' in js_block:
                self.log_issue(file_path, "WARNING", f"JavaScript block {i+1} contains console.log")
            
            # Check for missing semicolons
            js_lines = js_block.split('\n')
            for line_num, line in enumerate(js_lines):
                line = line.strip()
                if (line.endswith(')') or line.endswith('}')) and not line.endswith(';') and not line.endswith(','):
                    if line and not line.startswith('//') and not line.startswith('*'):
                        self.log_issue(file_path, "WARNING", f"Missing semicolon in JS line: {line[:50]}...")
    
    def check_form_validation(self, file_path, content, lines):
        """Check form validation"""
        
        # Find forms
        forms = re.findall(r'<form[^>]*>(.*?)</form>', content, re.DOTALL)
        
        for i, form in enumerate(forms):
            # Check for CSRF protection (Flask-WTF)
            if 'csrf_token' not in form and 'method="post"' in form.lower():
                self.log_issue(file_path, "WARNING", f"Form {i+1} missing CSRF protection")
            
            # Check for required field validation
            inputs = re.findall(r'<input[^>]*>', form)
            has_required = any('required' in inp for inp in inputs)
            has_validation = 'is-invalid' in form or 'was-validated' in form
            
            if inputs and not has_required and not has_validation:
                self.log_issue(file_path, "WARNING", f"Form {i+1} missing client-side validation")
    
    def analyze_flask_app(self):
        """Analyze Flask app for issues"""
        print("\n🐍 Analyzing Flask Application")
        print("-" * 40)
        
        app_file = self.web_dir / "app.py"
        if not app_file.exists():
            self.log_issue("app.py", "ERROR", "Flask app file not found")
            return
        
        try:
            with open(app_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            self.check_flask_routes(app_file, content, lines)
            self.check_error_handling_flask(app_file, content, lines)
            self.check_security_issues(app_file, content, lines)
            
        except Exception as e:
            self.log_issue(app_file, "ERROR", f"Could not read Flask app: {e}")
    
    def check_flask_routes(self, file_path, content, lines):
        """Check Flask routes for issues"""
        
        # Find route definitions
        routes = re.findall(r'@app\.route\([^)]+\)', content)
        
        # Check for missing error handling in routes
        route_functions = re.findall(r'@app\.route.*?\ndef\s+(\w+)', content, re.DOTALL)
        
        for func_name in route_functions:
            func_pattern = rf'def\s+{func_name}.*?(?=def|\Z)'
            func_match = re.search(func_pattern, content, re.DOTALL)
            
            if func_match:
                func_content = func_match.group(0)
                
                # Check for try-catch blocks
                if 'try:' not in func_content and 'except' not in func_content:
                    if 'return' in func_content and len(func_content) > 200:
                        self.log_issue(file_path, "WARNING", f"Route {func_name} missing error handling")
                
                # Check for SQL injection prevention (basic check)
                if 'query' in func_content.lower() and 'format' in func_content:
                    self.log_issue(file_path, "WARNING", f"Route {func_name} possible SQL injection risk")
    
    def check_error_handling_flask(self, file_path, content, lines):
        """Check error handling in Flask app"""
        
        # Check for global error handlers
        error_handlers = re.findall(r'@app\.errorhandler\((\d+)\)', content)
        
        common_errors = ['404', '500', '400']
        for error_code in common_errors:
            if error_code not in error_handlers:
                self.log_issue(file_path, "WARNING", f"Missing error handler for {error_code}")
        
        # Check for logging
        if 'import logging' not in content and 'logger' not in content:
            self.log_issue(file_path, "WARNING", "No logging implementation found")
    
    def check_security_issues(self, file_path, content, lines):
        """Check for security issues"""
        
        # Check for hardcoded secrets
        secret_patterns = [
            r'secret_key\s*=\s*["\'][^"\']{10,}["\']',
            r'password\s*=\s*["\'][^"\']+["\']',
            r'api_key\s*=\s*["\'][^"\']+["\']'
        ]
        
        for pattern in secret_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if 'your_' not in match.lower() and 'example' not in match.lower():
                    self.log_issue(file_path, "ERROR", "Hardcoded secret found")
        
        # Check for debug mode in production
        if "debug=True" in content:
            self.log_issue(file_path, "WARNING", "Debug mode enabled (should be False in production)")
        
        # Check for CORS issues
        if 'CORS' in content or 'cross_origin' in content:
            if '*' in content:
                self.log_issue(file_path, "WARNING", "Permissive CORS configuration")
    
    def analyze_static_files(self):
        """Analyze static CSS and JS files"""
        print("\n📁 Analyzing Static Files")
        print("-" * 40)
        
        static_dir = self.web_dir / "static"
        if not static_dir.exists():
            self.log_issue("static/", "ERROR", "Static directory not found")
            return
        
        # Check CSS files
        css_files = list(static_dir.rglob("*.css"))
        for css_file in css_files:
            self.check_css_file(css_file)
        
        # Check JS files
        js_files = list(static_dir.rglob("*.js"))
        for js_file in js_files:
            self.check_js_file(js_file)
    
    def check_css_file(self, file_path):
        """Check CSS file for issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for missing vendor prefixes (basic check)
            modern_props = ['transform', 'transition', 'animation', 'box-shadow']
            for prop in modern_props:
                if f'{prop}:' in content and f'-webkit-{prop}:' not in content:
                    self.log_issue(file_path, "WARNING", f"Missing vendor prefixes for {prop}")
            
            # Check for !important overuse
            important_count = content.count('!important')
            if important_count > 10:
                self.log_issue(file_path, "WARNING", f"Excessive use of !important ({important_count} times)")
            
        except Exception as e:
            self.log_issue(file_path, "ERROR", f"Could not read CSS file: {e}")
    
    def check_js_file(self, file_path):
        """Check JavaScript file for issues"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for missing 'use strict'
            if "'use strict'" not in content and '"use strict"' not in content:
                if len(content.strip()) > 100:
                    self.log_issue(file_path, "WARNING", "Missing 'use strict' directive")
            
            # Check for console.log statements
            console_logs = content.count('console.log')
            if console_logs > 0:
                self.log_issue(file_path, "WARNING", f"Contains {console_logs} console.log statements")
            
            # Check for missing error handling
            try_count = content.count('try {')
            catch_count = content.count('catch')
            if try_count != catch_count:
                self.log_issue(file_path, "WARNING", "Mismatched try-catch blocks")
            
        except Exception as e:
            self.log_issue(file_path, "ERROR", f"Could not read JS file: {e}")
    
    def generate_analysis_report(self):
        """Generate analysis report"""
        print("\n" + "="*60)
        print("📊 STATIC CODE ANALYSIS REPORT")
        print("="*60)
        
        total_issues = len(self.issues) + len(self.warnings)
        
        print(f"\n📈 ANALYSIS SUMMARY:")
        print(f"   🔴 Critical Issues: {len(self.issues)}")
        print(f"   🟡 Warnings: {len(self.warnings)}")
        print(f"   📊 Total Issues: {total_issues}")
        
        if self.issues:
            print(f"\n🔴 CRITICAL ISSUES ({len(self.issues)}):")
            print("-" * 40)
            for issue in self.issues:
                print(f"• {issue['file']}: {issue['message']}")
        
        if self.warnings:
            print(f"\n🟡 WARNINGS ({len(self.warnings)}):")
            print("-" * 40)
            for warning in self.warnings:
                print(f"• {warning['file']}: {warning['message']}")
        
        if total_issues == 0:
            print("\n🎉 NO ISSUES FOUND! Code looks good!")
        
        print("="*60)
        return total_issues == 0
    
    def run_analysis(self):
        """Run complete static analysis"""
        print("🔍 FRONTEND STATIC CODE ANALYSIS")
        print("="*60)
        print("Analyzing code for potential issues...")
        
        self.analyze_html_templates()
        self.analyze_flask_app()
        self.analyze_static_files()
        
        return self.generate_analysis_report()

if __name__ == "__main__":
    analyzer = FrontendCodeAnalyzer()
    clean_code = analyzer.run_analysis()
    
    if clean_code:
        print("\n🎯 RESULT: CODE ANALYSIS PASSED ✅")
    else:
        print("\n🚨 RESULT: ISSUES FOUND ❌")
        print("Please review and fix the issues above.")
