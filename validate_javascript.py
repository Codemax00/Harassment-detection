#!/usr/bin/env python3
"""
JavaScript Validation Tool for Safety Detector
Checks for common JavaScript syntax errors in templates
"""

import os
import re
from pathlib import Path

def validate_javascript_in_file(file_path):
    """Validate JavaScript syntax in a template file"""
    print(f"🔍 Checking {file_path.name}...")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    errors = []
    
    # Extract JavaScript blocks
    js_blocks = re.findall(r'<script[^>]*>(.*?)</script>', content, re.DOTALL)
    
    for i, js_block in enumerate(js_blocks):
        lines = js_block.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            line = line.strip()
            if not line or line.startswith('//'):
                continue
            
            # Check for common syntax errors
            
            # Double closing braces
            if '}' in line and line.count('}') > line.count('{'):
                brace_diff = line.count('}') - line.count('{')
                if brace_diff > 1:
                    errors.append(f"Line {line_num}: Possible extra closing braces: {line}")
            
            # Missing semicolons (basic check)
            if (line.endswith('}') and not line.endswith('};') and 
                ('function' not in line and 'if' not in line and 'else' not in line and 
                 'for' not in line and 'while' not in line and 'try' not in line and 'catch' not in line)):
                # This is a heuristic check - may have false positives
                pass
            
            # Unmatched parentheses (basic check)
            if line.count('(') != line.count(')'):
                errors.append(f"Line {line_num}: Unmatched parentheses: {line}")
            
            # Unmatched brackets
            if line.count('[') != line.count(']'):
                errors.append(f"Line {line_num}: Unmatched brackets: {line}")
    
    return errors

def validate_all_templates():
    """Validate JavaScript in all template files"""
    print("🧪 JavaScript Validation Tool")
    print("=" * 40)
    
    templates_dir = Path('web_interface/templates')
    
    if not templates_dir.exists():
        print("❌ Templates directory not found")
        return False
    
    template_files = list(templates_dir.glob('*.html'))
    
    if not template_files:
        print("❌ No template files found")
        return False
    
    total_errors = 0
    
    for template_file in template_files:
        errors = validate_javascript_in_file(template_file)
        
        if errors:
            print(f"❌ {template_file.name}: {len(errors)} errors")
            for error in errors:
                print(f"   {error}")
            total_errors += len(errors)
        else:
            print(f"✅ {template_file.name}: No errors")
    
    print("\n" + "=" * 40)
    print(f"📊 Total JavaScript errors found: {total_errors}")
    
    if total_errors == 0:
        print("🎉 All JavaScript syntax looks good!")
    else:
        print("⚠️ JavaScript errors found. Please fix them.")
    
    return total_errors == 0

if __name__ == "__main__":
    validate_all_templates()
