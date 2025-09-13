# Multi-AI Helper for Harassment Detection
# Supports both Google Gemini and OpenAI GPT-4 Vision models

import google.generativeai as genai
import openai
import cv2
import base64
import os
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class AIModelManager:
    """Manages multiple AI models for harassment detection"""
    
    def __init__(self, config_file="ai_config.json"):
        """Initialize the AI model manager with configuration"""
        self.config = self.load_config(config_file)
        self.active_model = self.config['ai_model_settings']['active_model']
        self.models = {}
        self.setup_models()
    
    def load_config(self, config_file):
        """Load AI configuration from JSON file"""
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', config_file)
            with open(config_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading config: {e}")
            # Return default configuration
            return {
                "ai_model_settings": {
                    "active_model": "gemini",
                    "model_configs": {
                        "gemini": {"model_name": "gemini-1.5-flash", "api_key_env": "GEMINI_API_KEY"},
                        "openai": {"model_name": "gpt-4-vision-preview", "api_key_env": "OPENAI_API_KEY"}
                    }
                }
            }
    
    def setup_models(self):
        """Initialize available AI models"""
        model_configs = self.config['ai_model_settings']['model_configs']
        
        # Setup Gemini
        if 'gemini' in model_configs:
            self.models['gemini'] = self.setup_gemini(model_configs['gemini'])
        
        # Setup OpenAI
        if 'openai' in model_configs:
            self.models['openai'] = self.setup_openai(model_configs['openai'])
        
        print(f"AI Models initialized: {list(self.models.keys())}")
        print(f"Active model: {self.active_model}")
    
    def setup_gemini(self, config):
        """Setup Google Gemini AI"""
        try:
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                print(f"Warning: {config['api_key_env']} not found in environment")
                return None
            
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(config['model_name'])
            print("✓ Gemini AI initialized successfully")
            return model
        except Exception as e:
            print(f"✗ Gemini AI setup failed: {e}")
            return None
    
    def setup_openai(self, config):
        """Setup OpenAI GPT-4 Vision"""
        try:
            api_key = os.getenv(config['api_key_env'])
            if not api_key:
                print(f"Warning: {config['api_key_env']} not found in environment")
                return None
            
            openai.api_key = api_key
            # Test connection
            openai.models.list()
            print("✓ OpenAI API initialized successfully")
            return "openai_client"  # We'll use the global client
        except Exception as e:
            print(f"✗ OpenAI API setup failed: {e}")
            return None
    
    def get_active_model(self):
        """Get the currently active AI model"""
        if self.active_model in self.models and self.models[self.active_model]:
            return self.models[self.active_model]
        
        # Fallback to first available model
        for model_name, model in self.models.items():
            if model:
                print(f"Falling back to {model_name} model")
                self.active_model = model_name
                return model
        
        print("No AI models available!")
        return None
    
    def switch_model(self, model_name):
        """Switch to a different AI model"""
        if model_name in self.models and self.models[model_name]:
            self.active_model = model_name
            print(f"Switched to {model_name} model")
            return True
        else:
            print(f"Model {model_name} not available")
            return False
    
    def frame_to_base64(self, frame):
        """Convert video frame to base64 for AI analysis"""
        success, buffer = cv2.imencode('.jpg', frame)
        if not success:
            return None
        
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        return img_base64
    
    def create_harassment_prompt(self):
        """Create the harassment detection prompt"""
        if 'prompt_templates' in self.config:
            template = self.config['prompt_templates']['harassment_detection']
            
            prompt = f"{template['base_prompt']}\n\n"
            prompt += "Look for these warning signs:\n"
            
            for behavior in template['specific_behaviors']:
                prompt += f"- {behavior}\n"
            
            prompt += f"\n{template['response_format']}\n"
            prompt += f"{template['context_instructions']}"
            
            return prompt
        else:
            # Default prompt
            return """
            Look at this image carefully. I need you to check if there is any harassment happening.
            
            Look for these warning signs:
            - Someone following another person closely
            - Someone blocking another person's path
            - Aggressive body language or gestures
            - Someone cornering another person
            - Any threatening behavior
            
            Answer with just:
            - "ALERT" if you see potential harassment
            - "SAFE" if everything looks normal
            
            Then give a short reason why.
            """
    
    def analyze_with_gemini(self, frame):
        """Analyze frame using Google Gemini"""
        try:
            model = self.models['gemini']
            if not model:
                return False, "Gemini model not available"
            
            # Convert frame to base64
            img_data = self.frame_to_base64(frame)
            if img_data is None:
                return False, "Could not process image"
            
            # Create the prompt
            prompt = self.create_harassment_prompt()
            
            # Send to Gemini
            response = model.generate_content([
                prompt, 
                {"mime_type": "image/jpeg", "data": img_data}
            ])
            
            if response and response.text:
                result = response.text.strip()
                if "ALERT" in result.upper():
                    return True, result
                else:
                    return False, result
            else:
                return False, "No response from Gemini"
                
        except Exception as e:
            print(f"Gemini analysis error: {e}")
            return False, f"Gemini error: {e}"
    
    def analyze_with_openai(self, frame):
        """Analyze frame using OpenAI GPT-4 Vision"""
        try:
            if 'openai' not in self.models or not self.models['openai']:
                return False, "OpenAI model not available"
            
            # Convert frame to base64
            img_data = self.frame_to_base64(frame)
            if img_data is None:
                return False, "Could not process image"
            
            # Create the prompt
            prompt = self.create_harassment_prompt()
            
            # Send to OpenAI
            response = openai.chat.completions.create(
                model="gpt-4-vision-preview",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_data}"
                                }
                            }
                        ]
                    }
                ],
                max_tokens=300
            )
            
            if response.choices and response.choices[0].message:
                result = response.choices[0].message.content.strip()
                if "ALERT" in result.upper():
                    return True, result
                else:
                    return False, result
            else:
                return False, "No response from OpenAI"
                
        except Exception as e:
            print(f"OpenAI analysis error: {e}")
            return False, f"OpenAI error: {e}"
    
    def analyze_for_harassment(self, frame):
        """Main function to analyze frame for harassment using active model"""
        if self.active_model == "gemini":
            return self.analyze_with_gemini(frame)
        elif self.active_model == "openai":
            return self.analyze_with_openai(frame)
        else:
            return False, f"Unknown model: {self.active_model}"
    
    def get_model_info(self):
        """Get information about available models"""
        info = {
            'active_model': self.active_model,
            'available_models': list(self.models.keys()),
            'model_status': {}
        }
        
        for model_name, model in self.models.items():
            info['model_status'][model_name] = "Available" if model else "Not configured"
        
        return info

# Global AI manager instance
ai_manager = None

def setup_ai_models(config_file="ai_config.json"):
    """Setup AI models with configuration"""
    global ai_manager
    ai_manager = AIModelManager(config_file)
    return ai_manager

def get_ai_manager():
    """Get the global AI manager instance"""
    global ai_manager
    if ai_manager is None:
        ai_manager = setup_ai_models()
    return ai_manager

def analyze_for_harassment(model_placeholder, frame):
    """Analyze frame for harassment (compatible with existing code)"""
    manager = get_ai_manager()
    return manager.analyze_for_harassment(frame)

def count_people_for_analysis(people_list):
    """Count how many people we detected (unchanged function)"""
    total_people = len(people_list)
    
    if total_people == 0:
        return "No people detected"
    elif total_people == 1:
        return "1 person present"
    elif total_people == 2:
        return "2 people present - monitoring interaction"
    else:
        return f"{total_people} people present - busy area"

def switch_ai_model(model_name):
    """Switch to a different AI model"""
    manager = get_ai_manager()
    return manager.switch_model(model_name)

def get_ai_model_info():
    """Get information about available AI models"""
    manager = get_ai_manager()
    return manager.get_model_info()

# Backward compatibility functions
def setup_gemini():
    """Setup Gemini AI (backward compatibility)"""
    manager = get_ai_manager()
    return manager.models.get('gemini')

def setup_openai():
    """Setup OpenAI (new function)"""
    manager = get_ai_manager()
    return manager.models.get('openai')
