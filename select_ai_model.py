# AI Model Selector - Choose between Gemini and OpenAI
# Simple utility to configure which AI model to use

import json
import os
import sys

# Add helpers folder to path
sys.path.append(os.path.join(os.path.dirname(__file__), 'helpers'))

from multi_ai import setup_ai_models, get_ai_model_info

def load_current_config():
    """Load current AI configuration"""
    try:
        with open('ai_config.json', 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading config: {e}")
        return None

def save_config(config):
    """Save updated configuration"""
    try:
        with open('ai_config.json', 'w') as f:
            json.dump(config, f, indent=2)
        return True
    except Exception as e:
        print(f"Error saving config: {e}")
        return False

def show_current_config():
    """Display current AI model configuration"""
    config = load_current_config()
    if not config:
        return
    
    settings = config['ai_model_settings']
    active_model = settings['active_model']
    
    print("=== Current AI Model Configuration ===")
    print(f"Active Model: {active_model.upper()}")
    print()
    
    for model_name, model_config in settings['model_configs'].items():
        status = "ACTIVE" if model_name == active_model else "Available"
        print(f"{model_name.upper()} ({status}):")
        print(f"  Description: {model_config['description']}")
        print(f"  Model: {model_config['model_name']}")
        print(f"  API Key: {model_config['api_key_env']}")
        print(f"  Response Time: {model_config['avg_response_time']}")
        print(f"  Strengths: {', '.join(model_config['strengths'])}")
        print()

def test_ai_models():
    """Test available AI models"""
    print("=== Testing AI Models ===")
    print("Loading AI models...")
    
    try:
        ai_manager = setup_ai_models()
        model_info = get_ai_model_info()
        
        print(f"Available models: {model_info['available_models']}")
        print(f"Active model: {model_info['active_model']}")
        print()
        
        for model_name, status in model_info['model_status'].items():
            print(f"{model_name.upper()}: {status}")
        
        return ai_manager
    except Exception as e:
        print(f"Error testing models: {e}")
        return None

def switch_active_model():
    """Interactive model switching"""
    config = load_current_config()
    if not config:
        return
    
    settings = config['ai_model_settings']
    available_models = settings['available_models']
    current_model = settings['active_model']
    
    print("=== Switch AI Model ===")
    print(f"Current model: {current_model.upper()}")
    print()
    print("Available models:")
    
    for i, model in enumerate(available_models, 1):
        model_config = settings['model_configs'][model]
        status = " (CURRENT)" if model == current_model else ""
        print(f"  {i}. {model.upper()}{status}")
        print(f"     {model_config['description']}")
        print()
    
    try:
        choice = input("Select model number (or press Enter to cancel): ").strip()
        
        if not choice:
            print("No changes made.")
            return
        
        choice_num = int(choice)
        if 1 <= choice_num <= len(available_models):
            new_model = available_models[choice_num - 1]
            
            if new_model == current_model:
                print(f"{new_model.upper()} is already the active model.")
                return
            
            # Update configuration
            settings['active_model'] = new_model
            
            if save_config(config):
                print(f"✅ Successfully switched to {new_model.upper()}")
                print("Restart the application to use the new model.")
            else:
                print("❌ Failed to save configuration")
        else:
            print("Invalid selection.")
            
    except ValueError:
        print("Invalid input. Please enter a number.")
    except KeyboardInterrupt:
        print("\nCancelled.")

def show_model_comparison():
    """Show detailed comparison between models"""
    print("=== AI Model Comparison ===")
    print()
    
    print("GOOGLE GEMINI AI:")
    print("  ✓ Fast response time (1-3 seconds)")
    print("  ✓ Good multimodal understanding")
    print("  ✓ Free tier available")
    print("  ✓ Reliable for real-time use")
    print("  - Less detailed explanations")
    print()
    
    print("OPENAI GPT-4 VISION:")
    print("  ✓ Excellent reasoning capabilities")
    print("  ✓ Very detailed explanations")
    print("  ✓ High accuracy in complex scenarios")
    print("  ✓ Advanced contextual understanding")
    print("  - Slower response time (2-5 seconds)")
    print("  - Costs more per API call")
    print()
    
    print("RECOMMENDATION:")
    print("- Use GEMINI for real-time monitoring (faster)")
    print("- Use OPENAI for detailed analysis (more accurate)")
    print("- Both models work well for harassment detection")

def main():
    """Main menu for AI model selection"""
    while True:
        print("\n" + "="*50)
        print("AI MODEL SELECTOR - Safety Detector")
        print("="*50)
        print()
        print("1. Show current configuration")
        print("2. Test available models")
        print("3. Switch active model")
        print("4. Compare models")
        print("5. Exit")
        print()
        
        try:
            choice = input("Select option (1-5): ").strip()
            
            if choice == "1":
                show_current_config()
            elif choice == "2":
                test_ai_models()
            elif choice == "3":
                switch_active_model()
            elif choice == "4":
                show_model_comparison()
            elif choice == "5":
                print("Goodbye!")
                break
            else:
                print("Invalid choice. Please select 1-5.")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    main()
