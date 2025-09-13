# Multi-AI Setup Guide - Safety Detector

## Overview
The Safety Detector now supports **both Google Gemini and OpenAI GPT-4 Vision** for harassment detection! You can easily switch between models based on your needs.

## 🎯 AI Model Options

### Google Gemini AI
- **Speed**: Fast (1-3 seconds per analysis)
- **Cost**: More affordable
- **Best for**: Real-time monitoring
- **Strengths**: Quick responses, good reasoning
- **API Key**: GEMINI_API_KEY

### OpenAI GPT-4 Vision  
- **Speed**: Moderate (2-5 seconds per analysis)
- **Cost**: Higher per API call
- **Best for**: Detailed analysis
- **Strengths**: Excellent reasoning, detailed explanations
- **API Key**: OPENAI_API_KEY

## 🚀 Quick Setup

### 1. Install Dependencies
```bash
pip install openai==1.3.0
# or run: python install_dependencies.py
```

### 2. Configure API Keys
Edit your `.env` file:
```env
# Google Gemini API key
GEMINI_API_KEY=your_gemini_key_here

# OpenAI API key (optional - only if using OpenAI)
OPENAI_API_KEY=your_openai_key_here
```

### 3. Choose Your AI Model
```bash
python select_ai_model.py
```

### 4. Start Using
```bash
python start_here.py
```

## 📋 AI Model Selector Features

The `select_ai_model.py` utility provides:

1. **View Current Configuration**
   - Shows which model is active
   - Lists available models and their status

2. **Switch Models Easily**
   - Interactive menu to choose between Gemini/OpenAI
   - Instant switching with configuration save

3. **Test Model Availability**
   - Checks API connectivity
   - Validates API keys

4. **Compare Models**
   - Side-by-side feature comparison
   - Recommendations for different use cases

## 🔧 Configuration File

The `ai_config.json` file controls:
- Active AI model selection
- Model-specific settings
- Harassment detection prompts
- Analysis parameters

```json
{
  "ai_model_settings": {
    "active_model": "gemini",  // or "openai"
    "available_models": ["gemini", "openai"]
  }
}
```

## 💡 When to Use Each Model

### Use Gemini When:
- ✅ Real-time monitoring (live camera)
- ✅ Cost is a concern
- ✅ Fast response time needed
- ✅ Processing many videos

### Use OpenAI When:
- ✅ Maximum accuracy required
- ✅ Detailed incident analysis needed
- ✅ Processing critical evidence
- ✅ Legal documentation required

## 🛠️ How It Works

1. **Unified Interface**: Both models use the same harassment detection prompts
2. **Automatic Switching**: Change models without code changes
3. **Fallback Support**: Falls back to available model if primary fails
4. **Performance Tracking**: Times and logs both models equally

## 📊 Updated Features

### Live Camera Monitor
- Shows active AI model on startup
- Real-time model performance display
- Same controls work with both models

### Video Analyzer
- Displays which model is analyzing
- Consistent logging format for both models
- Evidence collection works identically

### Main Menu
- New option: "4. Select AI model"
- Easy access to model switching
- Updated help documentation

## 🔍 Testing Both Models

Run the test setup to verify both models:
```bash
python test_setup.py
```

This will check:
- ✅ OpenAI package installation
- ✅ API key configuration  
- ✅ Model connectivity
- ✅ Integration testing

## 🏃 Quick Start Commands

```bash
# Install OpenAI support
pip install openai==1.3.0

# Select AI model
python select_ai_model.py

# Test everything
python test_setup.py

# Start main application  
python start_here.py
```

## 📝 Model Comparison

| Feature | Gemini | OpenAI |
|---------|--------|--------|
| **Speed** | 1-3 seconds | 2-5 seconds |
| **Cost** | Lower | Higher |
| **Accuracy** | High | Very High |
| **Reasoning** | Good | Excellent |
| **Real-time Use** | ✅ Excellent | ✅ Good |
| **Detailed Analysis** | ✅ Good | ✅ Excellent |

## 🎉 Benefits

- **Flexibility**: Choose the right model for your use case
- **Web Interface**: Manage all settings through modern web UI
- **No Code Changes**: Switch models without touching code
- **Best of Both**: Combine fast monitoring with detailed analysis
- **Cost Control**: Use cheaper model for bulk processing
- **Redundancy**: Fallback if one service is down

## 🚀 Performance Features

### **Parallel Processing**
- ✅ **Ultra-fast analysis** with multi-threading
- ✅ **Configurable** thread count and batch size
- ✅ **Better CPU utilization** for faster results

### **Video Chunking**
- ✅ **3-5x faster** analysis for long videos
- ✅ **Splits videos** into 15s chunks for parallel processing
- ✅ **Overlap protection** ensures no missed incidents
- ✅ **Early alerts** from completed chunks

## 💰 Cost-Saving Auto-Stop Feature

**NEW**: The system now automatically stops AI analysis when harassment is detected to save costs!

### Video Analysis Auto-Stop
- ✅ **Stops immediately** when alert detected
- ✅ **User choice** to continue or stop (default: stop)
- ✅ **Shows progress** and estimated savings
- ✅ **60-90% cost reduction** typical

### Live Camera Auto-Pause
- ✅ **Visual pause indicators** on screen
- ✅ **Press 'c' to continue** monitoring
- ✅ **Evidence captured** before pausing
- ✅ **Smart cost management**

### Example Savings
```
🚨 HARASSMENT ALERT DETECTED! 🚨
💰 Stopping analysis to save AI costs...
📊 Current progress: 15.3% (65/420 frames)
Continue analysis? (y/n/s): [n]
💰 Estimated time saved: 156.7 seconds
```

The system now gives you complete control over AI model selection, intelligent cost management, and ultra-fast parallel processing! 🚀
