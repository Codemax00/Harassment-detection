---
marp: true
theme: uncover
backgroundColor: #f0f4f8
color: #333
headingDivider: 1
style: |
  h1 {
    font-size: 2.8em;
    color: #005A9C;
    text-shadow: 2px 2px 4px #ccc;
  }
  h2 {
    color: #005A9C;
    border-bottom: 2px solid #007bff;
    padding-bottom: 10px;
  }
  strong {
    color: #007bff;
  }
  .slide-content {
    display: flex;
    flex-direction: column;
    justify-content: center;
    align-items: center;
    height: 80%;
  }
  .columns {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 20px;
    width: 100%;
  }
  .columns ul {
    margin-top: 0;
  }
  img {
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(0,0,0,0.15);
  }

---

# **🛡️ Safety Detector**
### AI-Powered Harassment Detection System

**Project Overview Presentation**
*September 2025*

---

# **1. Project Vision & Mission**

**Vision:** To create safer public and private spaces through intelligent, real-time monitoring.

**Mission:** To automatically detect potential harassment incidents using AI, provide immediate alerts, and generate evidence for effective intervention. Our goal is to offer a proactive safety solution that is both powerful and privacy-conscious.

---

# **2. The Problem We Address**

*   **Inefficient Monitoring:** Manual surveillance is expensive, not scalable, and prone to human error.
*   **Delayed Response:** Incidents often go unnoticed until it's too late.
*   **Lack of Evidence:** Immediate, unbiased evidence is crucial for intervention but is often unavailable.
*   **Privacy Concerns:** Traditional surveillance can be overly intrusive.

**Our system is designed to solve these challenges.**

---

# **3. Our Solution: An AI Watchdog**

The **Safety Detector** is a smart surveillance system that:
*   Analyzes **live camera feeds** and pre-recorded videos.
*   Uses **Computer Vision** to detect people and **AI** to analyze their behavior.
*   Provides **real-time alerts** through a modern, accessible web interface.
*   Respects privacy by focusing on **behavioral patterns**, not facial recognition.

---

# **4. System Architecture**

A simple, powerful, and modular design.

<div class="columns">
<div>

**Input Layer**
*   Live Camera (USB, Built-in, ESP32)
*   Uploaded Video Files

</div>
<div>

**Processing Layer**
*   **YOLOv8:** For high-accuracy person detection.
*   **Gemini/GPT-4 AI:** For advanced behavioral and harassment analysis.
*   Parallel Processing & Video Chunking.

</div>
</div>

**Output Layer**
*   **Web Interface:** Real-time dashboard, alerts, and evidence management.

---

# **5. Technology Stack**

Built with robust and industry-standard technologies.

<div class="columns">
<div>

**Backend & AI**
*   **Python 3.11+**
*   **Flask:** For the web server and API.
*   **OpenCV:** For all computer vision tasks.
*   **YOLOv8:** State-of-the-art object detection.
*   **Google Gemini & OpenAI GPT-4:** For AI-powered reasoning.

</div>
<div>

**Frontend**
*   **HTML5 & CSS3**
*   **JavaScript (ES6+)**
*   **Bootstrap 5:** For a modern, responsive UI.
*   **Chart.js:** For data visualization (future).

</div>
</div>

---

# **6. Key Features**

*   **Real-time Monitoring:** Analyze live camera feeds with minimal latency.
*   **Multi-AI Support:** Flexibly switch between **Google Gemini** and **OpenAI GPT-4**.
*   **High Performance:** **3-5x faster** analysis with parallel processing and video chunking.
*   **Cost-Saving Engine:** Smart **auto-stop** on alert detection reduces API costs by up to 90%.
*   **Modern Web UI:** A responsive and user-friendly dashboard for monitoring and configuration.
*   **Camera Diagnostics:** Built-in tools to troubleshoot camera connectivity.

---

# **7. Web Interface Showcase**

An intuitive dashboard for complete control and monitoring.

*   **Live Feed:** Real-time video stream with AI overlays.
*   **Real-time Stats:** Live updates for FPS, people count, and alerts.
*   **Camera Management:** Select from available cameras, including ESP32 modules.
*   **Instant Alerts:** Immediate notifications directly in the UI when potential harassment is detected.
*   **Configuration:** Manage all system and AI settings from a simple web form.

---

# **8. Scalability & Future Enhancements**

Engineered for growth and future challenges.

<div class="columns">
<div>

**Current Performance**
*   **Optimized Streaming:** 15 FPS for smooth performance.
*   **Efficient AI:** Smart analysis reduces unnecessary processing.
*   **Fast Detection:** ~0.05s per frame for person detection.

</div>
<div>

**Future Enhancements**
*   **Multi-Camera Coordination:** Analyze feeds from multiple cameras simultaneously.
*   **Audio Analysis:** Integrate audio processing to detect verbal aggression.
*   **Mobile Application:** Develop a mobile app for on-the-go alerts and monitoring.
*   **Advanced Analytics:** A dashboard with historical data, heatmaps, and trend analysis.

</div>
</div>

---

# **9. Conclusion**

**Conclusion:** The Safety Detector is a robust, privacy-first AI system that provides a scalable and efficient solution for enhancing safety in various environments.

### **Thank You & Q&A**
