# 📱 Android Phone Live RTSP CCTV Setup Guide for Guardian Matrix

This guide walks you step-by-step through turning your Android smartphone into a simulated real-time IP CCTV camera that streams directly to **Guardian Matrix**.

---

## 🏗️ Architecture Overview

```
┌─────────────────────────┐
│   Android Phone Camera  │
│   (RTSP Server App)     │
└────────────┬────────────┘
             │
             │ RTSP stream over Wi-Fi (e.g. rtsp://192.168.1.50:8554/live)
             ▼
┌─────────────────────────┐
│     Guardian Matrix     │
│    (main.py / stream)   │
│  - Bounded Buffer (0s)  │
│  - Auto-reconnect       │
│  - Person Tracking      │
│  - Pose Estimation      │
│  - Temporal AI Risk     │
│  - Auto-Incident Video  │
└─────────────────────────┘
```

> [!IMPORTANT]
> **Network Prerequisite**: Your Android phone and your computer running Guardian Matrix **MUST be connected to the same Wi-Fi network** (or phone Wi-Fi hotspot connected to your PC).

---

## 📲 Step 1: Install an RTSP Camera App on Android

Choose either **Option A** (Recommended Open Source) or **Option B** (Quick Install from Google Play):

### Option A: Open-Source RTSP CCTV App (Zektopic)
1. Visit the repository: [Zektopic/RSTP-CCTV-App](https://github.com/Zektopic/RSTP-CCTV-App)
2. Download the latest `.apk` from the Releases section or build it in Android Studio.
3. Install the APK on your Android device (allow *Install from Unknown Sources* if prompted).
4. Launch the app and grant Camera and Microphone permissions.
5. Tap **Start RTSP Server**.
6. The app will display your stream URL, for example:
   ```
   rtsp://192.168.1.105:8554/live
   ```

### Option B: Google Play Store Alternative ("IP Webcam" / "RTSP Camera")
If you prefer installing directly from Google Play Store:
1. **App**: Search for **IP Webcam** (by Pavel Khlebovich) or **RTSP Camera**.
2. **Settings in IP Webcam**:
   - Scroll down and open **Video preferences**: set resolution to `1280x720` or `640x480` (for high FPS, low latency).
   - Under **Connection settings** or **RTSP**: enable RTSP server (default port is `8080` for HTTP or `8554` for RTSP).
3. Tap **Start server** at the bottom of the screen.
4. Note down the IP address shown on your screen (e.g., `http://192.168.1.105:8080` or `rtsp://192.168.1.105:8080/h264_pcm.sdp`).

---

## 🔍 Step 2: Verify Network Reachability from Your Computer

Before running the AI pipeline, verify that your PC can communicate with your phone. Open PowerShell or Command Prompt on your computer:

```powershell
# Replace with your phone's IP address
ping 192.168.1.105
```

If you receive replies (`Reply from 192.168.1.105: bytes=32 time=...`), the network path is open!

---

## ⚙️ Step 3: Configure Guardian Matrix

You can set your phone's RTSP URL in two easy ways **without changing any code**:

### Method 1: In `.env` (Persistent)
Open `.env` in the root of `Harassment-detection` and update:
```dotenv
CCTV_RTSP_URL=rtsp://192.168.1.105:8554/live
```
Save the file.

### Method 2: Via Command Line `--url` Flag (Quick Override)
You can directly pass the URL when launching:
```powershell
.venv\Scripts\python.exe main.py --source rtsp --url "rtsp://192.168.1.105:8554/live"
```

---

## 🚀 Step 4: Run Guardian Matrix Live AI Detection

Run the live AI monitor with GUI HUD:
```powershell
.venv\Scripts\python.exe main.py --source rtsp
```

Or pass your phone's RTSP URL directly:
```powershell
.venv\Scripts\python.exe main.py --source rtsp --url "rtsp://192.168.1.105:8554/live"
```

### What You Will See on Your Screen:
1. **Live CCTV HUD**:
   - 🟢 **Bounding boxes** with persistent IDs (`Person 1`, `Person 2`).
   - 🟡 **Pose Skeleton Overlay**: Real-time joints (shoulders, elbows, wrists, knees, ankles) tracking body movement.
   - 🔵 **Pairwise Proximity & Interaction Lines**: Displays real-time pixel distance and interaction duration.
   - 📊 **Telemetry HUD**: Stream status (`CONNECTED`), Stream FPS, AI FPS, Latency (ms), Dropped Frames, CPU/RAM utilization.
   - 🚨 **Risk Level Banner**: Real-time evaluation (`LOW`, `MEDIUM`, `HIGH`, `CRITICAL`) with detected evidence (e.g., `rapid approach`, `pursuit pattern`).
2. **Automatic Incident Evidence Clip**:
   - If an event is confirmed for 10 consecutive frames, an alert triggers.
   - A clip consisting of **5s before the event + event + 5s after the event** is saved to `incidents/YYYY-MM-DD_HH-MM-SS/event.mp4` with full `metadata.json`.

---

## 🧪 Step 5: Test Camera Disconnect & Auto-Recovery (Requirement 16 & 24)

1. While the system is actively monitoring, **turn off Wi-Fi or close the app on your phone**.
2. **Observe Guardian Matrix**:
   - The HUD immediately updates: `Camera: OFFLINE`.
   - The system automatically triggers exponential backoff reconnection attempts (`1s`, `2s`, `4s`, `8s`, up to `15s`) without crashing.
3. **Turn your phone camera / Wi-Fi back on**:
   - Guardian Matrix automatically detects the restored feed, reconnects, and resumes real-time AI detection within seconds!

---

## 🛠️ Troubleshooting & Tips

| Problem | Cause | Solution |
| :--- | :--- | :--- |
| **Ping fails / Request timed out** | Router AP Isolation or different Wi-Fi | Ensure PC and Phone are on the same Wi-Fi. Turn on your phone's Mobile Hotspot and connect your PC to it if your home router blocks LAN traffic. |
| **Stream stays OFFLINE** | Port blocked by Windows Firewall or wrong RTSP path | Check if the port (e.g. `8554`) is open. Verify the exact path shown in the Android app (e.g., `/live`, `/stream`, `/h264_pcm.sdp`). |
| **High Latency (> 1 second)** | High video resolution on phone | Lower the resolution in the phone app to `720p` (1280x720) or `480p`. Guardian Matrix uses a bounded buffer `maxsize=1` to drop stale frames and keep latency to ~100ms. |
