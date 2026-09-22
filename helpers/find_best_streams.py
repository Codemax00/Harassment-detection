"""
Finds and verifies working public live street/pedestrian CCTV streams from Live-Environment-Streams.
"""

import json
import cv2

with open("storage/live_environment_streams.json", "r", encoding="utf-8") as f:
    data = json.load(f)

features = data.get("features", [])

candidates = []
for f in features:
    p = f.get("properties", {})
    if (
        p.get("status") == "active"
        and p.get("url_type") == "hls"
        and not p.get("source_url_requires")
        and p.get("url")
    ):
        name = p.get("name", "").lower()
        env = p.get("environment", "").lower()
        if any(w in name or w in env for w in ["street", "square", "plaza", "traffic", "city", "urban"]):
            candidates.append(p)

print(f"Total candidate street/urban streams: {len(candidates)}")

# Test the first 10 candidate streams to find one that connects immediately
working_streams = []
for p in candidates[:15]:
    url = p.get("url")
    name = p.get("name", "Unknown").encode("ascii", "replace").decode("ascii")
    cap = cv2.VideoCapture(url)
    if cap.isOpened():
        ret, frame = cap.read()
        cap.release()
        if ret and frame is not None:
            print(f"SUCCESS: [{p.get('environment')}] {name}")
            print(f"URL: {url}")
            working_streams.append(p)
            if len(working_streams) >= 3:
                break
    else:
        cap.release()

if working_streams:
    print(f"\nRecommended live CCTV feed: {working_streams[0]['url']}")
