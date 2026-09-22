"""
Guardian Matrix - Live Environment Streams Loader
Integrates with william-ricchiuti/Live-Environment-Streams (5,000+ real live streams globally)
Allows downloading, filtering, and sampling live public CCTV/HLS/RTSP feeds for realistic testing.
"""

import os
import json
import urllib.request
from pathlib import Path
from typing import List, Dict, Any, Optional

STREAMS_GEOJSON_URL = "https://raw.githubusercontent.com/william-ricchiuti/Live-Environment-Streams/main/streams.geojson"
CACHE_PATH = Path(__file__).resolve().parent.parent / "storage" / "live_environment_streams.json"


def download_or_load_streams(force_refresh: bool = False) -> List[Dict[str, Any]]:
    """Downloads or reads cached streams.geojson from Live-Environment-Streams."""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if CACHE_PATH.exists() and not force_refresh:
        try:
            with open(CACHE_PATH, "r", encoding="utf-8") as f:
                data = json.load(f)
                return data.get("features", [])
        except Exception:
            pass

    print("[STREAMS] Fetching streams dataset from william-ricchiuti/Live-Environment-Streams...")
    try:
        req = urllib.request.Request(STREAMS_GEOJSON_URL, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            with open(CACHE_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f)
            print(f"[STREAMS] Cached {len(data.get('features', []))} streams to {CACHE_PATH}")
            return data.get("features", [])
    except Exception as e:
        print(f"[WARNING] Could not fetch live streams catalog: {e}")
        return []


def get_urban_cctv_streams(limit: int = 15) -> List[Dict[str, Any]]:
    """Filters active urban, street, and public square camera streams suitable for CCTV AI testing."""
    features = download_or_load_streams()
    cctv_streams = []

    for f in features:
        props = f.get("properties", {})
        # Target active direct HLS/RTSP streams with no auth requirements
        if (
            props.get("status") == "active"
            and props.get("url_type") in ("hls", "rtsp")
            and not props.get("source_url_requires")
            and props.get("url")
        ):
            cctv_streams.append({
                "name": props.get("name", "Unknown Camera"),
                "environment": props.get("environment", "urban"),
                "country": props.get("country", "Global"),
                "resolution": props.get("resolution", "1080p"),
                "url": props.get("url"),
                "url_type": props.get("url_type"),
            })
            if len(cctv_streams) >= limit:
                break

    return cctv_streams


if __name__ == "__main__":
    streams = get_urban_cctv_streams(limit=10)
    print(f"\nFound {len(streams)} active CCTV / outdoor streams:")
    for i, s in enumerate(streams):
        safe_name = s['name'].encode('ascii', 'replace').decode('ascii')
        print(f"{i+1}. [{s['country']}] {safe_name} ({s['environment']}) -> {s['url']}")
