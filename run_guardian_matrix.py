"""
Guardian Matrix Research CLI & Benchmark Runner.
Usage:
    python run_guardian_matrix.py --video <video_path>
    python run_guardian_matrix.py --benchmark
    python run_guardian_matrix.py --ablation
"""

import argparse
import json
import os
import sys
import time
from typing import Optional

# Add current dir to path
sys.path.insert(0, os.path.dirname(__file__))

from helpers.guardian_matrix_adapter import GuardianMatrixVideoAnalyzer
from src.evaluation.benchmark import AblationStudyRunner, EvaluationSuite


def run_video_analysis(video_path: str, max_frames: int = 150):
    print(f"\n=======================================================")
    print(f"   GUARDIAN MATRIX — Research Video Analysis Pipeline   ")
    print(f"=======================================================")
    print(f"Input Video: {video_path}")
    print(f"Max Frames: {max_frames}")

    analyzer = GuardianMatrixVideoAnalyzer()
    out_video = os.path.join("outputs", "annotated_" + os.path.basename(video_path))
    os.makedirs("outputs", exist_ok=True)

    def progress(f, total, t, summary):
        if f % 15 == 0:
            act_str = summary.get('action', summary.get('pattern', 'UNKNOWN'))
            cat_str = summary.get('category', 'NORMAL')
            state_str = summary.get('incident_state', summary.get('policy_state', 'NORMAL'))
            risk_val = summary.get('risk_score', summary.get('confidence', 0.0))
            people = summary.get('people_count', 0)
            lat = summary.get('processing_time_ms', 0.0)
            print(f"  Frame {f:04d}/{total:04d} ({t:05.1f}s) | "
                  f"State: {state_str:<15} | "
                  f"Cat: {cat_str:<22} | "
                  f"Act: {act_str:<10} | "
                  f"Risk: {risk_val:.2f} | "
                  f"People: {people} | "
                  f"{lat:5.1f}ms")

    results = analyzer.analyze_video(
        video_path=video_path,
        output_annotated_path=out_video,
        progress_callback=progress,
        max_frames=max_frames
    )

    print("\n-------------------------------------------------------")
    print("                 ANALYSIS SUMMARY                      ")
    print("-------------------------------------------------------")
    print(f"Processed Frames   : {results['total_frames']}")
    print(f"Average FPS        : {results['processed_fps']}")
    print(f"Dominant Category  : {results['predicted_category']}")
    print(f"Incident Episodes  : {results.get('incident_episodes_count', 0)}")
    print(f"Safety Alerts      : {results.get('safety_incidents_count', 0)}")
    print(f"Aggressive Alerts  : {results.get('aggressive_incidents_count', 0)}")
    print(f"Human Reviews Req  : {results['review_count']}")
    print(f"Annotated Video    : {results['annotated_video']}")
    if results.get('saved_bundles'):
        for b in results['saved_bundles']:
            print(f"Incident Saved     : {b}")
    print("=======================================================\n")
    return results


def run_ablation_study():
    print(f"\n=======================================================")
    print(f"   GUARDIAN MATRIX — Scientific Ablation Studies       ")
    print(f"   (Upgrade_Plan.md Section 25)                        ")
    print(f"=======================================================")
    runner = AblationStudyRunner()
    results = runner.run_all_ablations()

    print(f"{'EXP ID':<8} | {'F1-Score':<8} | {'False Alarms/Hr':<16} | {'Latency':<10} | {'Description'}")
    print("-" * 80)
    for r in results:
        print(f"{r.experiment_id:<8} | {r.f1_score:<8.2f} | {r.false_alarms_per_hour:<16.1f} | {r.mean_latency_ms:<7.1f} ms | {r.description}")
    print("=" * 80 + "\n")
    return results


def run_live_stream(source, max_frames: Optional[int] = None, show_gui: bool = False, record_path: Optional[str] = None):
    import cv2
    from helpers.guardian_matrix_adapter import GuardianMatrixLiveStreamManager

    print(f"\n=======================================================")
    print(f"   GUARDIAN MATRIX — Real-Time Live Stream Engine      ")
    print(f"=======================================================")
    print(f"Stream Source  : {source}")
    print(f"GUI Display    : {show_gui} (Press 'q' in window to exit)")
    print(f"Record Path    : {record_path or 'None'}")
    print(f"Max Frames     : {max_frames or 'Unlimited'}")
    print("-------------------------------------------------------")

    manager = GuardianMatrixLiveStreamManager()
    manager.start_stream(source=source)

    writer = None
    frame_count = 0

    try:
        while True:
            time.sleep(0.03)  # Sample poll
            if not manager.processor or not manager.processor.is_processing:
                break

            status = manager.get_latest_status()
            if not status.get("active"):
                continue

            current_fid = status.get("frame_id", 0)
            if current_fid > frame_count:
                frame_count = current_fid
                if frame_count % 15 == 0:
                    print(f"  Live Frame {frame_count:05d} | "
                          f"People: {status['people_detected']} | "
                          f"Pattern: {status['pattern']:<25} | "
                          f"State: {status['policy_state']:<8} | "
                          f"Conf: {status['confidence']:.2f} | "
                          f"{status['processing_time_ms']:5.1f}ms")

                # If recording requested
                if record_path:
                    latest = manager.processor.latest_result
                    if latest and latest.annotated_frame is not None:
                        if writer is None:
                            os.makedirs(os.path.dirname(os.path.abspath(record_path)) or ".", exist_ok=True)
                            h, w = latest.annotated_frame.shape[:2]
                            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                            writer = cv2.VideoWriter(record_path, fourcc, 15.0, (w, h))
                        writer.write(latest.annotated_frame)

                # If GUI requested
                if show_gui:
                    latest = manager.processor.latest_result
                    if latest and latest.annotated_frame is not None:
                        cv2.imshow("Guardian Matrix - Live Stream", latest.annotated_frame)
                        key = cv2.waitKey(1) & 0xFF
                        if key == ord('q'):
                            print("Live stream stopped by user ('q').")
                            break

                if max_frames and frame_count >= max_frames:
                    print(f"Reached max frames ({max_frames}). Stopping live stream.")
                    break

    except KeyboardInterrupt:
        print("\nStopping live stream via KeyboardInterrupt...")
    finally:
        if writer:
            writer.release()
        if show_gui:
            cv2.destroyAllWindows()
        manager.stop_stream()
        print("Live stream stopped successfully.\n=======================================================\n")


def main():
    parser = argparse.ArgumentParser(description="Guardian Matrix Research CLI")
    parser.add_argument("--video", type=str, help="Path to input video file")
    parser.add_argument("--stream", type=str, help="Live stream source: camera index (e.g. 0), RTSP url, HTTP url, or video file")
    parser.add_argument("--gui", action="store_true", help="Display live OpenCV GUI window")
    parser.add_argument("--record", type=str, help="Record annotated live stream to output MP4 path")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum frames to process")
    parser.add_argument("--ablation", action="store_true", help="Run 6-stage scientific ablation benchmark")

    args = parser.parse_args()

    if args.ablation:
        run_ablation_study()
    elif args.stream is not None:
        src = int(args.stream) if args.stream.isdigit() else args.stream
        run_live_stream(source=src, max_frames=args.max_frames, show_gui=args.gui, record_path=args.record)
    elif args.video:
        run_video_analysis(args.video, max_frames=args.max_frames or 150)
    else:
        # Default: run on sample video if present, or run ablation
        default_video = os.path.join("test_videos", "WhatsApp Video 2025-09-12 at 10.26.50 AM.mp4")
        if os.path.exists(default_video):
            run_video_analysis(default_video, max_frames=args.max_frames or 150)
        else:
            run_ablation_study()


if __name__ == "__main__":
    main()
