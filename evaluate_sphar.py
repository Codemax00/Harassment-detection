"""
SPHAR Dataset Benchmark Runner for Guardian Matrix.
Evaluates physical harassment, assault, and violence detection against
real-world surveillance footage from AlexanderMelde/SPHAR-Dataset.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Tuple, Any
import numpy as np

# Ensure project root is in sys.path
sys.path.insert(0, os.path.dirname(__file__))

from src.evaluation.sphar_loader import prepare_sphar_benchmark
from helpers.guardian_matrix_adapter import GuardianMatrixVideoAnalyzer
from src.policy.safety_policy import PolicyState


# Mapping between SPHAR action classes and Harassment/Violence Risk ground truth
# Attack/Violence/Incident: hitting, kicking, falling, panicking
# Benign/Normal: walking, neutral, sitting
HARASSMENT_RISK_CLASSES = {"hitting", "kicking", "falling", "panicking"}
BENIGN_CLASSES = {"walking", "neutral", "sitting"}


def evaluate_single_video(
    analyzer: GuardianMatrixVideoAnalyzer,
    video_path: str,
    max_frames: int = 120
) -> Dict[str, Any]:
    t0 = time.time()
    res = analyzer.analyze_video(video_path=video_path, max_frames=max_frames)
    elapsed = time.time() - t0

    # Aggregate frame-level predictions into video-level prediction
    # If any ALERT or high-confidence REVIEW was triggered on a risk pattern:
    alerts = res.get("alerts", [])
    reviews = res.get("reviews", [])
    total_frames = res.get("total_frames", 0)

    # Count pattern occurrences across frames
    pattern_counts: Dict[str, int] = {}
    for sample in res.get("frame_results_sample", []):
        pat = sample.get("pattern", "NORMAL_ACTIVITY")
        pattern_counts[pat] = pattern_counts.get(pat, 0) + 1

    has_alert = len(alerts) > 0
    has_review = len(reviews) > 0

    # Determine dominant detected action pattern
    risk_alerts = [a for a in alerts if a.get("pattern") in (
        "AGGRESSIVE_MOTION_PATTERN", "REPEATED_CONTACT_PATTERN", "FALL_PATTERN", "RESTRAINT_PATTERN", "PURSUIT_PATTERN"
    )]
    risk_reviews = [r for r in reviews if r.get("pattern") in (
        "AGGRESSIVE_MOTION_PATTERN", "REPEATED_CONTACT_PATTERN", "FALL_PATTERN", "RESTRAINT_PATTERN", "PURSUIT_PATTERN"
    )]

    is_predicted_risk = len(risk_alerts) >= 2 or len(risk_reviews) >= 4 or (len(risk_alerts) >= 1 and total_frames < 30)

    top_pattern = "NORMAL_ACTIVITY"
    if risk_alerts:
        top_pattern = risk_alerts[0].get("pattern", "AGGRESSIVE_MOTION_PATTERN")
    elif risk_reviews:
        top_pattern = risk_reviews[0].get("pattern", "AGGRESSIVE_MOTION_PATTERN")
    elif alerts:
        top_pattern = alerts[0].get("pattern", "ALERT")
    elif reviews:
        top_pattern = reviews[0].get("pattern", "REVIEW")

    return {
        "video_path": video_path,
        "total_frames": total_frames,
        "duration_seconds": res.get("duration_seconds", 0.0),
        "processed_fps": res.get("processed_fps", 0.0),
        "alerts_count": len(alerts),
        "risk_alerts_count": len(risk_alerts),
        "review_count": len(reviews),
        "is_predicted_risk": is_predicted_risk,
        "dominant_pattern": top_pattern,
        "elapsed_seconds": round(elapsed, 2)
    }


def run_benchmark(
    samples_per_class: int = 5,
    max_frames_per_video: int = 120,
    output_report: str = "sphar_benchmark_results.json"
) -> Dict[str, Any]:
    print(f"\n=======================================================")
    print(f"     SPHAR SURVEILLANCE BENCHMARK EVALUATION           ")
    print(f"=======================================================")
    print(f"Samples per class: {samples_per_class}")
    print(f"Max frames/video : {max_frames_per_video}")

    class_configs = {
        "hitting": samples_per_class,
        "kicking": samples_per_class,
        "falling": samples_per_class,
        "walking": samples_per_class,
        "neutral": samples_per_class,
        "sitting": max(2, samples_per_class - 1)
    }

    manifest = prepare_sphar_benchmark(samples_per_class=class_configs)
    analyzer = GuardianMatrixVideoAnalyzer()

    results: List[Dict[str, Any]] = []
    total_monitoring_seconds = 0.0

    print(f"\nRunning Guardian Matrix inference on {len(manifest)} test clips...")
    for idx, (vpath, gt_class) in enumerate(manifest, 1):
        print(f"[{idx:02d}/{len(manifest):02d}] Evaluating {gt_class:<10} | {os.path.basename(vpath)}...", end=" ", flush=True)
        try:
            v_res = evaluate_single_video(analyzer, vpath, max_frames=max_frames_per_video)
            v_res["ground_truth_class"] = gt_class
            is_gt_risk = gt_class in HARASSMENT_RISK_CLASSES
            v_res["is_gt_risk"] = is_gt_risk
            total_monitoring_seconds += v_res["duration_seconds"]

            status_str = "RISK ALERT" if v_res["is_predicted_risk"] else "NORMAL"
            match = (is_gt_risk == v_res["is_predicted_risk"])
            print(f"-> Pred: {status_str:<10} ({v_res['dominant_pattern']}) {'[CORRECT]' if match else '[MISMATCH]'}")
            results.append(v_res)
        except Exception as e:
            print(f"-> ERROR: {e}")

    # Compute binary metrics (Risk vs Benign)
    tp = sum(1 for r in results if r["is_gt_risk"] and r["is_predicted_risk"])
    fp = sum(1 for r in results if not r["is_gt_risk"] and r["is_predicted_risk"])
    tn = sum(1 for r in results if not r["is_gt_risk"] and not r["is_predicted_risk"])
    fn = sum(1 for r in results if r["is_gt_risk"] and not r["is_predicted_risk"])

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    accuracy = (tp + tn) / max(1, len(results))

    total_hours = max(1e-4, total_monitoring_seconds / 3600.0)
    false_alarms_per_hour = fp / total_hours

    # Per-class recall/accuracy
    per_class_stats: Dict[str, Dict[str, Any]] = {}
    for cls in class_configs:
        cls_items = [r for r in results if r["ground_truth_class"] == cls]
        if not cls_items:
            continue
        is_risk_cls = cls in HARASSMENT_RISK_CLASSES
        if is_risk_cls:
            correct = sum(1 for r in cls_items if r["is_predicted_risk"])
        else:
            correct = sum(1 for r in cls_items if not r["is_predicted_risk"])
        acc = correct / len(cls_items)
        per_class_stats[cls] = {
            "total": len(cls_items),
            "correct": correct,
            "accuracy": round(acc, 3),
            "dominant_patterns": [r["dominant_pattern"] for r in cls_items]
        }

    report = {
        "timestamp": time.time(),
        "total_clips": len(results),
        "total_monitoring_seconds": round(total_monitoring_seconds, 2),
        "total_monitoring_hours": round(total_hours, 4),
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
        "accuracy": round(accuracy, 4),
        "false_alarms_per_hour": round(false_alarms_per_hour, 2),
        "per_class": per_class_stats,
        "details": results
    }

    print("\n" + "=" * 65)
    print("            SPHAR BENCHMARK EVALUATION SUMMARY                 ")
    print("=" * 65)
    print(f"Total Test Clips       : {len(results)}")
    print(f"Accuracy               : {accuracy * 100:.1f}%")
    print(f"Precision              : {precision:.3f}")
    print(f"Recall (Attack Caught) : {recall:.3f} (TP={tp}, FN={fn})")
    print(f"F1-Score               : {f1:.3f}")
    print(f"False Positives        : {fp} (TN={tn})")
    print(f"False Alarms / Hour    : {false_alarms_per_hour:.2f}")
    print("-" * 65)
    print(f"{'Class':<12} | {'Total':<6} | {'Correct':<8} | {'Accuracy':<10}")
    print("-" * 65)
    for c, stat in per_class_stats.items():
        print(f"{c:<12} | {stat['total']:<6} | {stat['correct']:<8} | {stat['accuracy']*100:6.1f}%")
    print("=" * 65 + "\n")

    with open(output_report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"Detailed benchmark report saved to: {output_report}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPHAR Benchmark Runner")
    parser.add_argument("--samples", type=int, default=4, help="Samples per action class")
    parser.add_argument("--max_frames", type=int, default=100, help="Max frames per video")
    parser.add_argument("--output", type=str, default="sphar_benchmark_results.json", help="Output JSON path")
    args = parser.parse_args()

    run_benchmark(
        samples_per_class=args.samples,
        max_frames_per_video=args.max_frames,
        output_report=args.output
    )
