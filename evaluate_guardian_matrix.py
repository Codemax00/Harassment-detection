"""
Guardian Matrix Final Benchmark & Validation Evaluator.
Executes the upgraded multi-stage architecture across all surveillance video categories:
- Normal Activities / Hard Negatives: walking, crouching, standing, sitting, bending
- Safety Incidents: falling, collapses, faint
- Aggressive Interactions: hitting, kicking

Computes:
1. Multi-class confusion matrix (NORMAL, SAFETY, AGGRESSIVE)
2. Precision, Recall, Specificity, F1, FPR, FNR per category
3. Per-action breakdown (FALL, STRIKE, KICK, WALKING, etc.)
4. Operational metrics: FPS, P95 latency, False Alerts/Hour, Detection Latency
5. Forensic evidence bundle generation auditing
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Any

# Ensure project root in sys.path
root_dir = os.path.abspath(os.path.dirname(__file__))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from helpers.guardian_matrix_adapter import GuardianMatrixVideoAnalyzer
from src.taxonomy.event_taxonomy import EventCategory, ActionType, OperationalUrgency, EvidenceState


# Ground truth mapping for SPHAR categories
GT_CATEGORY_MAP = {
    "hitting": (EventCategory.AGGRESSIVE_INTERACTION, ActionType.STRIKE),
    "kicking": (EventCategory.AGGRESSIVE_INTERACTION, ActionType.KICK),
    "falling": (EventCategory.SAFETY_INCIDENT, ActionType.FALL),
    "walking": (EventCategory.NORMAL_ACTIVITY, ActionType.WALKING),
    "neutral": (EventCategory.NORMAL_ACTIVITY, ActionType.STANDING),
    "sitting": (EventCategory.NORMAL_ACTIVITY, ActionType.SITTING)
}


def evaluate_guardian_matrix(
    data_dir: str = "test_videos/sphar",
    output_results_path: str = "guardian_matrix_validation_results.json",
    max_frames_per_clip: Optional[int] = 80
) -> Dict[str, Any]:
    print("=" * 80)
    print(" GUARDIAN MATRIX: FINAL RESEARCH-GRADE VALIDATION BENCHMARK ")
    print("=" * 80)
    print(f"Dataset root: {data_dir}")
    print(f"Max frames per clip: {max_frames_per_clip or 'Full Video'}\n")

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    analyzer = GuardianMatrixVideoAnalyzer()

    # Discover all video clips grouped by class folder
    video_records = []
    classes = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    classes.sort()

    for cls_name in classes:
        cls_dir = os.path.join(data_dir, cls_name)
        v_files = [f for f in os.listdir(cls_dir) if f.endswith(('.mp4', '.avi'))]
        v_files.sort()
        for vf in v_files:
            video_records.append({
                "path": os.path.join(cls_dir, vf),
                "class_name": cls_name,
                "file_name": vf
            })

    print(f"Discovered {len(video_records)} surveillance clips across {len(classes)} categories.\n")

    # Metrics storage
    total_monitoring_seconds = 0.0
    total_frames = 0
    fps_measurements = []
    latencies = []
    detection_latencies = []
    clip_details = []

    # 3x3 Confusion Matrix: rows = Ground Truth, cols = Prediction
    # Index 0: NORMAL_ACTIVITY, 1: SAFETY_INCIDENT, 2: AGGRESSIVE_INTERACTION
    categories_ordered = [
        EventCategory.NORMAL_ACTIVITY,
        EventCategory.SAFETY_INCIDENT,
        EventCategory.AGGRESSIVE_INTERACTION
    ]
    cat_to_idx = {cat: i for i, cat in enumerate(categories_ordered)}
    conf_matrix = np.zeros((3, 3), dtype=int)

    # Per-action breakdown tracking
    action_stats: Dict[str, Dict[str, int]] = {}

    total_alerts = 0
    false_alerts = 0
    missed_incidents = 0
    all_saved_bundles = []

    t_bench_start = time.time()

    for idx, rec in enumerate(video_records, 1):
        v_path = rec["path"]
        gt_cls = rec["class_name"]
        gt_cat, gt_act = GT_CATEGORY_MAP.get(gt_cls, (EventCategory.NORMAL_ACTIVITY, ActionType.UNKNOWN))
        gt_idx = cat_to_idx[gt_cat]

        print(f"[{idx:02d}/{len(video_records):02d}] Evaluating: {gt_cls}/{rec['file_name']}...", end="", flush=True)

        t_clip_start = time.time()
        res = analyzer.analyze_video(video_path=v_path, max_frames=max_frames_per_clip)
        t_clip_elapsed = time.time() - t_clip_start

        duration = res["duration_seconds"]
        total_monitoring_seconds += duration
        total_frames += res["total_frames"]
        fps_measurements.append(res["processed_fps"])
        if res.get("mean_latency_ms"):
            latencies.append(res["mean_latency_ms"])
        if res.get("saved_bundles"):
            all_saved_bundles.extend(res["saved_bundles"])

        # Determine predicted category
        pred_cat_str = res["predicted_category"]
        if pred_cat_str == EventCategory.AGGRESSIVE_INTERACTION.value:
            pred_cat = EventCategory.AGGRESSIVE_INTERACTION
        elif pred_cat_str == EventCategory.SAFETY_INCIDENT.value:
            pred_cat = EventCategory.SAFETY_INCIDENT
        else:
            pred_cat = EventCategory.NORMAL_ACTIVITY

        pred_idx = cat_to_idx[pred_cat]
        conf_matrix[gt_idx, pred_idx] += 1

        # Track per-action stats
        act_key = gt_cls
        if act_key not in action_stats:
            action_stats[act_key] = {"total": 0, "correct": 0, "predicted_categories": []}
        action_stats[act_key]["total"] += 1
        action_stats[act_key]["predicted_categories"].append(pred_cat.value)
        if pred_cat == gt_cat:
            action_stats[act_key]["correct"] += 1

        # Alerts and Latency tracking
        episodes = res.get("incident_episodes_count", 1 if res.get("active_incidents_count", 0) > 0 else 0)
        has_incident = (episodes > 0)
        total_alerts += episodes

        if gt_cat == EventCategory.NORMAL_ACTIVITY:
            if has_incident:
                false_alerts += episodes
        else:
            # Incident ground truth
            if not has_incident:
                missed_incidents += 1
            else:
                # Find first alert timestamp for detection latency
                if res.get("alerts"):
                    first_alert_t = res["alerts"][0]["timestamp"]
                    detection_latencies.append(first_alert_t)

        status_flag = "[CORRECT]" if (pred_cat == gt_cat) else "[MISMATCH]"
        print(f" {status_flag} Pred: {pred_cat.value} (GT: {gt_cat.value}) | {res['processed_fps']:.1f} FPS")

        clip_details.append({
            "video": rec["file_name"],
            "ground_truth_class": gt_cls,
            "ground_truth_category": gt_cat.value,
            "predicted_category": pred_cat.value,
            "active_incidents": res["active_incidents_count"],
            "review_count": res["review_count"],
            "fps": res["processed_fps"],
            "duration": duration,
            "correct": bool(pred_cat == gt_cat)
        })

    total_bench_time = time.time() - t_bench_start
    total_monitoring_hours = total_monitoring_seconds / 3600.0

    # Calculate operational metrics
    avg_fps = float(np.mean(fps_measurements)) if fps_measurements else 0.0
    median_fps = float(np.median(fps_measurements)) if fps_measurements else 0.0
    mean_lat = float(np.mean(latencies)) if latencies else 0.0
    alerts_per_hour = total_alerts / max(1e-4, total_monitoring_hours)
    false_alerts_per_hour = false_alerts / max(1e-4, total_monitoring_hours)
    missed_per_hour = missed_incidents / max(1e-4, total_monitoring_hours)
    mean_det_latency = float(np.mean(detection_latencies)) if detection_latencies else 0.0

    # Calculate Category-Level Precision, Recall, F1
    category_metrics = {}
    for i, cat in enumerate(categories_ordered):
        tp = conf_matrix[i, i]
        fp = sum(conf_matrix[j, i] for j in range(3) if j != i)
        fn = sum(conf_matrix[i, j] for j in range(3) if j != i)
        tn = sum(conf_matrix[j, k] for j in range(3) for k in range(3) if j != i and k != i)

        prec = tp / max(1, tp + fp)
        rec = tp / max(1, tp + fn)
        f1 = (2 * prec * rec) / max(1e-4, prec + rec)
        spec = tn / max(1, tn + fp)
        fpr = fp / max(1, fp + tn)
        fnr = fn / max(1, fn + tp)

        category_metrics[cat.value] = {
            "TP": int(tp), "FP": int(fp), "TN": int(tn), "FN": int(fn),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "specificity": round(float(spec), 4),
            "false_positive_rate": round(float(fpr), 4),
            "false_negative_rate": round(float(fnr), 4)
        }

    # Attack Detection Specific (Aggression + Safety recall)
    agg_tp = conf_matrix[2, 2]
    agg_fn = conf_matrix[2, 0] + conf_matrix[2, 1]
    attack_recall = agg_tp / max(1, agg_tp + agg_fn)

    safety_tp = conf_matrix[1, 1]
    safety_fn = conf_matrix[1, 0] + conf_matrix[1, 2]
    safety_recall = safety_tp / max(1, safety_tp + safety_fn)

    # Check fall -> aggression leakage
    fall_to_agg_leakage = int(conf_matrix[1, 2])

    # Per-action accuracies
    per_action_summary = {}
    for act, stats in action_stats.items():
        tot = stats["total"]
        corr = stats["correct"]
        per_action_summary[act] = {
            "total": tot,
            "correct": corr,
            "accuracy": round(corr / max(1, tot), 3),
            "predicted_categories": stats["predicted_categories"]
        }

    overall_accuracy = sum(conf_matrix[i, i] for i in range(3)) / max(1, len(video_records))

    benchmark_summary = {
        "timestamp": time.time(),
        "total_clips": len(video_records),
        "total_monitoring_seconds": round(total_monitoring_seconds, 2),
        "total_monitoring_hours": round(total_monitoring_hours, 4),
        "total_frames_processed": total_frames,
        "performance": {
            "average_fps": round(avg_fps, 2),
            "median_fps": round(median_fps, 2),
            "mean_latency_ms": round(mean_lat, 2),
            "detection_latency_seconds": round(mean_det_latency, 2)
        },
        "operational": {
            "total_alerts": total_alerts,
            "false_alerts": false_alerts,
            "missed_incidents": missed_incidents,
            "alerts_per_hour": round(alerts_per_hour, 2),
            "false_alerts_per_hour": round(false_alerts_per_hour, 2),
            "missed_incidents_per_hour": round(missed_per_hour, 2),
            "saved_evidence_bundles_count": len(all_saved_bundles)
        },
        "targets_achieved": {
            "attack_recall": round(float(attack_recall), 4),
            "attack_recall_target_met": bool(attack_recall >= 0.85),
            "safety_recall": round(float(safety_recall), 4),
            "fall_to_aggression_leakage": fall_to_agg_leakage,
            "zero_fall_to_aggression_leakage": bool(fall_to_agg_leakage == 0),
            "overall_accuracy": round(float(overall_accuracy), 4)
        },
        "confusion_matrix": {
            "labels": [c.value for c in categories_ordered],
            "matrix": conf_matrix.tolist(),
            "description": "Rows = Ground Truth [NORMAL, SAFETY, AGGRESSIVE], Columns = Prediction [NORMAL, SAFETY, AGGRESSIVE]"
        },
        "category_metrics": category_metrics,
        "per_action_metrics": per_action_summary,
        "details": clip_details
    }

    # Save validation results
    with open(output_results_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_summary, f, indent=2)

    # Print Clean Formatted Report
    print("\n" + "=" * 80)
    print(" GUARDIAN MATRIX VALIDATION BENCHMARK RESULTS")
    print("=" * 80)
    print(f"Total Clips: {len(video_records)} | Total Duration: {total_monitoring_seconds:.1f}s ({total_monitoring_hours:.3f} hrs)")
    print(f"Throughput: {avg_fps:.1f} FPS (Median: {median_fps:.1f} FPS) | Mean Latency: {mean_lat:.1f} ms")
    print(f"Detection Latency: {mean_det_latency:.2f} seconds\n")

    print("-" * 80)
    print(" 3x3 CONFUSION MATRIX")
    print("-" * 80)
    print(f"{'':>25} | {'PRED NORMAL':>12} | {'PRED SAFETY':>12} | {'PRED AGGRESSIVE':>16}")
    print(f"{'GT NORMAL_ACTIVITY':>25} | {conf_matrix[0, 0]:>12} | {conf_matrix[0, 1]:>12} | {conf_matrix[0, 2]:>16}")
    print(f"{'GT SAFETY_INCIDENT':>25} | {conf_matrix[1, 0]:>12} | {conf_matrix[1, 1]:>12} | {conf_matrix[1, 2]:>16}")
    print(f"{'GT AGGRESSIVE_INTERACTION':>25} | {conf_matrix[2, 0]:>12} | {conf_matrix[2, 1]:>12} | {conf_matrix[2, 2]:>16}\n")

    print("-" * 80)
    print(" CATEGORY-LEVEL CLASSIFICATION METRICS")
    print("-" * 80)
    print(f"{'Category':<24} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'Spec':>6} | {'FPR':>6} | {'FNR':>6}")
    for cat_name, m in category_metrics.items():
        print(f"{cat_name:<24} | {m['precision']:>6.3f} | {m['recall']:>6.3f} | {m['f1_score']:>6.3f} | {m['specificity']:>6.3f} | {m['false_positive_rate']:>6.3f} | {m['false_negative_rate']:>6.3f}")

    print("\n" + "-" * 80)
    print(" PER-ACTION ACCURACY")
    print("-" * 80)
    for act, s in per_action_summary.items():
        print(f"  {act:<12}: {s['correct']}/{s['total']} ({s['accuracy']*100:.1f}%)")

    print("\n" + "-" * 80)
    print(" OPERATIONAL SAFETY TELEMETRY")
    print("-" * 80)
    print(f"Total Alerts: {total_alerts} ({alerts_per_hour:.1f}/hr)")
    print(f"False Alerts: {false_alerts} ({false_alerts_per_hour:.1f}/hr)")
    print(f"Missed Incidents: {missed_incidents} ({missed_per_hour:.1f}/hr)")
    print(f"Fall -> Aggression Leakage: {fall_to_agg_leakage} (Must be 0)")
    print(f"Forensic Evidence Bundles Generated: {len(all_saved_bundles)}")
    print(f"Results JSON saved to: {output_results_path}")
    print("=" * 80 + "\n")

    return benchmark_summary


if __name__ == "__main__":
    evaluate_guardian_matrix()
