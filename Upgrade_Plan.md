# Guardian Matrix — Complete Upgrade Plan

## 1. Project Objective

Upgrade Guardian Matrix from a basic real-time human/pose detection pipeline into a **research-grade, multi-stage human-interaction analysis system**.

The primary goal for the proof-of-concept phase is:

> Maximize detection quality, evidence quality, and scientific validity first. Hardware optimization for deployment is a later phase.

The system should detect **predefined physical-interaction and risk patterns** from video, not make an unrestricted claim that it can determine whether harassment has legally or socially occurred.

---

# 2. Current Concept

The original concept is:

```text
CCTV
  ↓
Human Detection
  ↓
Pose / Gesture Detection
  ↓
Harassment Detection
  ↓
Alert
```

This is too simple for reliable research.

The upgraded architecture separates:

1. Detection
2. Tracking
3. Pose estimation
4. 3D body understanding
5. Temporal motion analysis
6. Person-person interaction analysis
7. Event classification
8. Fast System-1 decision
9. Verification
10. Safety policy
11. Alerting
12. Evaluation

---

# 3. Target Architecture

```text
                         VIDEO / CCTV
                              │
                              ▼
                    ┌───────────────────┐
                    │ Person Detection  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Multi-Object      │
                    │ Tracking          │
                    └─────────┬─────────┘
                              │
                              ▼
               ┌────────────────────────────┐
               │ High-Precision Pose Layer │
               │                            │
               │ Sapiens / GEM-X / 3D Body │
               └──────────────┬─────────────┘
                              │
                    ┌─────────┴─────────┐
                    │                   │
                    ▼                   ▼
              2D Keypoints          3D Pose
                    │                   │
                    └─────────┬─────────┘
                              ▼
                    ┌───────────────────┐
                    │ Temporal Feature  │
                    │ Extraction        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Person-Person     │
                    │ Interaction       │
                    │ Analysis          │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Temporal Action   │
                    │ Classifier        │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Structured Event  │
                    │ Evidence          │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Laya System-1     │
                    │ Decision Layer    │
                    └─────────┬─────────┘
                              │
                         Confidence
                              │
                 ┌────────────┴────────────┐
                 │                         │
                 ▼                         ▼
              Routine                 Suspicious /
                                      Uncertain
                 │                         │
                 │                         ▼
                 │                System-2 Verification
                 │                         │
                 │                         ▼
                 │                  Safety Policy
                 │                         │
                 └────────────┬────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Final Event State │
                    └─────────┬─────────┘
                              │
                              ▼
                    Alert / Review / Log
```

---

# 4. Core Design Principle

Do not create:

```text
Pose → Laya → Alert
```

Create:

```text
Video
  ↓
Evidence extraction
  ↓
Temporal interpretation
  ↓
Structured event representation
  ↓
System-1 decision
  ↓
Verification
  ↓
Deterministic safety policy
  ↓
Alert / human review
```

The AI models should generate evidence.

The application policy should determine what happens with that evidence.

---

# 5. Model Upgrade Strategy

## 5.1 Person Detection

Evaluate modern detectors rather than locking the project to the existing YOLO implementation.

Candidates:

- Ultralytics YOLO family
- NVIDIA detection models
- RT-DETR-family detectors
- Other current state-of-the-art person detectors

Required output:

```python
PersonDetection(
    track_candidate,
    bounding_box,
    confidence,
    frame_id
)
```

The detector's job is only to answer:

> Where are the people?

It should not decide whether an interaction is harmful.

---

# 6. Multi-Object Tracking

Add persistent identity tracking.

Recommended candidates:

- ByteTrack
- BoT-SORT
- StrongSORT
- OC-SORT

Required output:

```python
TrackedPerson(
    track_id,
    bbox,
    detection_confidence,
    timestamp
)
```

Example:

```text
Frame 001 → Person 7
Frame 002 → Person 7
Frame 003 → Person 7
...
```

This enables temporal reasoning.

---

# 7. High-Precision Human Pose Layer

Evaluate at least three families of models.

## 7.1 NVIDIA GEM-X

Repository:

https://github.com/NVlabs/GEM-X

Purpose:

- Detailed human pose
- 3D motion
- Whole-body information
- 77-joint representation
- Monocular 3D human motion recovery

Use this as a major candidate for the research pipeline.

---

## 7.2 Meta Sapiens

Repository:

https://github.com/facebookresearch/sapiens

Purpose:

- Human-centric vision
- High-resolution pose estimation
- Whole-body pose
- Detailed body/face/hand/foot landmarks

Sapiens provides configurations supporting up to 133 keypoints.

---

## 7.3 SAM 3D Body

Repository:

https://github.com/facebookresearch/sam-3d-body

Purpose:

- 3D human-body reconstruction
- Body, hands and feet
- Difficult poses
- Occlusion handling

Use this as a research comparison rather than automatically assuming it is the final production model.

---

# 8. Pose Benchmark

Run the same annotated videos through:

```text
Baseline:
Current YOLO Pose

Candidate A:
Sapiens

Candidate B:
GEM-X

Candidate C:
SAM 3D Body
```

Measure:

- Keypoint quality
- Occlusion robustness
- Multi-person accuracy
- Small-person performance
- Side-view performance
- Back-view performance
- Low-light performance
- Crowded-scene performance
- Motion-blur performance
- Inference latency
- Memory consumption

The final pose model should be selected from measured results.

Do not select solely from model reputation or benchmark headlines.

---

# 9. Pose Representation

Normalize pose data before temporal analysis.

For every person:

```python
PoseFrame(
    track_id,
    timestamp,
    keypoints_2d,
    keypoints_3d,
    visibility,
    pose_confidence
)
```

Normalize coordinates relative to:

- Person bounding box
- Body center
- Scale
- Camera coordinate system where available

This reduces sensitivity to:

- Camera position
- Person size
- Distance from camera
- Image resolution

---

# 10. Temporal Motion Engine

Single-frame classification is insufficient.

Create a sliding temporal window.

Example:

```text
Frame t-90
Frame t-89
...
Frame t-1
Frame t
```

Possible window:

```text
1–3 seconds
```

depending on camera FPS and the event being studied.

Extract:

### Position

```text
x
y
z
```

### Velocity

```text
dx/dt
dy/dt
dz/dt
```

### Acceleration

```text
d²x/dt²
d²y/dt²
d²z/dt²
```

### Joint angles

Examples:

- elbow angle
- shoulder angle
- knee angle
- hip angle
- torso orientation

### Body orientation

```text
person A facing B
person A facing away
```

### Relative motion

```text
A → B
B → A
```

---

# 11. Person-Person Interaction Engine

This is one of the most important upgrades.

For every pair:

```text
Person A
Person B
```

calculate:

- Distance
- Relative velocity
- Relative direction
- Body orientation
- Hand-to-body distance
- Hand-to-hand distance
- Movement synchronization
- Duration of proximity
- Repeated contact
- Pursuit-like movement
- Separation/avoidance movement

Example:

```python
InteractionFeatures(
    person_a=7,
    person_b=12,
    distance=0.72,
    relative_velocity=...,
    hand_contact_probability=...,
    orientation_similarity=...,
    interaction_duration=3.4
)
```

These are **evidence features**, not conclusions.

---

# 12. Hand and Upper-Body Analysis

Whole-body pose is not enough.

Prioritize:

- Hands
- Wrists
- Elbows
- Shoulders
- Head
- Torso

For interaction analysis, hand trajectories can be particularly informative.

Represent:

```text
left_hand(t)
right_hand(t)
left_elbow(t)
right_elbow(t)
head(t)
torso(t)
```

Then calculate temporal trajectories.

---

# 13. Event Taxonomy

Do not train only:

```text
harassment
not_harassment
```

Create intermediate observable states.

Example:

```text
NORMAL_ACTIVITY
CLOSE_INTERACTION
RAPID_APPROACH
RAPID_SEPARATION
REPEATED_CONTACT_PATTERN
AGGRESSIVE_MOTION_PATTERN
PURSUIT_PATTERN
FALL_PATTERN
RESTRAINT_PATTERN
UNCERTAIN_INTERACTION
```

These are observable model targets.

The final application policy can combine them.

---

# 14. Temporal Action Model

Evaluate:

## ST-GCN

Spatial-Temporal Graph Convolutional Network.

Input:

```text
Joints × Coordinates × Time
```

Useful because the human skeleton naturally forms a graph.

Example:

```text
Shoulder
   │
Elbow
   │
Wrist
```

and:

```text
Hip
├── Knee
│    └── Ankle
```

ST-GCN can learn spatial and temporal relationships.

---

## Temporal Transformer

Alternative:

```text
Pose sequence
     ↓
Embedding
     ↓
Transformer
     ↓
Event representation
     ↓
Classifier
```

Compare:

```text
ST-GCN
vs
Temporal Transformer
```

rather than assuming one is superior.

---

# 15. Evidence Representation

Convert model outputs into structured evidence.

Example:

```json
{
  "scene": {
    "person_count": 2
  },
  "interaction": {
    "pair": [7, 12],
    "distance_m": 0.72,
    "duration_s": 3.4
  },
  "motion": {
    "rapid_approach": 0.91,
    "rapid_separation": 0.84,
    "repeated_contact": 0.76
  },
  "pose": {
    "confidence": 0.93
  },
  "temporal_model": {
    "interaction_pattern": "REPEATED_CONTACT_PATTERN",
    "confidence": 0.88
  }
}
```

This becomes the input to Laya.

---

# 16. Laya System-1 Layer

Laya should not receive raw video.

It should receive structured evidence.

Architecture:

```text
Vision Models
     ↓
Structured Evidence
     ↓
Laya
     ↓
Typed Decision
```

Example decision:

```text
NORMAL
SUSPICIOUS
REVIEW
```

or:

```text
choice:
  routine
  interaction
  suspicious
  uncertain
```

Use:

- `choice`
- `score`
- `noul`

where appropriate.

---

# 17. Confidence Architecture

Never use:

```python
if confidence > 0.8:
    alert()
```

as the entire safety mechanism.

Instead:

```text
Model confidence
       +
Pose quality
       +
Tracking quality
       +
Temporal consistency
       +
Evidence completeness
       +
Calibration
       ↓
Decision policy
```

Example:

```python
if (
    event_confidence >= threshold
    and pose_quality >= minimum_pose_quality
    and tracking_quality >= minimum_tracking_quality
    and temporal_consistency >= minimum_consistency
):
    state = "REVIEW_REQUIRED"
else:
    state = "UNCERTAIN"
```

Thresholds must be learned/validated on held-out data.

---

# 18. System-2 Verification

Use a slower model only when needed.

```text
System-1
   │
   ├── Clear routine → finish
   │
   ├── Clear predefined event → safety policy
   │
   └── Uncertain → System-2
```

System-2 can inspect:

- Event metadata
- Pose sequence
- Key frames
- Short video clip
- Structured evidence

Do not make a large language model the sole source of truth.

---

# 19. Safety Policy

Separate AI inference from real-world actions.

Recommended:

```text
AI inference
    ↓
Evidence
    ↓
Decision
    ↓
Safety policy
    ↓
Action
```

Possible states:

```text
NORMAL
MONITOR
REVIEW
ALERT
```

For the research prototype, the final state should initially trigger:

```text
Store event
Save evidence
Display alert
Request human verification
```

Avoid automatic irreversible interventions.

---

# 20. Dataset Strategy

Create a dedicated Guardian Matrix dataset.

Recommended structure:

```text
dataset/
├── train/
├── validation/
├── test/
├── annotations/
│   ├── poses/
│   ├── tracks/
│   └── events/
└── metadata/
```

Include:

- Normal activities
- Walking
- Running
- Standing
- Talking
- Friendly interaction
- Crowded scenes
- Physical contact
- Rapid movement
- Pursuit-like movement
- Fall-like movement
- Occlusion
- Low light
- Different camera angles

---

# 21. Annotation Format

Each video should contain:

```json
{
  "video_id": "GM_001",
  "fps": 30,
  "persons": [
    {
      "track_id": 7,
      "frames": []
    }
  ],
  "events": [
    {
      "start": 120,
      "end": 184,
      "label": "REPEATED_CONTACT_PATTERN"
    }
  ]
}
```

Keep event labels tied to observable behavior.

---

# 22. Train / Validation / Test Split

Do not randomly split individual frames.

That can cause data leakage.

Instead:

```text
Video A → train
Video B → train
Video C → validation
Video D → test
```

Ideally separate by:

- Camera
- Location
- Recording session
- People
- Scenario

The test set should represent unseen conditions.

---

# 23. Evaluation Metrics

Report at least:

## Detection

- Precision
- Recall
- mAP

## Tracking

- IDF1
- MOTA
- HOTA

## Pose

- PCK
- OKS / mAP where applicable

## Event classification

- Precision
- Recall
- F1
- Confusion matrix

## Safety-oriented metrics

- False-positive rate
- False-negative rate
- False alarms per hour
- Event detection latency

---

# 24. Critical Metric

For a CCTV system, frame-level accuracy can be misleading.

Measure:

```text
False alarms / hour
```

Example:

```text
System A:
97% frame accuracy
20 false alarms/hour

System B:
94% frame accuracy
2 false alarms/hour
```

The second system may be much more usable operationally.

Therefore report event-level metrics and false alarms per hour.

---

# 25. Ablation Study

To make the project scientifically stronger, compare components.

### Experiment 1

```text
YOLO Pose
+
Classifier
```

### Experiment 2

```text
Sapiens
+
Classifier
```

### Experiment 3

```text
GEM-X
+
Classifier
```

### Experiment 4

```text
Pose
+
Temporal Model
```

### Experiment 5

```text
Pose
+
Temporal Model
+
Interaction Features
```

### Experiment 6

```text
Pose
+
Temporal Model
+
Interaction Features
+
Laya
```

This shows which component actually contributes to the improvement.

---

# 26. Baseline vs Upgraded System

Create a formal comparison.

| Component | Baseline | Upgrade |
|---|---|---|
| Detection | Existing YOLO | Benchmark modern detectors |
| Tracking | Optional | ByteTrack/BoT-SORT |
| Pose | Basic YOLO pose | Sapiens/GEM-X/SAM 3D Body |
| Body representation | 2D | 2D + 3D |
| Temporal analysis | Limited | ST-GCN / Transformer |
| Interaction | Basic | Pairwise interaction engine |
| Decision | Direct classifier | Evidence → Laya |
| Uncertainty | Basic confidence | Calibration + quality gates |
| Verification | None | System-2 |
| Evaluation | Demo | Scientific benchmark |

---

# 27. Repository Structure

Recommended project:

```text
guardian-matrix/
│
├── README.md
├── requirements.txt
├── configs/
│   ├── detection.yaml
│   ├── pose.yaml
│   ├── tracking.yaml
│   ├── temporal.yaml
│   └── decision.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── annotations/
│   └── splits/
│
├── src/
│   ├── detection/
│   ├── tracking/
│   ├── pose/
│   ├── geometry/
│   ├── interaction/
│   ├── temporal/
│   ├── classification/
│   ├── system1/
│   ├── system2/
│   ├── policy/
│   ├── evaluation/
│   └── visualization/
│
├── experiments/
│   ├── baselines/
│   ├── pose_comparison/
│   ├── temporal_models/
│   └── ablations/
│
├── checkpoints/
│
├── notebooks/
│
├── tests/
│
└── outputs/
    ├── metrics/
    ├── plots/
    ├── event_clips/
    └── logs/
```

---

# 28. Development Phases

## Phase 1 — Baseline

Implement and freeze the existing pipeline.

Record:

- FPS
- Precision
- Recall
- F1
- False alarms/hour
- Detection latency

This becomes the baseline.

---

## Phase 2 — Tracking

Add:

```text
Detector → Tracker
```

Verify stable person IDs.

---

## Phase 3 — Pose Benchmark

Implement:

```text
YOLO Pose
Sapiens
GEM-X
SAM 3D Body
```

Run identical videos.

Generate comparison tables.

---

## Phase 4 — 3D / Whole-Body Representation

Add:

- 3D joints
- Hands
- Feet
- Face where relevant
- Body orientation

---

## Phase 5 — Interaction Engine

Implement:

- Person distance
- Relative velocity
- Contact likelihood
- Hand trajectories
- Body orientation
- Interaction duration
- Pursuit/separation features

---

## Phase 6 — Temporal Model

Implement:

```text
ST-GCN
```

Then compare against:

```text
Temporal Transformer
```

---

## Phase 7 — Structured Evidence

Create the unified event schema.

---

## Phase 8 — Laya

Add:

```text
Evidence → Laya → typed decision
```

Measure:

- accuracy
- calibration
- latency
- failure cases

---

## Phase 9 — System-2

Add an escalation path for uncertain events.

---

## Phase 10 — Evaluation

Run the complete test set.

Generate:

- Confusion matrix
- Precision/Recall/F1
- ROC/PR curves where appropriate
- False alarms/hour
- Latency distribution
- Pose comparison
- Ablation results

---

# 29. Research Experiments

Minimum experiments:

### Experiment A — Pose model

```text
YOLO Pose
vs
Sapiens
vs
GEM-X
```

### Experiment B — Temporal model

```text
ST-GCN
vs
Transformer
```

### Experiment C — Interaction features

```text
Pose only
vs
Pose + interaction geometry
```

### Experiment D — Decision layer

```text
Direct classifier
vs
Classifier + Laya
```

### Experiment E — Verification

```text
System-1 only
vs
System-1 + System-2
```

---

# 30. Visualization Dashboard

Build a research dashboard displaying:

```text
┌──────────────────────────────────────────┐
│             LIVE VIDEO                   │
│                                          │
│  Person 7          Person 12             │
│    skeleton           skeleton           │
│       │                  │               │
└──────────────────────────────────────────┘

People: 2
Tracking: Stable
Pose Quality: 94%
Interaction: Detected
Temporal Confidence: 88%

System-1:
SUSPICIOUS

Verification:
REQUIRED

Event Duration:
3.4 seconds
```

Also display:

- Bounding boxes
- Track IDs
- Skeletons
- 3D pose visualization
- Interaction lines
- Confidence values
- Event timeline

---

# 31. Logging

Every event should be reproducible.

Store:

```json
{
  "event_id": "GM-00031",
  "timestamp": "...",
  "video_id": "...",
  "track_ids": [7, 12],
  "pose_model": "GEM-X",
  "temporal_model": "ST-GCN",
  "decision_model": "Laya",
  "event_state": "REVIEW",
  "confidence": 0.87,
  "evidence": {},
  "model_versions": {}
}
```

Never store only:

```text
"harassment": true
```

Store the evidence and model versions that produced the decision.

---

# 32. Reproducibility

Every experiment should record:

```text
Git commit
Model version
Checkpoint
Dataset version
Configuration
Random seed
Python version
Framework versions
Inference settings
```

This is important if Guardian Matrix is presented as research.

---

# 33. Accuracy Improvement Strategy

Prioritize improvements in this order:

```text
1. Dataset quality
        ↓
2. Annotation quality
        ↓
3. Pose quality
        ↓
4. Tracking quality
        ↓
5. Temporal modeling
        ↓
6. Interaction features
        ↓
7. Decision calibration
        ↓
8. System-2 verification
        ↓
9. Deployment optimization
```

Do not start with model optimization before establishing a reliable dataset and evaluation protocol.

---

# 34. Important Failure Cases

Explicitly test:

- Multiple people touching
- Friends interacting
- Sports
- Dancing
- Helping someone
- Crowded public areas
- Partial occlusion
- Person leaving the frame
- Camera shake
- Low light
- Back-facing people
- Sitting/lying positions
- Objects covering hands
- Multiple simultaneous interactions
- Fast movement
- Small people in distant CCTV footage

These cases are essential for measuring false positives.

---

# 35. Privacy and Responsible Research

For the prototype:

- Use consented or appropriately licensed footage.
- Minimize unnecessary storage of identifiable imagery.
- Prefer storing derived pose/event data when raw video is not required.
- Restrict access to event footage.
- Document the intended use and limitations.
- Do not claim certainty beyond the validated event classes.

The system should assist human review rather than make unsupported judgments about people.

---

# 36. Final Target

The completed Guardian Matrix research system should demonstrate:

```text
              RAW CCTV
                  │
                  ▼
          Person Detection
                  │
                  ▼
              Tracking
                  │
                  ▼
       Whole-Body Pose / 3D Pose
                  │
                  ▼
        Temporal Motion Analysis
                  │
                  ▼
       Person-Person Interaction
                  │
                  ▼
          Event Classification
                  │
                  ▼
        Structured Evidence
                  │
                  ▼
          Laya System-1
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
     Confident           Uncertain
        │                   │
        │                   ▼
        │              System-2
        │                   │
        └─────────┬─────────┘
                  ▼
            Safety Policy
                  │
          ┌───────┴────────┐
          ▼                ▼
       MONITOR          REVIEW/ALERT
```

---

# 37. Success Criteria

The project should not be declared successful because a demo video looks convincing.

Declare the upgraded system successful only after it demonstrates measurable improvement over the baseline.

Minimum evidence:

- Improved event-level F1
- Reduced false alarms/hour
- Improved recall on target events
- Stable tracking
- Better pose quality
- Demonstrated temporal understanding
- Ablation results
- Unseen-video test results
- Reproducible experiments
- Documented failure cases

---

# 38. Immediate Implementation Order

Start in exactly this order:

```text
01. Freeze current Guardian Matrix baseline
02. Build evaluation dataset
03. Add ByteTrack/BoT-SORT
04. Integrate Sapiens
05. Integrate GEM-X
06. Benchmark pose models
07. Add whole-body/3D representation
08. Build interaction feature engine
09. Implement ST-GCN baseline
10. Implement temporal Transformer
11. Compare temporal models
12. Create structured evidence schema
13. Integrate Laya
14. Add confidence calibration
15. Add System-2 verification
16. Add safety policy
17. Build visualization dashboard
18. Run ablation studies
19. Run unseen test set
20. Produce research report
```

---

# 39. Final Research Position

Guardian Matrix should be presented as:

> **A multi-stage computer-vision system for detecting predefined human-interaction and motion-risk patterns from video using high-precision whole-body pose estimation, temporal modeling, structured evidence, and a calibrated System-1 decision layer.**

Avoid claiming:

> "AI can determine harassment from CCTV."

Instead demonstrate:

> "The system detects and classifies predefined observable interaction patterns and provides evidence for human review."

That framing makes the project technically stronger, easier to evaluate, and more defensible as an academic/research system.
