# Milestone 2 Filtering Completion Design

Date: 2026-05-28

## Purpose

Complete Milestone 2 by turning the current basic temporal filter into an explainable filtering pipeline that reduces false alarms before later FalseNet and alarm-system work.

The system should keep YOLOv26 responsible for candidate fire/smoke detection, then apply deterministic rules before a detection is considered accepted. The first production bias is aggressive false-positive reduction: uncertain, tiny, one-frame, low-confidence, or spammy detections should be rejected and logged with clear reasons.

## Current State

Already implemented:

- `YOLOEngine` loads an Ultralytics model and runs prediction.
- `normalize_yolo_output()` converts YOLO results into `Detection` objects.
- `ImageSource` and `VideoSource` provide a shared `(ret, frame)` interface.
- `render_detections()` draws detections.
- `TemporalFilter` checks minimum persistence by elapsed seconds.
- `scripts/demo_offline.py` runs the offline image/video flow.

Gaps to close for Milestone 2:

- Static filtering rules are not centralized.
- Bounding-box area threshold from config is not enforced.
- `min_consecutive_frames` from config is not enforced.
- Cooldown is not implemented.
- Accepted and rejected decisions are not logged.
- The demo script does not yet use the filtering pipeline.

## Architecture

The Milestone 2 flow should be:

```text
Input source
  -> YOLOEngine.predict()
  -> normalize_yolo_output()
  -> DetectionFilter
  -> TemporalFilter
  -> CooldownTracker
  -> DecisionLogger
  -> render accepted detections
```

Filtering should operate on normalized `Detection` objects so it stays independent from Ultralytics output internals and can be reused later by RTSP, webcam, cloud, FalseNet, and alarm event flows.

## Components

### Filtering Decision Schema

Create a small decision object for every accepted or rejected detection.

Fields:

- `detection`: the original `Detection`.
- `accepted`: boolean.
- `reason`: machine-readable reason string.
- `timestamp`: decision timestamp.

Accepted reasons should use explicit values such as `accepted`. Rejected reasons should be specific, such as `low_confidence`, `class_not_allowed`, `bbox_too_small`, `not_persistent`, and `cooldown_active`.

### DetectionFilter

`DetectionFilter` applies stateless rules:

- class ID must be in `classes.allowed`.
- confidence must be at least `inference.confidence_threshold`.
- bbox area ratio must be at least `filtering.min_bbox_area_ratio`.

The bbox area ratio is calculated as:

```text
detection.bbox_area / (frame_width * frame_height)
```

If frame dimensions are unavailable, bbox area filtering should fail closed by rejecting with `missing_frame_size` instead of silently accepting.

### TemporalFilter

Extend the current temporal filter to support both configured persistence checks:

- `filtering.min_persistence_seconds`
- `filtering.min_consecutive_frames`

Both checks must pass before a detection can be considered temporally stable. The filter state is keyed by source ID for this milestone. Later milestones may key by source plus tracked region if needed.

When no candidate detection is present for a source, temporal state for that source resets.

### CooldownTracker

Add a cooldown layer keyed by source ID.

Responsibilities:

- accept the first stable detection for a source.
- reject subsequent stable detections as `cooldown_active` until `filtering.alarm_cooldown_seconds` has elapsed.
- keep state minimal and deterministic for testing.

Cooldown should be applied after static filtering and temporal persistence. This keeps logs explainable: a detection only reaches cooldown if it already passed earlier checks.

### DecisionLogger

Write accepted and rejected decisions to JSON Lines.

Each record should include:

- decision timestamp.
- source ID.
- frame ID.
- detection class ID and class name.
- confidence.
- bbox coordinates.
- bbox area.
- accepted flag.
- reason.

Default output path should be configurable, with a safe default such as `outputs/detections.jsonl`. The logger should create parent directories when needed.

### Demo Integration

Update `scripts/demo_offline.py` so it:

1. loads filtering config.
2. runs YOLO and normalization as it does now.
3. applies static filtering.
4. applies temporal filtering based on whether the frame has statically valid candidates.
5. applies cooldown only to temporally stable accepted candidates.
6. logs every accepted and rejected decision.
7. renders accepted detections only.

The demo should remain usable for a single image. For single-image input, temporal filtering will normally reject detections unless thresholds are configured low enough for demonstration. This behavior is acceptable because Milestone 2 intentionally prevents single-frame alarms.

## Configuration

Extend `configs/default.yaml` under `filtering` or `logging` as needed.

Required settings:

- `filtering.min_persistence_seconds`
- `filtering.min_consecutive_frames`
- `filtering.min_bbox_area_ratio`
- `filtering.alarm_cooldown_seconds`
- `classes.allowed`

Optional setting:

- `logging.detections_path`, defaulting to `outputs/detections.jsonl` if absent.

## Testing

Unit tests should cover:

- low-confidence rejection.
- non-allowed class rejection.
- bbox-too-small rejection.
- missing frame-size rejection for area filtering.
- temporal rejection before enough seconds and frames.
- temporal acceptance after both thresholds pass.
- cooldown rejection after a first accepted detection.
- logger writes valid JSONL records.

Existing tests must keep passing.

## Success Criteria

Milestone 2 is complete when:

- one-frame detections are rejected by temporal filtering.
- tiny detections are rejected by bbox area ratio.
- low-confidence detections are rejected.
- classes outside config are rejected.
- repeated stable detections are blocked during cooldown.
- accepted and rejected records are written with clear reasons.
- the offline demo uses the filtering pipeline before rendering.
- the project test suite passes.

## Out of Scope

- FalseNet model loading and crop classification.
- alarm event objects and notification delivery.
- webcam and RTSP reconnect behavior.
- multi-object tracking across frames.
- dashboard or cloud API changes.

These remain later milestones.
