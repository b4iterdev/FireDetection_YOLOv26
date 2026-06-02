# Layered Fire and Smoke Detection with YOLOv26 and Temporal Filtering

## Abstract

This project implements a modular video-based fire and smoke detection system built around an Ultralytics YOLOv26 detector and a sequence of deterministic filters. The detector is used as a high-recall candidate generator: it proposes fire/smoke regions in each frame. The filtering pipeline then decides whether each candidate is credible enough to become an alarm event. This separation is important because frame-level object detectors can confuse hazardous fire with visually similar non-hazardous phenomena such as small controlled flames, reflections, bright lights, red/orange objects, video noise, or short-lived flashes.

The current solution reduces false alarms through four active stages: confidence/class filtering, spatial bounding-box filtering, temporal behavior tracking, and temporal persistence gating. Accepted detections are rendered onto frames, logged as structured decisions, and displayed either through the offline OpenCV demo or the Flask live web interface. The result is a practical alarm pipeline where YOLOv26 performs visual localization and the filters enforce operational rules about confidence, size, growth behavior, and persistence over time.

## Problem Statement

Fire detection from ordinary RGB video is difficult because fire is not a rigid object. Flame and smoke change shape, color, opacity, texture, and size across time. At the same time, many non-hazardous or non-fire objects share similar visual features:

- candles, lighters, gas stove flames, and controlled cooking flames;
- sunlight, reflections, lens flare, and bright lamps;
- red/orange objects such as clothing, vehicles, signs, and traffic lights;
- smoke-like fog, mist, clouds, dust, or compression artifacts;
- single-frame glitches, motion blur, and low-confidence detector noise.

A detector-only system tends to answer the question: **does this frame contain something that looks like fire or smoke?** An alarm system must answer a harder question: **is this a persistent, hazardous event worth alerting on?** This project addresses that gap by treating YOLO output as proposals and adding explicit decision layers after inference.

## System Overview

The active runtime pipeline is:

```text
Input frame
  -> YOLOEngine.predict(...)
  -> normalize_yolo_output(...)
  -> DetectionFilter.check(...)
  -> BehaviorTracker.check(...)
  -> TemporalFilter.check(...)
  -> DetectionDecision records
  -> render_detections(...)
  -> display / MJPEG stream / JSONL log
```

The main modules are:

| Responsibility | File | Role |
|---|---|---|
| Offline CLI entrypoint | `scripts/demo_offline.py` | Runs detection on an image/video path and displays annotated frames with OpenCV. |
| Web entrypoint | `scripts/web_app.py` | Launches the Flask interface. |
| Live web processing | `fire_detection_alarm/web/live.py` | Streams annotated frames as MJPEG from webcam, video file, or RTSP input. |
| Batch web processing | `fire_detection_alarm/web/pipeline.py` | Processes still-image uploads and can write annotated batch outputs for image/video sources. |
| Configuration | `fire_detection_alarm/app/config.py`, `configs/default.yaml` | Loads model, inference, and filter thresholds. |
| Model wrapper | `fire_detection_alarm/models/yolo_engine.py` | Wraps `ultralytics.YOLO.predict`. |
| Normalization | `fire_detection_alarm/detection/normalizer.py` | Converts raw YOLO boxes into canonical `Detection` objects. |
| Static filtering | `fire_detection_alarm/filtering/detection_filter.py` | Applies class, confidence, and area-ratio gates. |
| Behavior tracking | `fire_detection_alarm/filtering/behavior_tracker.py` | Tracks candidate behavior over frames and rejects stable small fire-like objects. |
| Temporal filtering | `fire_detection_alarm/filtering/temporal_filter.py` | Requires detections to persist for enough time and frames. |
| Rendering | `fire_detection_alarm/detection/renderer.py` | Draws accepted detections. |
| Logging | `fire_detection_alarm/logging/detection_logger.py` | Writes structured JSONL decision records. |

## Configuration

The default configuration is in `configs/default.yaml`:

```yaml
model:
  path: "models/fire_yolov26.pt"
  device: "mps"
  image_size: 640

inference:
  confidence_threshold: 0.65
  iou_threshold: 0.45
  max_fps: 5

filtering:
  min_persistence_seconds: 2.0
  min_consecutive_frames: 5
  min_bbox_area_ratio: 0.002

behavior_tracking:
  min_track_frames: 3
  max_stable_growth_ratio: 0.1
  max_non_hazard_area_ratio: 0.01
  min_growth_ratio: 0.5

classes:
  allowed: [0, 1]
```

Class IDs `0` and `1` are treated as smoke and fire candidates. The model path points to external weights; model weights should not be committed into the repository. `max_fps` is present in configuration for runtime policy, but the current processing loops do not actively throttle frames with it.

## Data Model

YOLO predictions are converted into a canonical schema before any filter sees them. The system uses `Detection` objects rather than raw YOLO tensors:

```python
Detection(
    source_id: str,
    frame_id: int,
    timestamp: float,
    class_id: int,
    class_name: str,
    confidence: float,
    bbox_xyxy: list[float],
    bbox_area: float,
)
```

The normalizer reads `result.boxes.data`, where each row is interpreted as:

```text
x1, y1, x2, y2, confidence, class_id
```

It computes:

```python
bbox_area = (x2 - x1) * (y2 - y1)
```

Each filter returns or contributes to a `DetectionDecision`:

```python
DetectionDecision(
    detection: Detection,
    accepted: bool,
    reason: str,
    timestamp: float,
)
```

This creates an audit trail. Instead of only knowing that an alarm was or was not produced, the system records why a detection was accepted or rejected.

## Stage 1: YOLOv26 Candidate Generation

YOLOv26 is used to localize fire/smoke candidates. The project wraps Ultralytics through `YOLOEngine`:

```python
results = engine.predict(
    frame,
    conf=confidence_threshold,
    iou=iou_threshold,
    imgsz=image_size,
)
```

The web and live paths pass `image_size` from configuration. The offline demo currently omits `imgsz`, so `YOLOEngine` uses its default `640`, which matches the default YAML value.

The detector stage is intentionally not the final alarm decision. It answers: **what regions in this frame look like fire or smoke?** Subsequent filters answer: **should this candidate be trusted as a meaningful alarm?**

Using a detector as a candidate generator is a common design pattern in video fire analytics. It keeps detection sensitive enough to find early fire-like regions while allowing downstream logic to suppress nuisance detections.

## Stage 2: Static Detection Filtering

`DetectionFilter` applies deterministic checks to each normalized detection:

1. **Frame-size validity**
   - Rejects when frame dimensions are missing or invalid.
   - Reason: `missing_frame_size`.

2. **Class allowlist**
   - Accepts only configured classes: `[0, 1]`.
   - Reason on rejection: `class_not_allowed`.

3. **Confidence threshold**
   - Rejects detections below `confidence_threshold` (`0.65` by default).
   - Reason: `low_confidence`.

4. **Minimum bounding-box area ratio**
   - Computes:
     ```python
     bbox_area_ratio = detection.bbox_area / frame_area
     ```
   - Rejects detections below `min_bbox_area_ratio` (`0.002` by default).
   - Reason: `bbox_too_small`.

This stage removes obvious false positives before the system spends stateful logic on them. Low-confidence boxes and extremely small boxes are common sources of nuisance alarms in object-detection systems.

## Stage 3: Behavior Tracking

The `BehaviorTracker` adds temporal behavior features after static filtering. It tracks candidates by:

```python
(source_id, class_id)
```

For each track, it stores:

```python
first_area
frames
```

The filter then compares the current area with the first observed area:

```python
growth_ratio = (current_bbox_area - first_area) / first_area
area_ratio = current_bbox_area / frame_area
```

It emits three active outcomes:

| Reason | Accepted? | Meaning |
|---|---:|---|
| `behavior_observed` | yes | The track is too new or not clearly hazardous/non-hazardous yet. |
| `behavior_growing` | yes | The region grew by at least `min_growth_ratio` (`0.5` by default). |
| `stable_small_fire` | no | The region stayed small and stable for at least `min_track_frames` frames. |

`stable_small_fire` is defined by:

```python
track.frames >= min_track_frames
area_ratio <= max_non_hazard_area_ratio
growth_ratio <= max_stable_growth_ratio
```

With defaults:

```yaml
min_track_frames: 3
max_stable_growth_ratio: 0.1
max_non_hazard_area_ratio: 0.01
min_growth_ratio: 0.5
```

This means a small fire-like object occupying at most 1% of the frame and growing no more than 10% after at least 3 tracked frames is treated as likely non-hazardous. Examples include candles, small stove flames, lighters, and stable reflections.

The system does not claim that all stable small flames are safe. Instead, it reduces false alarms for the use case where small stable flame-like detections are less likely to represent spreading hazardous fire than growing regions.

## Stage 4: Temporal Persistence

The `TemporalFilter` requires the post-filter signal to remain present long enough before an alarm is accepted. It tracks per-source state:

```python
active_starts: dict[str, float]
active_counts: dict[str, int]
```

The logic is:

```python
seconds_passed = min_seconds <= 0 or duration >= min_seconds
frames_passed = min_frames <= 0 or active_counts[source_id] >= min_frames
accepted = seconds_passed and frames_passed
```

Defaults require both:

- at least `2.0` seconds of continuous detection;
- at least `5` consecutive processed frames.

If a frame contains no post-filter detections, the source state resets. This is why transient objects, one-frame flashes, brief model mistakes, and intermittent low-confidence detections produce `not_persistent` instead of alarms.

The calling code maps temporal rejection to:

```text
not_persistent
```

Once persistence passes, the detection becomes:

```text
accepted
```

## Decision Reasons

The active decision reasons are:

| Reason | Stage | Meaning |
|---|---|---|
| `missing_frame_size` | Static filter | Frame dimensions are missing or invalid. |
| `class_not_allowed` | Static filter | Class ID is outside the configured allowlist. |
| `low_confidence` | Static filter | YOLO confidence is below threshold. |
| `bbox_too_small` | Static filter | Candidate region is too small relative to frame. |
| `behavior_observed` | Behavior tracker | Candidate is accepted for now; more temporal evidence may be needed. |
| `behavior_growing` | Behavior tracker | Candidate area grew enough to look more hazard-like. |
| `stable_small_fire` | Behavior tracker | Candidate is small and stable enough to suppress. |
| `not_persistent` | Temporal filter | Candidate has not persisted long enough in time and frames. |
| `accepted` | Final decision | Candidate passed active filters and persistence. |

The project intentionally no longer uses a cooldown gate or FalseNet stage. Current alarm acceptance depends on the active stages above.

## Offline and Live Execution

### Offline CLI

`scripts/demo_offline.py` reads from an image or video file and uses OpenCV display:

```bash
PYTHONPATH=. venv/bin/python scripts/demo_offline.py --input path/to/video.mp4 --model models/fire_yolov26.pt
```

For each frame, it applies the detector and filters, logs every decision, renders accepted boxes, and shows the annotated frame through OpenCV. The offline demo reads the model path from `--model` while using the YAML file for device, inference thresholds, class IDs, and filter thresholds.

### Web UI and Live Player

`scripts/web_app.py` starts a Flask server:

```bash
PYTHONPATH=. venv/bin/python scripts/web_app.py
```

The live UI supports:

- webcam input;
- local video-file playback;
- RTSP URL input;
- uploaded video live playback;
- MJPEG annotated frame streaming via `/stream.mjpeg`;
- status through `/api/live/status`.

The live session maintains:

```python
running
source_type
frame_count
accepted_count
latest_reason
error
```

Video uploads are played as live streams instead of waiting for batch completion. This allows the user to watch annotated frames and decision status at the same time. Still-image uploads use the batch web pipeline and produce annotated result files.

## How the Layers Reduce False Alarms

The design uses layered rejection. Each stage addresses a different false-positive pattern:

| Layer | False positives reduced |
|---|---|
| Confidence threshold | Weak YOLO guesses, detector noise, uncertain boxes. |
| Class allowlist | Non-fire classes or unexpected class IDs. |
| Minimum area ratio | Tiny sparks, small reflections, distant noise, compression artifacts. |
| Behavior tracking | Small stable flames/reflections that do not grow like an escalating fire. |
| Temporal persistence | Single-frame flashes, momentary glare, intermittent detections, brief motion blur. |

This architecture is more robust than a detector-only system because no single frame-level prediction immediately becomes an alarm. A detection must be plausible spatially, behaviorally, and temporally.

## Relationship to Prior Work and Common Patterns

This project follows patterns used in practical video fire detection systems:

1. **Temporal persistence / N-of-M logic**
   - Open-source projects such as `pedbrgs/Fire-Detection` and `antoniolorenz01/FireScope-V1.0` use temporal buffering or N-of-M decision logic to avoid alerting on isolated frames.
   - This project implements a per-source persistence gate requiring both time and consecutive frame count.

2. **Spatial filtering**
   - Fire detection systems commonly reject boxes below a minimum area threshold because tiny boxes are often noise, reflections, or irrelevant sparks.
   - This project uses `min_bbox_area_ratio` to compare each candidate to total frame area.

3. **Behavior / area-variation tracking**
   - Hybrid fire detection research often uses temporal behavior, area variation, or growth patterns to separate real events from stable false positives.
   - This project uses bounding-box area growth as a lightweight behavior feature.

4. **Class and confidence gating**
   - Per-class and confidence-based gates are common in YOLO-based safety systems.
   - This project gates by allowed class IDs and detector confidence before more expensive temporal logic.

5. **Live video analytics**
   - Industrial and research systems increasingly combine visual detection with temporal verification and live operator review.
   - The Flask live player provides real-time annotated feedback, making decisions auditable by showing the video and status together.

## Strengths

- **Modular pipeline**: detector, normalizer, filters, renderer, and logger are separated.
- **Auditable decisions**: every rejection/acceptance has a reason string.
- **Configurable thresholds**: model, inference, and filters are controlled through YAML.
- **Live operation**: web interface supports webcam, video file, RTSP, and uploaded video playback.
- **False-alarm reduction**: filters address confidence, size, behavior, and persistence.
- **Test coverage**: tests cover filtering, behavior tracking, temporal persistence, normalization, logging, and web/live routes.

## Limitations

- **Simple object tracking**: `BehaviorTracker` tracks by `(source_id, class_id)`, not by object identity or IoU. Multiple fire boxes of the same class in one source can share one track.
- **Temporal tuning required**: strict persistence thresholds can produce `not_persistent` during early frames or low-FPS processing.
- **RGB-only**: no thermal, smoke sensor, humidity, or multispectral fusion is used.
- **Detector-dependent**: if YOLO misses frames or confidence drops, temporal persistence resets.
- **No user authentication**: the web interface is intended for local use.
- **No database-backed incident history**: logs are JSONL files rather than a full incident store.

## Future Work

The most valuable next improvements are:

1. **IoU/centroid-based multi-object tracking**
   - Track each fire region separately instead of `(source_id, class_id)` only.

2. **Adaptive persistence thresholds**
   - Require shorter persistence for rapidly growing fire and longer persistence for ambiguous low-confidence detections.

3. **Separate smoke/fire policy**
   - Use class-specific thresholds because smoke and flame have different false-positive patterns.

4. **Incident timeline UI**
   - Persist accepted alarms and decision history in a searchable dashboard.

5. **Sensor or context fusion**
   - Combine RGB detections with thermal cameras, smoke sensors, humidity, or scene zones.

6. **Source-specific rules**
   - Allow different thresholds for kitchens, outdoor cameras, storage areas, or industrial zones.

## Conclusion

The system is best understood as a layered fire-alarm decision pipeline. YOLOv26 provides visual candidate detection, but alarms are produced only after candidates pass static checks, behavior tracking, and temporal persistence. This design reduces false alarms by forcing fire-like visual evidence to be sufficiently confident, sufficiently large, behaviorally plausible, and persistent across time. The live web player then makes the result usable by showing annotated video and current decision state together.
