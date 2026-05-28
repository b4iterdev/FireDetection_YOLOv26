# Fire Detection Alarm System Design

Date: 2026-05-27

## 1. Purpose

Build a Fire Detection Alarm System that consumes an externally trained YOLOv26 fire detection model and runs inference across images, video files, webcams, and RTSP streams. The system should start as a model demo and evolve into a cloud GPU alarm service with dashboard alerts, notifications, and optional external integrations.

The first milestone is an offline model demo: load a trained YOLOv26 model, run detection on images and videos, draw bounding boxes, save annotated outputs, and produce detection logs.

## 2. Scope

### In scope

- Load a trained YOLOv26 model produced outside this application.
- Run inference on images, video files, webcams, and RTSP streams.
- Normalize model detections into a stable internal schema.
- Apply aggressive false alarm filtering.
- Generate alarm events from filtered detections.
- Save annotated outputs, snapshots, and detection logs.
- Support local PC demo workflows.
- Prepare the architecture for cloud GPU deployment.
- Prepare later dashboard, notification, and external integration features.

### Out of scope

- Dataset labeling UI.
- Dataset management platform.
- Training pipeline implementation.
- Training dashboard.

The application only needs to use a model trained elsewhere. It may document model preparation expectations, but it will not implement labeling or training workflows.

## 3. Deployment Targets

The primary production target is a cloud GPU server. The system should also support local PC demos and RTSP stream inference.

Supported modes:

1. Local demo mode for images and videos.
2. Local webcam mode.
3. RTSP stream mode.
4. Cloud GPU inference service mode.
5. Later dashboard and notification mode.

## 4. Architecture

Use a modular pipeline architecture:

```text
Input Source
   ↓
Frame Reader
   ↓
YOLOv26 Inference Engine
   ↓
Detection Normalizer
   ↓
FalseNet Threat Classifier
   ↓
False Alarm Filter
   ↓
Alarm Event Manager
   ↓
Outputs
```

Each stage has one responsibility. This keeps the first demo simple while making RTSP, cloud, dashboard, and notification features easier to add later.

## 5. Components

### 5.1 Input Source Layer

The input layer makes images, videos, webcams, and RTSP streams look the same to the rest of the application.

Supported input types:

- Single image.
- Image folder.
- Video file.
- Webcam.
- RTSP stream.

Each frame should include:

```text
source_id
frame_id
timestamp
image/frame data
```

Common source interface:

```text
source.open()
source.read_frame()
source.close()
```

### 5.2 YOLOv26 Inference Engine

The inference engine isolates all YOLOv26-specific logic.

Responsibilities:

- Load the trained model.
- Select device: CPU, CUDA, or other supported accelerator.
- Run prediction on frames.
- Return raw detections.
- Hide YOLOv26 output details from the rest of the app.

The engine exposes a generic `predict(frame)` interface.

### 5.3 Detection Normalizer

The normalizer converts model output into a stable internal format:

```text
Detection:
  source_id
  frame_id
  timestamp
  class_name
  confidence
  bbox_xyxy
  bbox_area
```

This lets filtering, rendering, logging, and alarms remain independent from model output format changes.

### 5.4 Renderer

The renderer draws detections on images or video frames.

Responsibilities:

- Draw bounding boxes.
- Draw class names and confidence scores.
- Save annotated images.
- Write annotated video output.

### 5.5 FalseNet Threat Classifier

FalseNet is an optional second-stage classifier used after YOLOv26 finds a fire candidate. YOLOv26 remains responsible for locating possible fire regions. FalseNet is responsible for classifying whether each detected fire region is likely to be a real threat.

FalseNet should use the YOLOv26 fire crop as input for the first implementation. Crop-only classification is simpler, faster, and easier to integrate than whole-frame context classification. Whole-frame or crop-plus-context classification can be added later if crop-only FalseNet is not accurate enough.

Recommended FalseNet classes:

```text
threat_fire
non_threat_fire
fire_like_false_positive
uncertain
```

Examples FalseNet should help suppress:

- Candles.
- Cooking flames.
- Fireplaces.
- Welding or sparks.
- Reflections or orange glare.
- Fire shown on a screen.
- Small contained flames that should not trigger emergency alarms.

FalseNet should not be mandatory for the first model demo. It belongs after basic YOLOv26 inference and before the full alarm system. The system should still work if FalseNet is disabled.

FalseNet output should include:

```text
classification
confidence
model_version
crop_bbox
```

Recommended behavior:

- `threat_fire` continues to the temporal false alarm filter.
- `non_threat_fire` is logged but does not trigger alarms.
- `fire_like_false_positive` is rejected and logged.
- `uncertain` follows conservative configuration, either reject or require stronger temporal confirmation.

### 5.6 False Alarm Filter

False alarm filtering should be aggressive. The system should prefer fewer false alarms over early but unreliable alarms.

Initial filtering rules:

- Confidence must meet a configurable threshold.
- Detection class must be allowed, such as `fire` or `smoke`.
- If enabled, FalseNet must classify the crop as `threat_fire` or pass the configured `uncertain` policy.
- Bounding box area must exceed a minimum ratio.
- Fire must persist for a configured number of frames or seconds.
- Single-frame detections should not trigger alarms.
- Region-of-interest and ignore-zone rules may suppress detections in known noisy areas.
- Cooldown should prevent repeated notifications for the same ongoing event.
- Thresholds should be configurable per source.

Later filtering improvements:

- Fire-like color and texture heuristics.
- Motion consistency checks.
- Smoke/fire temporal risk scoring.
- Whole-frame or crop-plus-context FalseNet classification.
- Scene-specific ignore zones.

False alarm categories to consider:

- Sunlight, reflections, and glare.
- Orange or red objects.
- Smoke, fog, steam, and dust.
- Welding, sparks, and cooking flames.
- Screen artifacts, low-light noise, and camera compression artifacts.

### 5.7 Alarm Event Manager

The alarm manager converts accepted detections into alarm events.

Alarm event fields:

```text
event_id
source_id
start_time
last_seen_time
severity
max_confidence
snapshot_path
detections
status
```

Statuses:

```text
active
cooldown
resolved
dismissed
```

Responsibilities:

- Create event IDs.
- Store event evidence.
- Apply cooldown.
- Classify severity.
- Trigger configured outputs.
- Avoid duplicate notification spam.

### 5.8 Output and Notification Layer

Start with simple outputs and expand later.

Initial outputs:

- Annotated image/video.
- Terminal logs.
- JSON or CSV detection log.
- Event snapshots.

Later outputs:

- Dashboard event.
- Webhook.
- Telegram.
- Email.
- SMS.
- External hardware or third-party API trigger.

Notification failure must not prevent alarm event creation.

### 5.9 Storage Layer

For the first version, local filesystem storage is enough.

Store:

- Annotated snapshots.
- Annotated videos.
- JSON event metadata.
- Detection logs.
- Rejected detection logs.

For cloud production, storage can evolve into:

- PostgreSQL for event metadata.
- Object storage for snapshots and clips.

### 5.10 API and Dashboard Layer

The API and dashboard are later phases.

Possible API endpoints:

```text
GET  /health
GET  /sources
POST /sources
GET  /events
GET  /events/{id}
POST /events/{id}/dismiss
POST /infer/image
POST /infer/video
```

Dashboard views:

- Stream list.
- Stream health.
- Recent alarm events.
- Event detail and evidence.
- Per-camera filtering settings.
- Notification settings.

## 6. Data Flow

### 6.1 Offline Image and Video Demo

```text
User selects image/video
   ↓
App loads YOLOv26 model
   ↓
Frames are read
   ↓
Model predicts fire detections
   ↓
Detections are normalized
   ↓
FalseNet classifies YOLO fire crops if enabled
   ↓
False alarm filter checks confidence, size, and persistence
   ↓
Accepted detections are rendered
   ↓
Annotated output and logs are saved
```

Expected outputs:

- Annotated image or video.
- JSON or CSV detection log.
- Summary report with detection count, average confidence, timestamps, and filtered versus accepted detections.

### 6.2 Webcam and RTSP Live Flow

```text
Camera / RTSP stream
   ↓
Frame reader with reconnect handling
   ↓
FPS sampling
   ↓
YOLOv26 inference
   ↓
FalseNet crop classification
   ↓
Temporal false alarm filtering
   ↓
Alarm event manager
   ↓
Dashboard / notification / webhook
```

RTSP requirements:

- Reconnect if stream drops.
- One failed camera should not crash other streams.
- Process configurable FPS, such as 3 to 10 FPS.
- Support per-camera threshold, persistence, and cooldown settings.

### 6.3 Cloud GPU Flow

```text
RTSP cameras / uploaded videos
   ↓
Cloud inference worker
   ↓
GPU YOLOv26 inference
   ↓
FalseNet crop classification
   ↓
False alarm filter
   ↓
Alarm event API
   ↓
Dashboard + notifications
```

Recommended deployment shape:

- Inference service for model execution.
- API and dashboard service for users and alarm events.
- Optional queue for multiple streams.
- Persistent storage for events, snapshots, and logs.

## 7. Configuration

Use configuration files so behavior can change without code edits.

Example global configuration:

```yaml
model:
  path: "models/fire_yolov26.pt"
  device: "auto"
  image_size: 640

inference:
  confidence_threshold: 0.65
  iou_threshold: 0.45
  max_fps: 5

filtering:
  min_persistence_seconds: 2.0
  min_consecutive_frames: 5
  min_bbox_area_ratio: 0.002
  alarm_cooldown_seconds: 60
  uncertain_falsenet_policy: "require_stronger_temporal_confirmation"

falsenet:
  enabled: false
  model_path: "models/falsenet.pt"
  input_mode: "crop"
  threat_threshold: 0.7

classes:
  allowed:
    - fire
    - smoke

outputs:
  save_annotated: true
  save_json_log: true
  save_snapshots: true

notifications:
  webhook:
    enabled: false
    url: ""
  telegram:
    enabled: false
    bot_token: ""
    chat_id: ""
```

Example per-source configuration:

```yaml
sources:
  - id: "demo_video"
    type: "video"
    uri: "samples/fire_demo.mp4"
    enabled: true
    confidence_threshold: 0.7
    falsenet_enabled: false

  - id: "warehouse_camera_1"
    type: "rtsp"
    uri: "rtsp://example/stream"
    enabled: true
    max_fps: 5
    min_persistence_seconds: 3
    cooldown_seconds: 120
    falsenet_enabled: true
```

Per-source settings are important because different camera scenes need different thresholds.

## 8. Recommended Project Structure

```text
fire_detection_alarm/
  app/
    main.py
    config.py

  models/
    loader.py
    yolo_v26_engine.py

  inputs/
    base.py
    image_source.py
    video_source.py
    webcam_source.py
    rtsp_source.py

  detection/
    schema.py
    normalizer.py
    renderer.py

  falsenet/
    classifier.py
    schema.py

  filtering/
    false_alarm_filter.py
    temporal_filter.py
    roi_filter.py
    cooldown.py

  alarms/
    event.py
    manager.py
    notifiers/
      base.py
      webhook.py
      email.py
      telegram.py

  storage/
    event_store.py
    media_store.py

  api/
    server.py
    routes.py

  dashboard/
    # optional later frontend

  configs/
    default.yaml
    sources.yaml

  scripts/
    run_image.py
    run_video.py
    run_webcam.py
    run_rtsp.py

  tests/
```

## 9. Milestones

### Milestone 1: Model Demo

Goal: prove the model can detect fire in images and videos.

Deliverables:

- CLI or simple app entrypoint.
- YOLOv26 model loading.
- Image inference.
- Video inference.
- Bounding boxes and labels drawn on output.
- Annotated output saved.
- Detection log saved.
- Configurable confidence threshold.

Success criteria:

- User can run detection on sample images and videos.
- Output clearly shows detected fire regions.
- Detection logs contain timestamp, class, confidence, and bounding box.
- False positives can be reduced through threshold configuration.

### Milestone 2: False Alarm Filtering

Goal: make detections more reliable.

Deliverables:

- Temporal confirmation.
- Minimum detection duration.
- Bounding box area filtering.
- Cooldown logic.
- Per-source config.
- Accepted and rejected detection logs.

Success criteria:

- One-frame false detections do not trigger alarms.
- Repeated stable fire detections create alarm events.
- Filtering behavior is explainable from logs.

### Milestone 2.5: FalseNet Threat Classification

Goal: distinguish real threat fire from non-threat fire or fire-like false positives.

Deliverables:

- FalseNet model loader.
- Crop extraction from YOLOv26 fire detections.
- Crop-only classification interface.
- Threat classification schema.
- Configurable FalseNet thresholds.
- Logs for threat, non-threat, false-positive, and uncertain classifications.

Success criteria:

- YOLOv26 fire detections can be passed into FalseNet as crops.
- Non-threat fire crops can be rejected before alarm creation.
- FalseNet can be disabled without breaking the normal YOLOv26 pipeline.
- Uncertain classifications follow an explicit configured policy.

### Milestone 3: Webcam and RTSP Support

Goal: support live streams.

Deliverables:

- Webcam input adapter.
- RTSP input adapter.
- Stream reconnect handling.
- FPS sampling control.
- Live preview or annotated stream output.

Success criteria:

- App can process a local webcam.
- App can process an RTSP stream.
- Dropped RTSP stream does not crash the app.

### Milestone 4: Alarm System

Goal: turn filtered detections into alarm events.

Deliverables:

- Alarm event object.
- Alarm severity.
- Event snapshot saving.
- Cooldown.
- Notification interface.
- Initial webhook, Telegram, or email notifier.

Success criteria:

- Accepted fire detection creates an alarm event.
- Alarm contains timestamp, source, confidence, snapshot, and event ID.
- Repeated detections do not spam notifications.

### Milestone 5: Cloud GPU Deployment

Goal: run inference on a GPU server.

Deliverables:

- Containerized app.
- Model path configuration.
- CUDA-compatible runtime.
- API endpoint for image, video, or stream jobs.
- Logs and health check.
- Deployment documentation.

Success criteria:

- App runs on a cloud GPU server.
- Model loads successfully.
- Inference can process an uploaded video or RTSP stream.
- Service exposes health and status information.

### Milestone 6: Dashboard

Goal: monitor streams and alarm events.

Deliverables:

- Stream list.
- Alarm event list.
- Event detail page.
- Detection snapshot or video evidence.
- Source configuration.
- Threshold and cooldown settings.

Success criteria:

- User can see recent alarms.
- User can inspect evidence.
- User can adjust per-camera filtering settings.

## 10. Error Handling

### Model errors

Handle:

- Missing model file.
- Unsupported model format.
- CUDA unavailable.
- Prediction failure.

Expected behavior:

- Show clear error messages.
- Fail fast for demo commands.
- Expose unhealthy status in server mode.

### Input errors

Handle:

- Missing image or video file.
- Invalid webcam index.
- RTSP connection failure.
- RTSP stream timeout.
- Corrupted frames.

Expected behavior:

- Image and video mode fail with clear messages.
- RTSP mode retries connection with backoff.
- One failed stream does not crash other streams.

### Runtime errors

Handle:

- GPU out-of-memory.
- Slow inference.
- Bad frame shape.
- Model output mismatch.

Expected behavior:

- Log failure with source and frame metadata.
- Skip bad frames when safe.
- Allow lower FPS or batch size through configuration.

### Notification errors

Handle:

- Webhook timeout.
- Telegram or email failure.
- Duplicate alarm attempts.
- Missing snapshot path.

Expected behavior:

- Alarm event is still created.
- Notification failure is logged.
- Notification errors do not crash inference.

## 11. Testing Plan

### Milestone 1 tests

- Model path validation.
- Image loading.
- Video loading.
- Detection output normalization.
- Annotated output saved.
- JSON or CSV detection log saved.

Manual acceptance test:

```text
Run model on sample image/video.
Confirm output file contains bounding boxes.
Confirm detection log contains timestamp, class, confidence, and bbox.
```

### False alarm filter tests

Use synthetic detection sequences:

```text
Case 1: one-frame fire detection → rejected
Case 2: low confidence detection → rejected
Case 3: tiny bounding box → rejected
Case 4: stable fire for required duration → accepted
Case 5: repeated fire during cooldown → no duplicate alarm
```

### FalseNet tests

- YOLOv26 detection crop is extracted with the expected bounding box.
- FalseNet `threat_fire` classification passes to temporal filtering.
- FalseNet `non_threat_fire` classification is rejected and logged.
- FalseNet `fire_like_false_positive` classification is rejected and logged.
- FalseNet `uncertain` classification follows the configured policy.
- Disabled FalseNet does not block the YOLOv26-only pipeline.

### Webcam and RTSP tests

- Webcam opens and closes correctly.
- RTSP reconnect does not crash app.
- FPS limiter works.
- Dropped frames do not block the pipeline.

### Alarm tests

- Accepted detection creates event.
- Event includes snapshot path.
- Cooldown prevents spam.
- Failed notification does not delete event.

### Cloud deployment tests

- Container starts.
- Health endpoint works.
- Model loads on GPU.
- Image inference endpoint returns detections.
- Logs show model, device, and source status.

## 12. Recommended Build Order

1. Project skeleton and config loader.
2. Model loader and YOLOv26 inference wrapper.
3. Image inference command.
4. Video inference command.
5. Detection normalization and rendering.
6. Detection logs and annotated outputs.
7. False alarm filtering.
8. Optional FalseNet crop-only threat classifier.
9. Alarm event manager.
10. Webcam support.
11. RTSP support.
12. Notification interface.
13. Cloud API service.
14. Dashboard.
15. Production storage.

This order avoids overbuilding before the model demo works.

## 13. Model Preparation Requirements

The app expects a trained YOLOv26-compatible model produced by an external workflow.

Expected model properties:

- Detects at least the `fire` class.
- May optionally detect `smoke`.
- Can be loaded by the chosen YOLOv26 runtime.
- Provides bounding boxes, class labels, and confidence scores.

Recommended validation before using the model in the alarm system:

- Test on fire and non-fire images.
- Test against common false-positive scenes.
- Measure precision, recall, and false-positive rate.
- Choose conservative thresholds for alarm deployment.

## 14. Open Decisions for Implementation Planning

These decisions should be made before implementation begins:

1. Which YOLOv26 runtime/package will be used.
2. Which model formats must be supported first, such as `.pt`, ONNX, or TensorRT.
3. Which FalseNet runtime and model format should be supported first.
4. Whether FalseNet should be trained externally like the YOLOv26 model or provided as a simple classifier interface first.
5. Whether the first UI is CLI-only or includes a simple local preview window.
6. Which notification channel should be implemented first.
7. Whether the cloud service should use FastAPI, another Python framework, or a simpler CLI worker first.
