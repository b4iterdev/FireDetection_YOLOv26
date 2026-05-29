## Milestone 3 Live Stream Input Design

Date: 2026-05-29

## Purpose

Extend the current offline YOLOv26 demo so the same CLI can process live webcam and RTSP inputs in addition to images and video files.

Milestone 3 should preserve the existing detection pipeline after frame acquisition:

- YOLO inference
- detection normalization
- filtering
- decision logging
- rendering

The new work is limited to live-source acquisition behavior: webcam selection, RTSP handling, reconnect behavior, FPS sampling, and live preview continuity.

## Current State

Already implemented:

- `scripts/demo_offline.py` is the single runnable CLI entrypoint.
- `ImageSource` and `VideoSource` provide a shared `read()` / `release()` interface.
- YOLO inference, normalization, filtering, logging, and rendering work for file-based inputs.
- `configs/default.yaml` already defines `inference.max_fps`, but the current demo does not use it.

Gaps to close for Milestone 3:

- webcam input is not supported.
- RTSP input is not supported.
- there is no reconnect behavior for dropped live streams.
- FPS sampling is not enforced for live sources.
- the live preview has no reconnect or failed-stream status behavior.

## Architecture

Milestone 3 keeps one CLI entrypoint and extends the input-source layer.

```text
CLI input string
  -> source selection
  -> ImageSource | VideoSource | WebcamSource | RTSPSource
  -> frame read / throttling / reconnect handling
  -> YOLOEngine.predict()
  -> normalize_yolo_output()
  -> filtering pipeline
  -> detection logger
  -> render accepted detections
  -> live preview window
```

The pipeline after source selection must stay compatible with the current offline behavior so Milestone 2 filtering remains unchanged.

## Input Model

### Single Entrypoint

Keep `scripts/demo_offline.py` as the only CLI entrypoint for this milestone.

Accepted `--input` formats:

- image path
- video path
- webcam numeric index such as `0`
- webcam URI such as `webcam://0`
- RTSP URI such as `rtsp://user:pass@host/stream`

The script should classify the input string and create the correct source object.

### Source Responsibilities

`ImageSource` and `VideoSource` remain file-oriented sources.

Add:

- `WebcamSource`
- `RTSPSource`

Live-source responsibilities:

- open the underlying `cv2.VideoCapture`.
- return frames through the existing `(ret, frame)` read contract.
- enforce `max_fps` for live inputs.
- drop extra frames instead of queueing them.
- keep enough internal state to expose reconnect / failure status for preview overlays and logs.

The source layer should own reconnect and FPS logic so `scripts/demo_offline.py` stays small and testable.

## Live Source Behavior

### WebcamSource

`WebcamSource` is a lightweight wrapper around `cv2.VideoCapture(index)`.

Behavior:

- accept either an integer index or a parsed `webcam://<index>` value.
- open once and read frames continuously.
- enforce `max_fps` by dropping extra frames and only returning the newest eligible frame.
- if a read fails, surface the failure to the caller without adding RTSP-style reconnect logic in this milestone.

Webcam errors should produce clear messages for invalid indexes or capture-open failures.

### RTSPSource

`RTSPSource` wraps `cv2.VideoCapture(rtsp_url)` and owns reconnect behavior.

Behavior:

- open the RTSP stream when the source is created or explicitly opened.
- read frames continuously.
- enforce `max_fps` by dropping extra frames and only returning the newest eligible frame.
- when a read fails or the stream is not open, release the capture and begin reconnect attempts.
- retry with exponential backoff.
- use a default retry budget of `5` attempts.
- after the retry budget is exhausted, keep the process alive and mark the stream failed in preview/logging state.

The first version is single-stream only, but the reconnect behavior should remain source-local so it does not assume the whole process must exit.

## Preview Behavior

Live preview remains in `scripts/demo_offline.py` using OpenCV window display.

During RTSP reconnect attempts:

- keep the preview window open.
- continue displaying the last good frame.
- render a visible status overlay indicating reconnect state.

After RTSP retries are exhausted:

- keep the window open.
- continue displaying the last good frame.
- render a failed-stream overlay.
- avoid crashing the process automatically.

If a stream has never produced a good frame, a blank frame or status-only frame may be used internally, but the user-visible contract is a persistent preview window with clear status.

## FPS Sampling

Live inputs must honor `max_fps`.

First-version policy:

- do not queue frames.
- if frames arrive faster than `max_fps`, drop extra frames.
- only process the newest eligible frame.

This prevents backlog and keeps the filtering timestamps aligned with the actual processed stream rather than a delayed queue.

## Logging and Status

Milestone 3 must preserve the existing detection decision logging.

Additional logging for live sources should include source-lifecycle events such as:

- reconnect attempt started
- reconnect attempt succeeded
- reconnect budget exhausted

These events can be emitted through terminal logging or a lightweight status path in the source layer for this milestone. They do not replace the existing detection decision JSONL records.

## Configuration

Extend `configs/default.yaml` with live-stream settings.

Required settings:

- `inference.max_fps`
- `streaming.retry_limit`
- `streaming.initial_backoff_seconds`
- `streaming.max_backoff_seconds`
- `streaming.backoff_multiplier`

Optional settings for future compatibility:

- `streaming.open_timeout_ms`
- `streaming.read_timeout_ms`

The first implementation may apply these settings globally rather than per-source. Per-camera overrides remain later work.

## Error Handling

### File Inputs

Existing image/video behavior should remain fail-fast with clear errors.

### Webcam Inputs

Handle:

- invalid webcam index
- capture-open failure
- corrupted frame

Expected behavior:

- fail with a clear error if the webcam cannot be opened.
- skip or surface bad frames safely when possible.

### RTSP Inputs

Handle:

- initial connection failure
- dropped connection after successful frames
- repeated reconnect failure
- corrupted frame

Expected behavior:

- retry with backoff when the stream drops.
- keep preview open while reconnecting.
- after five failed reconnect attempts, mark the stream failed and keep the process alive.

## Testing

Unit tests should cover:

- webcam source opens, reads, and releases correctly.
- webcam input parsing for `0` and `webcam://0`.
- RTSP source reconnect attempts after failed reads.
- RTSP retry budget exhaustion behavior.
- FPS limiter drops extra frames and returns only the newest eligible frame.
- live-source failures do not alter the existing file-source behavior.

Integration-style tests should cover:

- `scripts/demo_offline.py` selects the correct source type from the input string.
- the existing inference/filter/render path still runs once a live source yields frames.

Tests must rely on mocked `cv2.VideoCapture` behavior so the suite does not require a real webcam or RTSP endpoint.

Existing tests must keep passing.

## Success Criteria

Milestone 3 is complete when:

- the CLI accepts webcam indexes and RTSP URLs through `--input`.
- webcam frames can be processed through the existing YOLO pipeline.
- RTSP frames can be processed through the existing YOLO pipeline.
- RTSP disconnects trigger bounded reconnect attempts with backoff.
- the preview window stays open during reconnect attempts.
- after retry exhaustion, the process stays alive and the stream is marked failed in preview/logs.
- live sources honor `max_fps` by dropping extra frames.
- the project test suite passes.

## Out of Scope

- multi-camera orchestration
- per-camera config overrides
- dashboard integration
- alarm event creation
- notification delivery
- FalseNet integration
- queue-based backpressure or asynchronous streaming workers

These remain later milestones.
