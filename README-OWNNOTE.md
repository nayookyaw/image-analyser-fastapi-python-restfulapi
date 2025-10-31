uv 0.9.5

-> Export FASTER model [Speed boost: dynamic INT8 quantization]
* This keeps accuracy decent and speeds up CPU inference

python - <<EOF
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(
    model_input='src/image_analyser_backend/models/yolov8n.onnx',
    model_output='src/image_analyser_backend/models/yolov8n-int8.onnx',
    weight_type=QuantType.QInt8
)
print('Wrote INT8 model → src/.../yolov8n-int8.onnx')
EOF 

## DI (request.app.state.detector/request.app.state.rules)

Feature	Module-Level Singleton	Depends(get_detector)
Testable	                ❌ Hard to override	✅ Easily mockable
Lifecycle-safe	            ❌ Risky in reloads	✅ Tied to app startup
Clean DI in routes	        ❌ Manual wiring	    ✅ Declarative
Works with FastAPI tools	❌ No	            ✅ Yes (e.g. overrides)

## Validation request body
| Library         | Speed                              | Ergonomics     | JSON Schema       | Works smoothly with FastAPI    |
| --------------- | ---------------------------------- | -------------- | ----------------  | -----------------------------  |
| **msgspec**     | 🥇 Fastest                         | Minimal, typed | ❌ (not built-in) | ⚠️ Manual wiring (not native)  |
| **Pydantic v2** | 🥈 Very fast (much faster than v1) | Great DX       | ✅                | ✅ First-class                 |

## Prepare for mock model for unit testing
fallback_mock: bool = True

## Having rule parameters outside the codebase allows:
- Non-developers to tune rules
- Instant updates without restart (reload config)
- Environment-specific deployments
- Version control of rule policies
- Faster compliance updates

## Unit testing dependencies
`uv add --dev httpx pytest`

# Run unit testing
`uv run pytest -q`

| Package                            | Purpose in Project         | Why Need It                                                                   |
| ---------------------------------- | -------------------------- | ----------------------------------------------------------------------------- |
| **pillow**                         | Image processing           | Opens uploaded images & converts to arrays + annotated output                 |
| **pydantic**                       | Data modeling & validation | Defines schemas like `AnalysisResult`, ensures input/output types are correct |
| **python-multipart**               | File upload handling       | Enables reading `UploadFile` for images                                       |
| **numpy**                          | Numeric processing         | Used for YOLO preprocessing (letterbox, array ops)                            |
| **pyyaml**                         | Load YAML settings         | Loads rule engine tuning values from config files                             |
| **pytest** *(dev)*                 | Testing framework          | Ensure correctness with automated tests                                       |
| **pydantic-settings**              | Config management          | Load model/config paths from `.env` or settings class                         |



| Where                                | Variable                      | Effect                                                                    |
| ------------------------------------ | ----------------------------- | ------------------------------------------------------------------------- |
| **settings (configurable via .env)** | `CONFIDENCE_THRESHOLD`        | YOLO: how sure a class must be to count as detection                      |
|                                      | `IOU_THRESHOLD`               | YOLO: overlap allowed for NMS filtering out duplicate boxes               |
|                                      | `DEFAULT_PROXIMITY_THRESHOLD` | Rule: how close (in pixels) a person & forklift must be to trigger a rule |
|                                      | `DEFAULT_RULE_WEIGHTS`        | Rule: how important proximity vs PPE violations are                       |
|                                      | `DEFAULT_RULE_CAPS`           | Rule: max severity each rule can cause (0–10 scale)                       |
|                                      | `DEFAULT_MISSING_PPE_LABELS`  | Rule: which labels count as PPE (helmet/vest)                             |

* Note:
CONFIDENCE_THRESHOLD = 0.7 → ignore all low-confidence detections
IOU_THRESHOLD = 0.3 → more strict merging of overlapping boxes

* proximity_threshold 
📍 What it means:
    Minimum pixel distance allowed between a person and hazard (like a forklift).
    If distance < this threshold → 🚨 breach triggered
📍 Effect:
    Lower value → must be very close before triggering
    Higher value → more safety conscious → triggers earlier
📍 Where used in code: _proximity_rule()
    if center_distance < config.proximity_threshold:
        severity = (threshold - distance) / threshold * cap

* missing_ppe_labels
📍 What it means:
    Labels considered required PPE for detected persons.
    Default:
    ["helmet", "vest"]
📍 Use case:
    If a person is detected but none of these PPE labels exist in detections → 🚨 PPE breach
📍 Where used: _ppe_rule()
    Checks if any PPE detection intersects with each person detection.

✅ What is cap?
    Cap = Maximum allowed severity for a rule.
    It prevents a single rule from pushing the total risk score beyond a reasonable limit.

    So even if the situation is extremely dangerous (e.g., person touching forklift),
    the severity under that rule cannot go infinite — it caps out at a maximum.
✅ Where is it applied in code?
Inside _proximity_rule() in rule_engine.py



| Technology                         | Speed (Edge CPU)  | Accuracy           | Model Size             | Training Difficulty | Deployment Ease   | Best Use Case                                              |
| ---------------------------------- | ----------------- | ------------------ | ---------------------- | ------------------- | ----------------- | ---------------------------------------------------------- |
| **YOLOv8n ONNX** ✅                 | ⭐⭐⭐⭐⭐ **Fastest** | Medium             | ⚡ Very small (~3–7 MB) | Easy                | ⭐⭐⭐⭐⭐ **Easiest** | Real-time detection on CPU/IoT/mobile, API microservices   |
| YOLOv8-L / X                       | Medium            | ⭐⭐⭐⭐⭐ High         | Large (40–120 MB)      | Easy                | High              | High-accuracy camera analytics, GPU clusters               |
| **Detectron2**                     | Slow (CPU)        | ⭐⭐⭐⭐⭐ **Superior** | Large                  | Hard                | Medium            | Research, segmentation, keypoints, highest accuracy        |
| **Transformers (DETR, RT-DETR)**   | Medium-Slow       | ⭐⭐⭐⭐⭐ Advanced     | Large                  | Harder              | Medium            | Vision-language tasks, high-precision structured detection |
| **OpenCV DNN / Classic**           | Very fast         | Low                | Tiny                   | Easy                | Very easy         | Simple or legacy detection tasks (non-AI)                  |
| **TensorFlow** (SSD, EfficientDet) | Medium            | High               | Medium                 | Medium              | Medium            | Cloud + mobile Android Edge TPU                            |
| **NVIDIA-Triton + API**            | GPU Fast          | Very High          | Medium                 | Medium              | ⭐⭐⭐⭐⭐ Enterprise  | High-scale inference server deployments                    |
