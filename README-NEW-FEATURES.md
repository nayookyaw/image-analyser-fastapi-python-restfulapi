
    Author - Nay Oo Kyaw
    Date - 2025-10-30

## How to run backend API FastAPI server
`uv sync`
`uv run uvicorn image_analyser_backend.app:app --reload`

## Overview of project Architecture
What this backend does <br>
A FastAPI service that accepts an image upload, runs object detection (YOLOv8 ONNX via ONNX Runtime, or a built-in mock), <br>
evaluates simple safety rules (proximity + PPE), and returns a structured risk assessment JSON (overall risk + detailed breaches + optional annotated preview).

## Folder Architecuture
- routers (API layer) → request/response & HTTP concerns only
- request body schemas → validate the input request body
- services (business/use-case layer) → orchestration & domain rules
- repositories/dao (data access) → persistence only
- schemas (Pydantic models) → I/O contracts (request/response/DTO)
- models (ORM entities) → database mapping
- core (config, logging, exceptions)
- db (session, migrations)
- inference (computer-vision model code) ← for Image Analyser

## Model Choice (AI model) should you choose?
- **YOLOv8-nano ONNX (+ optional INT8) is perfect.**
- Simple install, runs fast on CPU, easy to Dockerize.
- Keeps focus on architecture/rules, not CUDA setup.

If Intel CPU and care about throughput:
- **Use OpenVINO with YOLOv8.**
If NVIDIA GPU and want wow-level latency:
- **Export YOLO to TensorRT FP16.**
If need a tiny dependency footprint or edge device:
- **MobileNet-SSD / NanoDet / YOLOX-Nano (OpenCV-DNN or TFLite).**

## Model deployment
-> YOLOv8-nano → ONNX → ONNX Runtime (CPU, optional INT8)`

`uv add ultralytics onnx onnxruntime onnxruntime-tools numpy pillow`

-> Export YOLOv8-nano to ONNX
`yolo export model=yolov8n.pt format=onnx opset=12 dynamic=True imgsz=640`

**detector Workflow**
- Loads the ONNX model via ONNX Runtime
- Runs inference on CPU
- Parses YOLOv8 outputs ([x,y,w,h, conf, class_probs…])
- Applies simple NMS
- Maps class IDs → labels (COCO subset you care about)
- Falls back to mock if model missing
