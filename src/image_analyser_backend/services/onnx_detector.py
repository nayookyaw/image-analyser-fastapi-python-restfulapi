import os
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Sequence

import numpy as np
import onnxruntime as ort
from PIL import Image

# -----------------------------
# Data structures
# -----------------------------
@dataclass
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    score: float
    class_id: int
    label: str

@dataclass
class InferenceResult:
    detections: List[Detection]
    width: int
    height: int
    inference_ms: float
    preprocess_ms: float
    postprocess_ms: float
    model_name: str
    provider: str


# -----------------------------
# COCO 80 labels (Ultralytics order)
# -----------------------------
COCO80 = [
    "person","bicycle","car","motorcycle","airplane","bus","train","truck","boat","traffic light",
    "fire hydrant","stop sign","parking meter","bench","bird","cat","dog","horse","sheep","cow",
    "elephant","bear","zebra","giraffe","backpack","umbrella","handbag","tie","suitcase","frisbee",
    "skis","snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard",
    "tennis racket","bottle","wine glass","cup","fork","knife","spoon","bowl","banana","apple",
    "sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair","couch",
    "potted plant","bed","dining table","toilet","tv","laptop","mouse","remote","keyboard","cell phone",
    "microwave","oven","toaster","sink","refrigerator","book","clock","vase","scissors","teddy bear",
    "hair drier","toothbrush"
]


# -----------------------------
# Utility: letterbox resize (keeps aspect ratio; pads to multiple of stride)
# -----------------------------
def letterbox(
    im: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    stride: int = 32
) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize and pad image to fit new_shape with stride-multiple constraints."""
    shape = im.shape[:2]  # (h, w)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    new_unpad = (int(round(shape[1] * r)), int(round(shape[0] * r)))  # (w, h)

    # Compute padding
    dw = new_shape[1] - new_unpad[0]  # width padding
    dh = new_shape[0] - new_unpad[1]  # height padding

    # resize
    if shape[::-1] != new_unpad:
        # Pillow >= 9.1 provides the Resampling enum; older versions expose BILINEAR on Image.
        # Use a safe fallback to support both versions and avoid attribute errors from type checkers.
        try:
            resample = Image.Resampling.BILINEAR  # type: ignore[attr-defined]
        except Exception:
            resample = Image.BILINEAR  # type: ignore[attr-defined]
        im = np.array(Image.fromarray(im).resize(new_unpad, resample=resample))

    # padding: add border
    top = dh // 2
    bottom = dh - top
    left = dw // 2
    right = dw - left

    # pad
    # im = np.pad(im, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=color)
    im = np.pad(im, ((top, bottom), (left, right), (0, 0)), mode="constant", constant_values=114)

    return im, r, (left, top)  # return padded image, scale, and top-left pad


# -----------------------------
# NMS (class-aware)
# -----------------------------
def nms(
    boxes: np.ndarray,  # (N,4) in xyxy
    scores: np.ndarray, # (N,)
    class_ids: np.ndarray, # (N,)
    iou_thresh: float = 0.45,
    max_dets: int = 300
) -> List[int]:
    """Return indices of boxes to keep (class-aware NMS)."""
    keep_indices: List[int] = []
    if len(boxes) == 0:
        return keep_indices

    # Process per-class to keep things simple and predictable
    classes = np.unique(class_ids)
    for c in classes:
        idxs = np.where(class_ids == c)[0]
        b = boxes[idxs]
        s = scores[idxs]

        order = np.argsort(-s)
        idxs = idxs[order]
        b = b[order]

        while idxs.size > 0:
            i = idxs[0]
            keep_indices.append(i)
            if len(keep_indices) >= max_dets:
                break

            if idxs.size == 1:
                break

            ious = _iou(b[0], b[1:])
            remain = np.where(ious <= iou_thresh)[0] + 1
            idxs = idxs[remain]
            b = b[remain]

    return keep_indices[:max_dets]

def _iou(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    """IoU of a box with many boxes, all in xyxy."""
    # box: (4,), boxes: (M,4)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])

    inter = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    area1 = (box[2] - box[0]) * (box[3] - box[1])
    area2 = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area1 + area2 - inter + 1e-7
    return inter / union


# -----------------------------
# Detector
# -----------------------------
class OnnxYoloV8Detector:
    """
    Production-ready ONNX Runtime detector for YOLOv8 (Ultralytics export).
    - Supports dynamic input shape (e.g. export with imgsz=640 dynamic=True)
    - CPU by default; pass a providers list to change EP (e.g., ["CUDAExecutionProvider","CPUExecutionProvider"])
    - Graceful fallback to mock detections if the model file is missing
    """

    def __init__(
        self,
        model_path: str,
        class_names: Optional[Sequence[str]] = None,
        providers: Optional[List[str]] = None,
        input_size: Tuple[int, int] = (640, 640),
        conf_threshold: float = 0.25,
        nms_iou_threshold: float = 0.45,
        max_dets: int = 300,
        prefer_sigmoid: bool = True,
        fallback_mock: bool = True,
    ):
        self.model_path = model_path
        self.class_names = list(class_names) if class_names is not None else COCO80
        self.input_size = input_size
        self.conf_threshold = conf_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_dets = max_dets
        self.prefer_sigmoid = prefer_sigmoid
        self.fallback_mock = fallback_mock

        self._session: Optional[ort.InferenceSession] = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None
        self._provider = "CPUExecutionProvider"

        if os.path.isfile(self.model_path):
            sess_opts = ort.SessionOptions()
            # You can tune intra_op_num_threads here for CPU perf
            # sess_opts.intra_op_num_threads = os.cpu_count() or 1

            if providers is None:
                providers = ["CPUExecutionProvider"]
            self._session = ort.InferenceSession(self.model_path, sess_options=sess_opts, providers=providers)
            self._provider = self._session.get_providers()[0] if self._session.get_providers() else "Unknown"

            # cache IO names
            self._input_name = self._session.get_inputs()[0].name
            self._output_name = self._session.get_outputs()[0].name

            # warmup
            self._warmup()
        else:
            if not self.fallback_mock:
                raise FileNotFoundError(f"Model not found: {self.model_path} (and fallback_mock=False)")

    def _warmup(self):
        dummy = np.zeros((1, 3, self.input_size[0], self.input_size[1]), dtype=np.float32)
        if self._session is not None:
            self._session.run([self._output_name], {self._input_name: dummy})

    def _preprocess(self, img: Image.Image, size: Tuple[int, int]) -> Tuple[np.ndarray, np.ndarray, float, Tuple[float, float]]:
        t0 = time.time()
        im0 = np.asarray(img.convert("RGB"))
        padded, scale, pad = letterbox(im0, new_shape=size, stride=32)
        # HWC -> CHW, [0,1]
        x = padded.transpose(2, 0, 1).astype(np.float32) / 255.0
        x = np.expand_dims(x, 0)  # (1,3,H,W)
        return x, im0, scale, pad

    def _postprocess(
        self,
        preds: np.ndarray,
        scale: float,
        pad: Tuple[float, float],
        orig_wh: Tuple[int, int],
    ) -> Tuple[List[Detection], float]:
        """
        Handle typical Ultralytics YOLOv8 ONNX outputs.
        - Common shapes:
          (1, 84, N) or (1, N, 84)
        - First 4 entries are boxes in xywh (center format) in the model input space
        - Remaining are class scores (already sigmoid in most exports). We optionally apply sigmoid if values look like logits.
        """
        t0 = time.time()
        if preds.ndim == 3:
            preds = preds[0]  # (84,N) or (N,84)
        elif preds.ndim == 2:
            pass
        else:
            raise ValueError(f"Unexpected preds shape: {preds.shape}")

        # Normalize to (N, 4 + C)
        if preds.shape[0] in (84, 85) and preds.shape[0] > preds.shape[1]:
            preds = preds.transpose(1, 0)  # (N,84/85)

        # Now preds is (N, 4 + C)
        if preds.shape[1] < 5:
            raise ValueError(f"Unexpected number of columns in preds: {preds.shape}")

        boxes_xywh = preds[:, :4]
        class_scores = preds[:, 4:]

        # Many exports already have sigmoid applied; if we detect logits (values outside [0,1]), apply sigmoid.
        if self.prefer_sigmoid and (class_scores.max() > 1.0 or class_scores.min() < 0.0):
            class_scores = 1.0 / (1.0 + np.exp(-class_scores))

        scores = class_scores.max(axis=1)
        class_ids = class_scores.argmax(axis=1)

        # Threshold by confidence
        keep = scores >= self.conf_threshold
        if not np.any(keep):
            return [], (time.time() - t0) * 1000.0
        boxes_xywh = boxes_xywh[keep]
        scores = scores[keep]
        class_ids = class_ids[keep]

        # Convert xywh (center) -> xyxy in padded/letterboxed space
        xyxy = _xywh_to_xyxy(boxes_xywh)

        # Map back to original image coordinates (remove pad, divide by scale)
        # padded coords: (x - pad_x, y - pad_y) / scale
        pad_x, pad_y = pad
        xyxy[:, [0, 2]] -= pad_x
        xyxy[:, [1, 3]] -= pad_y
        xyxy /= (scale + 1e-9)

        # Clip to original size
        w, h = orig_wh
        xyxy[:, 0::2] = np.clip(xyxy[:, 0::2], 0, w - 1)
        xyxy[:, 1::2] = np.clip(xyxy[:, 1::2], 0, h - 1)

        # Run NMS (class-aware)
        keep_idx = nms(xyxy, scores, class_ids, iou_thresh=self.nms_iou_threshold, max_dets=self.max_dets)
        xyxy = xyxy[keep_idx]
        scores = scores[keep_idx]
        class_ids = class_ids[keep_idx]

        detections: List[Detection] = []
        for i in range(len(keep_idx)):
            cid = int(class_ids[i])
            label = self.class_names[cid] if 0 <= cid < len(self.class_names) else str(cid)
            x1, y1, x2, y2 = xyxy[i].tolist()
            detections.append(Detection(x1, y1, x2, y2, float(scores[i]), cid, label))

        return detections, (time.time() - t0) * 1000.0

    def infer(
        self,
        image: Image.Image,
        size: Optional[Tuple[int, int]] = None
    ) -> InferenceResult:
        """Run end-to-end inference on a PIL image."""
        size = size or self.input_size

        # If no model session (file missing) and fallback enabled -> mock detections
        if self._session is None:
            return self._mock_result(image)

        t_pre = time.time()
        x, im0, scale, pad = self._preprocess(image, size)
        preprocess_ms = (time.time() - t_pre) * 1000.0

        t0 = time.time()
        preds = self._session.run([self._output_name], {self._input_name: x})[0]
        inference_ms = (time.time() - t0) * 1000.0

        # Ensure preds is a numpy ndarray
        if not isinstance(preds, np.ndarray):
            preds = np.array(preds)
        detections, post_ms = self._postprocess(preds, scale, pad, (im0.shape[1], im0.shape[0]))

        return InferenceResult(
            detections=detections,
            width=im0.shape[1],
            height=im0.shape[0],
            inference_ms=inference_ms,
            preprocess_ms=preprocess_ms,
            postprocess_ms=post_ms,
            model_name=os.path.basename(self.model_path),
            provider=self._provider
        )

    def _mock_result(self, image: Image.Image) -> InferenceResult:
        """Return a stable fake result for development when the model file is missing."""
        w, h = image.size
        # simple centered box
        cx, cy = w * 0.5, h * 0.5
        bw, bh = w * 0.3, h * 0.3
        x1, y1 = cx - bw / 2, cy - bh / 2
        x2, y2 = cx + bw / 2, cy + bh / 2
        det = Detection(x1, y1, x2, y2, 0.90, 0, self.class_names[0] if self.class_names else "obj")
        return InferenceResult(
            detections=[det],
            width=w,
            height=h,
            inference_ms=0.1,
            preprocess_ms=0.1,
            postprocess_ms=0.1,
            model_name=f"mock_model {os.path.basename(self.model_path)}",
            provider="mock"
        )


# -----------------------------
# Helpers
# -----------------------------
def _xywh_to_xyxy(xywh: np.ndarray) -> np.ndarray:
    """Convert (cx,cy,w,h) -> (x1,y1,x2,y2)."""
    xyxy = np.empty_like(xywh)
    xyxy[:, 0] = xywh[:, 0] - xywh[:, 2] / 2.0
    xyxy[:, 1] = xywh[:, 1] - xywh[:, 3] / 2.0
    xyxy[:, 2] = xywh[:, 0] + xywh[:, 2] / 2.0
    xyxy[:, 3] = xywh[:, 1] + xywh[:, 3] / 2.0
    return xyxy


# -----------------------------
# CLI demo/testing
# if __name__ == "__main__":
#   xxxxx
# -----------------------------