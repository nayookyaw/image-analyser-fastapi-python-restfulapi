from __future__ import annotations
import io
import os
import base64
from typing import List
from PIL import Image, ImageDraw, ImageFont

from image_analyser_backend.schemas.analysis_result import Detection, RuleBreach

class ImageUtil:
    @classmethod
    def read_image_bytes(cls, data: bytes) -> Image.Image:
        return Image.open(io.BytesIO(data)).convert("RGB")

    @classmethod
    def annotate_image_base64(cls, image_bytes: bytes, detections: List[Detection], breaches: List[RuleBreach]) -> str:
        img = cls.read_image_bytes(image_bytes)
        draw = ImageDraw.Draw(img)
        font = ImageFont.load_default()

        # Map detection index -> max severity for coloring
        idx2sev: dict[int, float] = {}
        for b in breaches:
            for i in (b.involved_indices or []):
                idx2sev[i] = max(idx2sev.get(i, 0.0), b.severity)

        for i, d in enumerate(detections):
            bb = d.bbox
            box = [bb.x1, bb.y1, bb.x2, bb.y2]
            sev = idx2sev.get(i, 0.0)
            # color by severity (0=green, >0=amber, high=red)
            color = (0, 255, 0) if sev == 0 else (255, 165, 0) if sev < 2.5 else (255, 0, 0)
            draw.rectangle(box, outline=color, width=2)
            caption = f"{d.label} {d.score:.2f}"
            tw, th = draw.textbbox((0, 0), caption, font=font)[2:]
            draw.rectangle([box[0], box[1] - th - 4, box[0] + tw + 4, box[1]], fill=color)
            draw.text((box[0] + 2, box[1] - th - 2), caption, fill=(0, 0, 0), font=font)

        buf = io.BytesIO()
        img.save(buf, format="PNG")
        b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

        # save into disk for debugging purpose only
        os.makedirs("output", exist_ok=True)
        img.save("output/annotated_image.png", format="PNG")

        return f"data:image/png;base64,{b64}"