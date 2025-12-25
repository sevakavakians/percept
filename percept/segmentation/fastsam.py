"""FastSAM segmentation for PERCEPT.

FastSAM (Fast Segment Anything Model) provides class-agnostic instance
segmentation using Hailo-8 acceleration. Falls back gracefully when
hardware is not available.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

from percept.core.pipeline import PipelineModule
from percept.core.adapter import DataSpec, PipelineData
from percept.core.schema import ObjectMask


# Check for Hailo availability
def is_hailo_available() -> bool:
    """Check if Hailo runtime is available."""
    try:
        from hailo_platform import HEF, VDevice
        return True
    except ImportError:
        return False


HAILO_AVAILABLE = is_hailo_available()


@dataclass
class FastSAMConfig:
    """Configuration for FastSAM segmentation."""

    # Model settings
    model_path: str = "/usr/share/hailo-models/fast_sam_s.hef"

    # Detection thresholds
    conf_threshold: float = 0.25
    iou_threshold: float = 0.5
    max_detections: int = 20

    # Size filtering
    min_area: int = 500  # Minimum mask area in pixels
    max_area: int = 500000  # Maximum mask area


class FastSAMInference:
    """FastSAM inference on Hailo-8.

    Handles model loading, preprocessing, inference, and postprocessing
    for class-agnostic instance segmentation.
    """

    # YOLOv8 regression max (distribution focal loss)
    REG_MAX = 16

    # Strides for each detection scale
    STRIDES = [32, 16, 8]

    def __init__(self, model_path: str):
        """Initialize FastSAM inference.

        Args:
            model_path: Path to FastSAM HEF file
        """
        if not HAILO_AVAILABLE:
            raise RuntimeError("Hailo runtime not available")

        from hailo_platform import HEF, VDevice, ConfigureParams
        from hailo_platform import InputVStreamParams, OutputVStreamParams

        self.model_path = Path(model_path)
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")

        self.hef = HEF(str(self.model_path))
        self.vdevice = None
        self.network_group = None
        self._configured = False

        # Parse model info
        self.input_vstream_info = self.hef.get_input_vstream_infos()
        self.output_vstream_info = self.hef.get_output_vstream_infos()

        if self.input_vstream_info:
            info = self.input_vstream_info[0]
            self.input_shape = info.shape  # (H, W, C)
            self.input_name = info.name

        self.output_names = [info.name for info in self.output_vstream_info]
        self._preprocess_info = {}

    def configure(self):
        """Configure Hailo device for inference."""
        if self._configured:
            return

        from hailo_platform import VDevice
        from hailo_platform import InputVStreamParams, OutputVStreamParams

        self.vdevice = VDevice()
        self.network_group = self.vdevice.configure(self.hef)[0]

        self._input_vstreams_params = InputVStreamParams.make(self.network_group)
        self._output_vstreams_params = OutputVStreamParams.make(self.network_group)

        self._configured = True

    def preprocess(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for inference.

        Args:
            image: BGR image from OpenCV (H, W, 3)

        Returns:
            Preprocessed image (H, W, 3) uint8
        """
        target_h, target_w = self.input_shape[0], self.input_shape[1]
        h, w = image.shape[:2]

        # Resize maintaining aspect ratio with letterboxing
        scale = min(target_w / w, target_h / h)
        new_w, new_h = int(w * scale), int(h * scale)

        resized = cv2.resize(image, (new_w, new_h))

        # Create padded image (gray padding)
        padded = np.full((target_h, target_w, 3), 114, dtype=np.uint8)
        pad_h = (target_h - new_h) // 2
        pad_w = (target_w - new_w) // 2
        padded[pad_h:pad_h + new_h, pad_w:pad_w + new_w] = resized

        # Convert BGR to RGB
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)

        # Store preprocessing info for postprocessing
        self._preprocess_info = {
            'scale': scale,
            'pad_h': pad_h,
            'pad_w': pad_w,
            'orig_h': h,
            'orig_w': w,
        }

        return rgb.astype(np.uint8)

    def infer(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """Run inference on preprocessed image.

        Args:
            image: Preprocessed image from preprocess()

        Returns:
            Dictionary of output tensors
        """
        if not self._configured:
            self.configure()

        from hailo_platform import InferVStreams

        # Add batch dimension if needed
        if len(image.shape) == 3:
            image = np.expand_dims(image, axis=0)

        input_data = {self.input_name: image}

        with InferVStreams(
            self.network_group,
            self._input_vstreams_params,
            self._output_vstreams_params
        ) as infer_pipeline:
            with self.network_group.activate():
                results = infer_pipeline.infer(input_data)

        return results

    def _get_output(self, outputs: Dict[str, np.ndarray], key_part: str) -> Optional[np.ndarray]:
        """Get output tensor by partial key match."""
        for key, data in outputs.items():
            if key_part in key:
                return data
        return None

    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Apply softmax along axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        """Apply sigmoid activation."""
        return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))

    def _decode_boxes(
        self,
        raw_boxes: List[np.ndarray],
        image_dims: Tuple[int, int]
    ) -> np.ndarray:
        """Decode YOLOv8 distribution-based box predictions."""
        boxes = None

        for box_distribute, stride in zip(raw_boxes, self.STRIDES):
            if box_distribute.ndim == 4:
                box_distribute = box_distribute[0]

            h, w = box_distribute.shape[:2]

            # Create grid
            grid_x = np.arange(w) + 0.5
            grid_y = np.arange(h) + 0.5
            grid_x, grid_y = np.meshgrid(grid_x, grid_y)
            ct_col = grid_x.flatten() * stride
            ct_row = grid_y.flatten() * stride
            center = np.stack((ct_col, ct_row, ct_col, ct_row), axis=1)

            # Distribution to distance
            reg_range = np.arange(self.REG_MAX)
            box_distribute = np.reshape(box_distribute, (-1, 4, self.REG_MAX))
            box_distance = self._softmax(box_distribute, axis=-1)
            box_distance = np.sum(box_distance * reg_range, axis=-1)
            box_distance = box_distance * stride

            # Decode box
            box_distance = np.concatenate(
                [box_distance[:, :2] * (-1), box_distance[:, 2:]],
                axis=-1
            )
            decode_box = center + box_distance

            # Convert to xywh
            xmin = decode_box[:, 0]
            ymin = decode_box[:, 1]
            xmax = decode_box[:, 2]
            ymax = decode_box[:, 3]

            xywh_box = np.stack([
                (xmin + xmax) / 2,
                (ymin + ymax) / 2,
                xmax - xmin,
                ymax - ymin
            ], axis=1)

            boxes = xywh_box if boxes is None else np.concatenate([boxes, xywh_box], axis=0)

        return boxes

    def _xywh_to_xyxy(self, boxes: np.ndarray) -> np.ndarray:
        """Convert boxes from xywh to xyxy format."""
        result = np.copy(boxes)
        result[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        result[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        result[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        result[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        return result

    def _crop_mask(self, masks: np.ndarray, boxes: np.ndarray) -> np.ndarray:
        """Zero out mask regions outside bbox."""
        n_masks, h, w = masks.shape
        boxes_int = np.ceil(boxes).astype(int)
        boxes_int = np.clip(boxes_int, 0, None)

        for k in range(n_masks):
            x1, y1, x2, y2 = boxes_int[k]
            x2 = min(x2, w)
            y2 = min(y2, h)
            masks[k, :y1, :] = 0
            masks[k, y2:, :] = 0
            masks[k, :, :x1] = 0
            masks[k, :, x2:] = 0

        return masks

    def _process_masks(
        self,
        protos: np.ndarray,
        mask_coeffs: np.ndarray,
        boxes: np.ndarray,
        img_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Generate instance masks from prototypes and coefficients."""
        mh, mw, c = protos.shape
        ih, iw = img_shape

        # Generate masks: coeffs @ protos.T -> sigmoid
        masks = self._sigmoid(mask_coeffs @ protos.reshape(-1, c).T)
        masks = masks.reshape(-1, mh, mw)

        # Crop masks to bbox
        downsampled_boxes = boxes.copy()
        downsampled_boxes[:, 0] *= mw / iw
        downsampled_boxes[:, 2] *= mw / iw
        downsampled_boxes[:, 1] *= mh / ih
        downsampled_boxes[:, 3] *= mh / ih
        masks = self._crop_mask(masks, downsampled_boxes)

        # Upsample to image size
        if masks.shape[0] > 0:
            masks = np.transpose(masks, (1, 2, 0))  # HWN
            masks = cv2.resize(masks, (iw, ih), interpolation=cv2.INTER_LINEAR)
            if masks.ndim == 2:
                masks = masks[..., np.newaxis]
            masks = np.transpose(masks, (2, 0, 1))  # NHW

        return masks

    def postprocess(
        self,
        outputs: Dict[str, np.ndarray],
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_detections: int = 20,
    ) -> List[Dict[str, Any]]:
        """Postprocess FastSAM outputs to get instance masks.

        Args:
            outputs: Raw model outputs
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            max_detections: Maximum detections

        Returns:
            List of dicts with mask, bbox, confidence, area
        """
        info = self._preprocess_info
        input_h, input_w = self.input_shape[0], self.input_shape[1]

        # Get outputs by layer name (scale-specific)
        raw_boxes = [
            self._get_output(outputs, 'conv74'),  # 20x20x64
            self._get_output(outputs, 'conv61'),  # 40x40x64
            self._get_output(outputs, 'conv45'),  # 80x80x64
        ]
        raw_scores = [
            self._get_output(outputs, 'conv75'),  # 20x20x1
            self._get_output(outputs, 'conv62'),  # 40x40x1
            self._get_output(outputs, 'conv46'),  # 80x80x1
        ]
        raw_coeffs = [
            self._get_output(outputs, 'conv73'),  # 20x20x32
            self._get_output(outputs, 'conv60'),  # 40x40x32
            self._get_output(outputs, 'conv44'),  # 80x80x32
        ]
        protos = self._get_output(outputs, 'conv48')  # 160x160x32

        # Check all outputs exist
        if any(x is None for x in raw_boxes + raw_scores + raw_coeffs) or protos is None:
            return []

        # Remove batch dimensions
        raw_boxes = [x[0].astype(np.float32) if x.ndim == 4 else x.astype(np.float32) for x in raw_boxes]
        raw_scores = [x[0].astype(np.float32) if x.ndim == 4 else x.astype(np.float32) for x in raw_scores]
        raw_coeffs = [x[0].astype(np.float32) if x.ndim == 4 else x.astype(np.float32) for x in raw_coeffs]
        protos = protos[0].astype(np.float32) if protos.ndim == 4 else protos.astype(np.float32)

        # Decode boxes
        decoded_boxes = self._decode_boxes(raw_boxes, (input_h, input_w))

        # Flatten scores and coefficients
        scores = np.concatenate([s.reshape(-1, 1) for s in raw_scores], axis=0)
        coeffs = np.concatenate([c.reshape(-1, 32) for c in raw_coeffs], axis=0)

        scores = self._sigmoid(scores).squeeze(-1)

        # Filter by confidence
        keep = scores > conf_threshold
        if not np.any(keep):
            return []

        filtered_boxes = decoded_boxes[keep]
        filtered_scores = scores[keep]
        filtered_coeffs = coeffs[keep]

        # Convert to xyxy for NMS
        boxes_xyxy = self._xywh_to_xyxy(filtered_boxes)

        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            filtered_boxes.tolist(),
            filtered_scores.tolist(),
            conf_threshold,
            iou_threshold,
        )

        if len(indices) == 0:
            return []

        # Flatten indices
        if isinstance(indices, np.ndarray):
            indices = indices.flatten()
        else:
            indices = [i[0] if isinstance(i, (list, tuple)) else i for i in indices]

        indices = indices[:max_detections]

        # Get final detections
        final_boxes = boxes_xyxy[indices]
        final_scores = filtered_scores[indices]
        final_coeffs = filtered_coeffs[indices]

        # Generate masks
        masks = self._process_masks(protos, final_coeffs, final_boxes, (input_h, input_w))

        # Build results
        results = []
        for i in range(len(indices)):
            # Binarize mask
            mask = (masks[i] > 0.5).astype(np.uint8) * 255

            # Scale mask to original size
            mask_orig = cv2.resize(mask, (info['orig_w'], info['orig_h']))

            # Scale bbox to original size
            bbox = final_boxes[i].copy()
            bbox[0] = (bbox[0] - info['pad_w']) / info['scale']
            bbox[1] = (bbox[1] - info['pad_h']) / info['scale']
            bbox[2] = (bbox[2] - info['pad_w']) / info['scale']
            bbox[3] = (bbox[3] - info['pad_h']) / info['scale']

            # Clamp to image bounds
            bbox[0] = max(0, min(info['orig_w'], bbox[0]))
            bbox[1] = max(0, min(info['orig_h'], bbox[1]))
            bbox[2] = max(0, min(info['orig_w'], bbox[2]))
            bbox[3] = max(0, min(info['orig_h'], bbox[3]))

            area = int((mask_orig > 0).sum())

            results.append({
                'mask': mask_orig,
                'bbox': [int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])],
                'confidence': float(final_scores[i]),
                'area': area,
            })

        # Sort by area (larger first)
        results.sort(key=lambda x: x['area'], reverse=True)

        return results

    def segment(
        self,
        image: np.ndarray,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5,
        max_detections: int = 20,
    ) -> List[Dict[str, Any]]:
        """Run full segmentation pipeline.

        Args:
            image: BGR image from OpenCV
            conf_threshold: Confidence threshold
            iou_threshold: NMS IoU threshold
            max_detections: Maximum detections

        Returns:
            List of dicts with mask, bbox, confidence, area
        """
        preprocessed = self.preprocess(image)
        outputs = self.infer(preprocessed)
        return self.postprocess(outputs, conf_threshold, iou_threshold, max_detections)

    def release(self):
        """Release Hailo device resources."""
        if self.vdevice:
            self.vdevice = None
        self._configured = False

    def __enter__(self):
        self.configure()
        return self

    def __exit__(self, *args):
        self.release()


class FastSAMSegmenter(PipelineModule):
    """Pipeline module for FastSAM instance segmentation.

    Uses Hailo-8 acceleration when available, otherwise returns empty results.
    """

    def __init__(self, config: Optional[FastSAMConfig] = None):
        """Initialize segmenter.

        Args:
            config: Configuration options
        """
        self.config = config or FastSAMConfig()
        self._inference = None
        self._hailo_available = HAILO_AVAILABLE

        if self._hailo_available:
            try:
                self._inference = FastSAMInference(self.config.model_path)
            except Exception as e:
                self._hailo_available = False

    @property
    def name(self) -> str:
        return "fastsam_segmenter"

    @property
    def input_spec(self) -> DataSpec:
        return DataSpec(
            data_type="image",
            dtype="uint8",
            required_fields=["image"],
        )

    @property
    def output_spec(self) -> DataSpec:
        return DataSpec(
            data_type="masks",
            required_fields=["masks"],
        )

    def process(self, data: PipelineData) -> PipelineData:
        """Segment objects using FastSAM.

        Args:
            data: PipelineData with 'image' field (BGR)

        Returns:
            PipelineData with 'masks' field
        """
        image = data.image
        depth = data.get("depth")

        result = data.copy()
        result.masks = []

        if not self._hailo_available or self._inference is None:
            return result

        # Run segmentation
        try:
            detections = self._inference.segment(
                image,
                conf_threshold=self.config.conf_threshold,
                iou_threshold=self.config.iou_threshold,
                max_detections=self.config.max_detections,
            )
        except Exception:
            return result

        # Convert to ObjectMask instances
        masks = []
        for det in detections:
            area = det['area']

            # Filter by area
            if area < self.config.min_area or area > self.config.max_area:
                continue

            # Compute median depth if available
            depth_median = 0.0
            if depth is not None:
                mask_bool = det['mask'] > 0
                region_depth = depth[mask_bool]
                valid_depth = region_depth[region_depth > 0]
                if len(valid_depth) > 0:
                    depth_median = float(np.median(valid_depth))

            masks.append(ObjectMask(
                mask=det['mask'],
                bbox=tuple(det['bbox']),
                confidence=det['confidence'],
                depth_median=depth_median,
                point_count=area,
            ))

        result.masks = masks
        return result

    def is_available(self) -> bool:
        """Check if FastSAM inference is available."""
        return self._hailo_available and self._inference is not None


def segment_with_fastsam(
    image: np.ndarray,
    model_path: Optional[str] = None,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.5,
) -> List[ObjectMask]:
    """Convenience function for FastSAM segmentation.

    Args:
        image: BGR image
        model_path: Path to FastSAM HEF model
        conf_threshold: Confidence threshold
        iou_threshold: NMS IoU threshold

    Returns:
        List of ObjectMask instances
    """
    if not HAILO_AVAILABLE:
        return []

    config = FastSAMConfig(
        conf_threshold=conf_threshold,
        iou_threshold=iou_threshold,
    )
    if model_path:
        config.model_path = model_path

    segmenter = FastSAMSegmenter(config)
    data = PipelineData(image=image)
    result = segmenter.process(data)

    return result.masks
