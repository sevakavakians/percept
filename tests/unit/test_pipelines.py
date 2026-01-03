"""Unit tests for PERCEPT classification pipeline modules."""

import numpy as np
import pytest

from percept.core.schema import ObjectSchema, Detection
from percept.pipelines.router import (
    PipelineType,
    RouterConfig,
    RoutingDecision,
    PipelineRouter,
    RouterModule,
    get_pipeline_for_class,
    PERSON_CLASSES,
    VEHICLE_CLASSES,
)
from percept.pipelines.person import (
    Posture,
    PersonPipelineConfig,
    Keypoint,
    PoseResult,
    ClothingResult,
    PoseEstimator,
    ClothingAnalyzer,
    FaceDetector,
    PersonPipeline,
    PersonPipelineModule,
    KEYPOINT_NAMES,
    KEYPOINT_INDICES,
)
from percept.pipelines.vehicle import (
    VehicleType,
    VehicleOrientation,
    VehiclePipelineConfig,
    VehicleColorResult,
    VehicleTypeResult,
    LicensePlateResult,
    VehicleColorAnalyzer,
    VehicleTypeClassifier,
    LicensePlateDetector,
    VehiclePipeline,
    VehiclePipelineModule,
)
from percept.pipelines.generic import (
    GenericPipelineConfig,
    ClassificationResult,
    ColorResult,
    ShapeResult,
    ImageNetClassifier,
    ColorAnalyzer,
    ShapeAnalyzer,
    SizeEstimator,
    GenericPipeline,
    GenericPipelineModule,
)


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def sample_image():
    """Create a sample BGR image."""
    np.random.seed(42)
    return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)


@pytest.fixture
def sample_mask():
    """Create a sample binary mask."""
    mask = np.zeros((480, 640), dtype=np.uint8)
    mask[100:400, 150:500] = 255
    return mask


@pytest.fixture
def person_crop():
    """Create a sample person crop (tall rectangle)."""
    np.random.seed(42)
    return np.random.randint(0, 255, (300, 150, 3), dtype=np.uint8)


@pytest.fixture
def vehicle_crop():
    """Create a sample vehicle crop (wide rectangle)."""
    np.random.seed(42)
    return np.random.randint(0, 255, (150, 300, 3), dtype=np.uint8)


@pytest.fixture
def sample_detection():
    """Create a sample detection."""
    return Detection(
        class_id=0,
        class_name="person",
        confidence=0.9,
        bbox=(100, 100, 200, 200),
    )


@pytest.fixture
def sample_object():
    """Create a sample ObjectSchema."""
    obj = ObjectSchema(
        primary_class="unknown",
        confidence=0.9,
        bounding_box_2d=(100, 100, 250, 350),
    )
    return obj


# =============================================================================
# Router Tests
# =============================================================================


class TestPipelineType:
    """Test PipelineType enum."""

    def test_pipeline_types(self):
        assert PipelineType.PERSON.value == "person"
        assert PipelineType.VEHICLE.value == "vehicle"
        assert PipelineType.GENERIC.value == "generic"
        assert PipelineType.FACE.value == "face"


class TestRouterConfig:
    """Test RouterConfig dataclass."""

    def test_default_config(self):
        config = RouterConfig()
        assert config.default_pipeline == "generic"
        assert config.enable_person_pipeline == True
        assert config.min_routing_confidence == 0.3

    def test_custom_config(self):
        config = RouterConfig(min_routing_confidence=0.5)
        assert config.min_routing_confidence == 0.5


class TestPipelineRouter:
    """Test PipelineRouter."""

    def test_route_person(self, sample_mask):
        router = PipelineRouter()
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.9,
            bbox=(100, 100, 200, 300),
        )

        decision = router.route(detection, sample_mask)
        assert decision.pipeline_type == PipelineType.PERSON
        assert decision.confidence == 0.9

    def test_route_vehicle(self, sample_mask):
        router = PipelineRouter()
        detection = Detection(
            class_id=2,
            class_name="car",
            confidence=0.85,
            bbox=(100, 100, 300, 250),
        )

        decision = router.route(detection, sample_mask)
        assert decision.pipeline_type == PipelineType.VEHICLE

    def test_route_generic(self, sample_mask):
        router = PipelineRouter()
        detection = Detection(
            class_id=56,
            class_name="chair",
            confidence=0.8,
            bbox=(100, 100, 200, 200),
        )

        decision = router.route(detection, sample_mask)
        assert decision.pipeline_type == PipelineType.GENERIC

    def test_route_low_confidence(self, sample_mask):
        router = PipelineRouter()
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=0.2,
            bbox=(100, 100, 200, 200),
        )

        decision = router.route(detection, sample_mask)
        assert decision.pipeline_type == PipelineType.GENERIC
        assert decision.reason == "confidence_below_threshold"

    def test_route_batch(self):
        router = PipelineRouter()
        detections = [
            Detection(class_id=0, class_name="person", confidence=0.9, bbox=(100, 100, 200, 300)),
            Detection(class_id=2, class_name="car", confidence=0.8, bbox=(300, 100, 500, 200)),
            Detection(class_id=99, class_name="unknown", confidence=0.7, bbox=(500, 100, 600, 200)),
        ]

        decisions = router.route_batch(detections)
        assert len(decisions) == 3
        assert decisions[0].pipeline_type == PipelineType.PERSON
        assert decisions[1].pipeline_type == PipelineType.VEHICLE
        assert decisions[2].pipeline_type == PipelineType.GENERIC

    def test_register_pipeline(self):
        router = PipelineRouter()
        mock_pipeline = object()
        router.register_pipeline(PipelineType.PERSON, mock_pipeline)

        retrieved = router.get_pipeline(PipelineType.PERSON)
        assert retrieved is mock_pipeline

    def test_get_stats(self, sample_mask):
        router = PipelineRouter()
        router.reset_stats()

        for _ in range(3):
            router.route(Detection(class_id=0, class_name="person", confidence=0.9, bbox=(0, 0, 100, 200)), sample_mask)

        stats = router.get_stats()
        assert stats.get("person", 0) == 3


class TestRouterModule:
    """Test RouterModule pipeline integration."""

    def test_module_properties(self):
        module = RouterModule()
        assert module.name == "router"
        assert module.input_spec.data_type == "detections"
        assert module.output_spec.data_type == "routed_objects"


class TestGetPipelineForClass:
    """Test convenience function."""

    def test_person_classes(self):
        assert get_pipeline_for_class("person") == PipelineType.PERSON
        assert get_pipeline_for_class("pedestrian") == PipelineType.PERSON

    def test_vehicle_classes(self):
        assert get_pipeline_for_class("car") == PipelineType.VEHICLE
        assert get_pipeline_for_class("truck") == PipelineType.VEHICLE

    def test_generic_fallback(self):
        assert get_pipeline_for_class("chair") == PipelineType.GENERIC
        assert get_pipeline_for_class("unknown") == PipelineType.GENERIC


# =============================================================================
# Person Pipeline Tests
# =============================================================================


class TestPosture:
    """Test Posture enum."""

    def test_postures(self):
        assert Posture.STANDING.value == "standing"
        assert Posture.SITTING.value == "sitting"
        assert Posture.LYING.value == "lying"


class TestKeypoint:
    """Test Keypoint dataclass."""

    def test_create_keypoint(self):
        kp = Keypoint(x=100.5, y=200.5, confidence=0.9, name="nose")
        assert kp.x == 100.5
        assert kp.name == "nose"

    def test_is_visible(self):
        kp_visible = Keypoint(x=100, y=200, confidence=0.8, name="nose")
        kp_hidden = Keypoint(x=100, y=200, confidence=0.1, name="nose")

        assert kp_visible.is_visible(threshold=0.3) == True
        assert kp_hidden.is_visible(threshold=0.3) == False


class TestPoseResult:
    """Test PoseResult dataclass."""

    def test_create_pose_result(self):
        keypoints = [
            Keypoint(x=100, y=50, confidence=0.9, name="nose"),
            Keypoint(x=95, y=55, confidence=0.8, name="left_eye"),
        ]
        pose = PoseResult(
            keypoints=keypoints,
            bbox=(50, 20, 150, 300),
            confidence=0.9,
            posture=Posture.STANDING,
        )

        assert pose.confidence == 0.9
        assert pose.posture == Posture.STANDING

    def test_get_keypoint(self):
        keypoints = [Keypoint(x=100, y=50, confidence=0.9, name=name)
                     for name in KEYPOINT_NAMES]
        pose = PoseResult(
            keypoints=keypoints,
            bbox=(0, 0, 100, 200),
            confidence=0.9,
        )

        nose = pose.get_keypoint("nose")
        assert nose is not None
        assert nose.name == "nose"

    def test_to_dict(self):
        keypoints = [Keypoint(x=100, y=50, confidence=0.9, name="nose")]
        pose = PoseResult(
            keypoints=keypoints,
            bbox=(0, 0, 100, 200),
            confidence=0.9,
            posture=Posture.STANDING,
        )

        d = pose.to_dict()
        assert "keypoints" in d
        assert d["posture"] == "standing"


class TestClothingResult:
    """Test ClothingResult dataclass."""

    def test_create_clothing_result(self):
        result = ClothingResult(
            upper_color="blue",
            upper_type="shirt",
            lower_color="black",
            lower_type="pants",
            confidence=0.7,
        )
        assert result.upper_color == "blue"
        assert result.lower_type == "pants"

    def test_to_dict(self):
        result = ClothingResult(
            upper_color="red",
            upper_type="jacket",
            lower_color="gray",
            lower_type="shorts",
            confidence=0.8,
        )
        d = result.to_dict()
        assert d["upper"]["color"] == "red"
        assert d["lower"]["type"] == "shorts"


class TestPoseEstimator:
    """Test PoseEstimator."""

    def test_create_estimator(self):
        estimator = PoseEstimator()
        assert estimator is not None

    def test_is_available(self):
        estimator = PoseEstimator()
        # May or may not be available depending on Hailo
        assert isinstance(estimator.is_available(), bool)

    def test_estimate_empty_image(self):
        estimator = PoseEstimator()
        empty_image = np.zeros((100, 100, 3), dtype=np.uint8)
        poses = estimator.estimate(empty_image)
        # Should return list (possibly empty)
        assert isinstance(poses, list)


class TestClothingAnalyzer:
    """Test ClothingAnalyzer."""

    def test_analyze_basic(self, person_crop):
        analyzer = ClothingAnalyzer()
        result = analyzer.analyze(person_crop)

        assert result is not None
        assert result.upper_color is not None
        assert result.lower_color is not None

    def test_analyze_with_keypoints(self, person_crop):
        analyzer = ClothingAnalyzer()
        keypoints = [Keypoint(x=75, y=50 + i * 15, confidence=0.9, name=name)
                     for i, name in enumerate(KEYPOINT_NAMES)]

        result = analyzer.analyze(person_crop, keypoints)
        assert result is not None


class TestFaceDetector:
    """Test FaceDetector."""

    def test_detect_basic(self, person_crop):
        detector = FaceDetector()
        faces = detector.detect(person_crop)

        # Should return list (possibly empty)
        assert isinstance(faces, list)


class TestPersonPipeline:
    """Test PersonPipeline."""

    def test_process_object(self, sample_object, person_crop):
        pipeline = PersonPipeline()
        result = pipeline.process_object(sample_object, person_crop)

        assert "clothing" in result.attributes
        assert "face" in result.attributes


class TestPersonPipelineModule:
    """Test PersonPipelineModule."""

    def test_module_properties(self):
        module = PersonPipelineModule()
        assert module.name == "person_pipeline"
        assert "object" in module.input_spec.required_fields


# =============================================================================
# Vehicle Pipeline Tests
# =============================================================================


class TestVehicleType:
    """Test VehicleType enum."""

    def test_vehicle_types(self):
        assert VehicleType.SEDAN.value == "sedan"
        assert VehicleType.SUV.value == "suv"
        assert VehicleType.TRUCK.value == "truck"


class TestVehicleOrientation:
    """Test VehicleOrientation enum."""

    def test_orientations(self):
        assert VehicleOrientation.FRONT.value == "front"
        assert VehicleOrientation.REAR.value == "rear"
        assert VehicleOrientation.SIDE_LEFT.value == "side_left"


class TestVehicleColorResult:
    """Test VehicleColorResult dataclass."""

    def test_create_result(self):
        result = VehicleColorResult(
            primary_color="blue",
            secondary_color=None,
            color_confidence=0.9,
            is_metallic=True,
            hsv_values=(120, 200, 180),
        )
        assert result.primary_color == "blue"
        assert result.is_metallic == True

    def test_to_dict(self):
        result = VehicleColorResult(
            primary_color="red",
            secondary_color="black",
            color_confidence=0.8,
            is_metallic=False,
            hsv_values=(0, 200, 180),
        )
        d = result.to_dict()
        assert d["primary"] == "red"
        assert d["secondary"] == "black"


class TestVehicleColorAnalyzer:
    """Test VehicleColorAnalyzer."""

    def test_analyze_basic(self, vehicle_crop):
        analyzer = VehicleColorAnalyzer()
        result = analyzer.analyze(vehicle_crop)

        assert result is not None
        assert result.primary_color is not None

    def test_analyze_with_mask(self, vehicle_crop, sample_mask):
        analyzer = VehicleColorAnalyzer()
        # Resize mask to match crop
        import cv2
        mask = cv2.resize(sample_mask, (vehicle_crop.shape[1], vehicle_crop.shape[0]))
        result = analyzer.analyze(vehicle_crop, mask)

        assert result is not None

    def test_analyze_grayscale(self):
        analyzer = VehicleColorAnalyzer()
        # Create a gray image
        gray_image = np.full((100, 150, 3), 128, dtype=np.uint8)
        result = analyzer.analyze(gray_image)

        assert result.primary_color in ["gray", "silver", "white", "black"]


class TestVehicleTypeClassifier:
    """Test VehicleTypeClassifier."""

    def test_classify_basic(self, vehicle_crop):
        classifier = VehicleTypeClassifier()
        result = classifier.classify(vehicle_crop)

        assert result is not None
        assert isinstance(result.vehicle_type, VehicleType)

    def test_classify_with_hint(self, vehicle_crop):
        classifier = VehicleTypeClassifier()
        result = classifier.classify(vehicle_crop, class_hint="truck")

        assert result.vehicle_type == VehicleType.TRUCK
        assert result.confidence >= 0.8

    def test_classify_wide_aspect(self):
        classifier = VehicleTypeClassifier()
        wide_crop = np.random.randint(0, 255, (100, 350, 3), dtype=np.uint8)
        result = classifier.classify(wide_crop)

        # Wide aspect should be bus or truck
        assert result.vehicle_type in [VehicleType.BUS, VehicleType.TRUCK, VehicleType.PICKUP]


class TestLicensePlateDetector:
    """Test LicensePlateDetector."""

    def test_detect_basic(self, vehicle_crop):
        detector = LicensePlateDetector()
        result = detector.detect(vehicle_crop)

        assert isinstance(result, LicensePlateResult)

    def test_detect_disabled(self, vehicle_crop):
        config = VehiclePipelineConfig(enable_license_plate=False)
        detector = LicensePlateDetector(config)
        result = detector.detect(vehicle_crop)

        assert result.detected == False


class TestVehiclePipeline:
    """Test VehiclePipeline."""

    def test_process_object(self, sample_object, vehicle_crop):
        sample_object.primary_class = "car"
        pipeline = VehiclePipeline()
        result = pipeline.process_object(sample_object, vehicle_crop)

        assert "color" in result.attributes
        assert "vehicle_type" in result.attributes
        assert "license_plate" in result.attributes


class TestVehiclePipelineModule:
    """Test VehiclePipelineModule."""

    def test_module_properties(self):
        module = VehiclePipelineModule()
        assert module.name == "vehicle_pipeline"


# =============================================================================
# Generic Pipeline Tests
# =============================================================================


class TestClassificationResult:
    """Test ClassificationResult dataclass."""

    def test_create_result(self):
        result = ClassificationResult(
            class_name="laptop",
            class_id=42,
            confidence=0.85,
            top_k=[("laptop", 0.85), ("keyboard", 0.10)],
        )
        assert result.class_name == "laptop"
        assert result.confidence == 0.85

    def test_to_dict(self):
        result = ClassificationResult(
            class_name="chair",
            class_id=56,
            confidence=0.75,
            top_k=[("chair", 0.75)],
        )
        d = result.to_dict()
        assert d["class"] == "chair"
        assert len(d["top_k"]) == 1


class TestColorResult:
    """Test ColorResult dataclass."""

    def test_create_result(self):
        result = ColorResult(
            dominant_colors=[("blue", 0.6), ("white", 0.3)],
            color_histogram=np.zeros(180),
            is_multicolored=False,
        )
        assert len(result.dominant_colors) == 2

    def test_to_dict(self):
        result = ColorResult(
            dominant_colors=[("red", 0.8)],
            color_histogram=None,
            is_multicolored=False,
        )
        d = result.to_dict()
        assert d["dominant"][0]["color"] == "red"


class TestShapeResult:
    """Test ShapeResult dataclass."""

    def test_create_result(self):
        result = ShapeResult(
            aspect_ratio=1.5,
            solidity=0.9,
            extent=0.8,
            circularity=0.3,
            num_vertices=4,
            shape_type="rectangular",
        )
        assert result.shape_type == "rectangular"

    def test_to_dict(self):
        result = ShapeResult(
            aspect_ratio=1.0,
            solidity=0.95,
            extent=0.9,
            circularity=0.9,
            num_vertices=0,
            shape_type="circular",
        )
        d = result.to_dict()
        assert d["type"] == "circular"


class TestImageNetClassifier:
    """Test ImageNetClassifier."""

    def test_create_classifier(self):
        classifier = ImageNetClassifier()
        assert classifier is not None

    def test_is_available(self):
        classifier = ImageNetClassifier()
        assert isinstance(classifier.is_available(), bool)

    def test_classify_returns_result(self, sample_image):
        classifier = ImageNetClassifier()
        result = classifier.classify(sample_image)

        # Should always return ClassificationResult
        assert isinstance(result, ClassificationResult)


class TestColorAnalyzer:
    """Test ColorAnalyzer (generic)."""

    def test_analyze_basic(self, sample_image):
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(sample_image)

        assert result is not None
        assert len(result.dominant_colors) > 0

    def test_analyze_with_mask(self, sample_image, sample_mask):
        analyzer = ColorAnalyzer()
        result = analyzer.analyze(sample_image, sample_mask)

        assert result is not None

    def test_analyze_empty_image(self):
        analyzer = ColorAnalyzer()
        empty = np.zeros((0, 0, 3), dtype=np.uint8)
        result = analyzer.analyze(empty)

        assert result.dominant_colors[0][0] == "unknown"


class TestShapeAnalyzer:
    """Test ShapeAnalyzer."""

    def test_analyze_basic(self, sample_image):
        analyzer = ShapeAnalyzer()
        result = analyzer.analyze(sample_image)

        assert result is not None
        assert result.aspect_ratio > 0

    def test_analyze_with_mask(self, sample_image, sample_mask):
        analyzer = ShapeAnalyzer()
        result = analyzer.analyze(sample_image, sample_mask)

        assert result is not None

    def test_shape_classification(self):
        analyzer = ShapeAnalyzer()

        # Create circular mask
        mask = np.zeros((100, 100), dtype=np.uint8)
        cv2 = __import__('cv2')
        cv2.circle(mask, (50, 50), 40, 255, -1)

        image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
        result = analyzer.analyze(image, mask)

        # Should detect circular shape
        assert result.circularity > 0.7


class TestSizeEstimator:
    """Test SizeEstimator."""

    def test_estimate_pixel_size(self):
        estimator = SizeEstimator()
        bbox = (100, 100, 200, 300)
        result = estimator.estimate(bbox)

        assert result["pixel_width"] == 100
        assert result["pixel_height"] == 200
        assert result["pixel_area"] == 20000

    def test_estimate_with_depth(self):
        estimator = SizeEstimator()
        bbox = (100, 100, 200, 200)
        depth = np.full((480, 640), 2000, dtype=np.float32)  # 2 meters
        intrinsics = {"fx": 600, "fy": 600, "cx": 320, "cy": 240}

        result = estimator.estimate(bbox, depth, intrinsics)

        assert "real_width_m" in result
        assert "depth_m" in result


class TestGenericPipeline:
    """Test GenericPipeline."""

    def test_process_object(self, sample_object, sample_image):
        pipeline = GenericPipeline()
        crop = sample_image[100:300, 100:300]
        result = pipeline.process_object(sample_object, crop)

        assert "classification" in result.attributes
        assert "color" in result.attributes
        assert "shape" in result.attributes


class TestGenericPipelineModule:
    """Test GenericPipelineModule."""

    def test_module_properties(self):
        module = GenericPipelineModule()
        assert module.name == "generic_pipeline"


# =============================================================================
# Integration Tests
# =============================================================================


class TestPipelineIntegration:
    """Integration tests for pipelines."""

    def test_router_with_person_pipeline(self, sample_object, person_crop):
        router = PipelineRouter()
        person_pipeline = PersonPipeline()
        router.register_pipeline(PipelineType.PERSON, person_pipeline)

        # Create person detection
        sample_object.primary_class = "person"

        # Route should select person pipeline
        detection = Detection(
            class_id=0,
            class_name="person",
            confidence=sample_object.confidence,
            bbox=sample_object.bounding_box_2d,
        )
        decision = router.route(detection)
        assert decision.pipeline_type == PipelineType.PERSON

    def test_router_with_vehicle_pipeline(self, sample_object, vehicle_crop):
        router = PipelineRouter()
        vehicle_pipeline = VehiclePipeline()
        router.register_pipeline(PipelineType.VEHICLE, vehicle_pipeline)

        sample_object.primary_class = "car"

        detection = Detection(
            class_id=2,
            class_name="car",
            confidence=sample_object.confidence,
            bbox=sample_object.bounding_box_2d,
        )
        decision = router.route(detection)
        assert decision.pipeline_type == PipelineType.VEHICLE

    def test_full_person_processing(self, sample_object, person_crop):
        pipeline = PersonPipeline()
        sample_object.primary_class = "person"

        result = pipeline.process_object(sample_object, person_crop)

        # Check all expected attributes
        assert "clothing" in result.attributes
        assert "face" in result.attributes
        assert result.attributes["clothing"]["upper"]["color"] is not None

    def test_full_vehicle_processing(self, sample_object, vehicle_crop):
        pipeline = VehiclePipeline()
        sample_object.primary_class = "car"

        result = pipeline.process_object(sample_object, vehicle_crop)

        assert "color" in result.attributes
        assert "vehicle_type" in result.attributes
        assert result.attributes["color"]["primary"] is not None

    def test_full_generic_processing(self, sample_object, sample_image):
        pipeline = GenericPipeline()
        crop = sample_image[100:300, 100:300]

        result = pipeline.process_object(sample_object, crop)

        assert "classification" in result.attributes
        assert "color" in result.attributes
        assert "shape" in result.attributes
