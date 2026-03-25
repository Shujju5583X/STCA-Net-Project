"""
Tests for prediction utilities.
"""
import os
import pytest
import torch
import numpy as np
from PIL import Image
from unittest.mock import patch, MagicMock


class TestComputeFrequencyScore:
    """Test frequency-domain analysis."""
    
    def test_returns_float_in_range(self):
        from utils.prediction import compute_frequency_score
        # Create a random test image
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        score = compute_frequency_score(img)
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0, f"Score {score} out of range [0, 1]"

    def test_solid_color_image(self):
        from utils.prediction import compute_frequency_score
        # Solid color = all energy in DC component (low freq)
        img = Image.fromarray(np.full((256, 256, 3), 128, dtype=np.uint8))
        score = compute_frequency_score(img)
        assert 0.0 <= score <= 1.0

    def test_noise_image(self):
        from utils.prediction import compute_frequency_score
        # Pure noise = high-frequency energy spread
        np.random.seed(42)
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        score = compute_frequency_score(img)
        assert 0.0 <= score <= 1.0

    def test_small_image(self):
        from utils.prediction import compute_frequency_score
        # Very small image should not crash
        img = Image.fromarray(np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8))
        score = compute_frequency_score(img)
        assert 0.0 <= score <= 1.0


class TestFaceDetection:
    """Test shared face detection module."""
    
    def test_extract_face_pil_input(self):
        from utils.face_detection import extract_face
        # Random image (unlikely to have a face) → should return center crop
        img = Image.fromarray(np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8))
        result, found = extract_face(img)
        assert isinstance(result, Image.Image)
        assert isinstance(found, bool)

    def test_extract_face_numpy_input(self):
        from utils.face_detection import extract_face
        # Numpy array input
        arr = np.random.randint(0, 255, (300, 400, 3), dtype=np.uint8)
        result, found = extract_face(arr)
        assert isinstance(result, Image.Image)

    def test_center_crop_fallback(self):
        from utils.face_detection import extract_face
        # Solid color image → definitely no face → center crop
        img = Image.fromarray(np.full((200, 300, 3), 128, dtype=np.uint8))
        result, found = extract_face(img)
        assert found == False
        assert isinstance(result, Image.Image)
        # Center crop of non-square image should be square
        assert result.size[0] == result.size[1]


class TestDetectNonPhotographic:
    """Test anime/cartoon detection."""
    
    def test_returns_tuple(self):
        from utils.prediction import detect_non_photographic
        img = Image.fromarray(np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8))
        result = detect_non_photographic(img)
        assert isinstance(result, tuple)
        assert len(result) == 2
        is_non_photo, score = result
        assert isinstance(is_non_photo, bool)
        assert isinstance(score, float)


class TestPredictImage:
    """Test the full image prediction pipeline."""
    
    def test_predict_image_returns_valid_dict(self, tmp_path):
        from utils.prediction import predict_image
        from models.stca_net import STCANet
        
        # Create a test image file
        img = Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8))
        img_path = str(tmp_path / "test.jpg")
        img.save(img_path)
        
        # Create untrained model
        model = STCANet()
        model.eval()
        
        result = predict_image(model, img_path, device='cpu')
        
        # Check all expected keys
        expected_keys = ['prediction', 'confidence', 'fake_probability', 'real_probability',
                        'face_detected', 'frequency_score', 'signature_found']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        assert result['prediction'] in ('REAL', 'FAKE')
        assert 0 <= result['confidence'] <= 100
        assert 0 <= result['fake_probability'] <= 100
        assert 0 <= result['real_probability'] <= 100

    def test_predict_image_file_not_found(self):
        from utils.prediction import predict_image
        from models.stca_net import STCANet
        
        model = STCANet()
        with pytest.raises(FileNotFoundError):
            predict_image(model, "/nonexistent/path.jpg")


class TestPredictVideoFrames:
    """Test video frame prediction pipeline."""
    
    def test_predict_video_frames_valid(self):
        from utils.prediction import predict_video_frames
        from models.stca_net import STCANet
        
        model = STCANet()
        model.eval()
        
        # Create fake frames
        frames = [
            Image.fromarray(np.random.randint(0, 255, (112, 112, 3), dtype=np.uint8))
            for _ in range(5)
        ]
        
        result = predict_video_frames(model, frames, device='cpu')
        
        expected_keys = ['prediction', 'confidence', 'frames_analyzed',
                        'fake_probability', 'real_probability', 'frequency_score']
        for key in expected_keys:
            assert key in result, f"Missing key: {key}"
        
        assert result['frames_analyzed'] == 5
        assert result['prediction'] in ('REAL', 'FAKE')

    def test_predict_video_frames_empty(self):
        from utils.prediction import predict_video_frames
        from models.stca_net import STCANet
        
        model = STCANet()
        with pytest.raises(ValueError):
            predict_video_frames(model, [], device='cpu')
