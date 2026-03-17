import os
import cv2
import numpy as np
import torch
import logging
from torchvision import transforms
from PIL import Image
from scipy.fft import dctn

logger = logging.getLogger(__name__)

# Setup standard ImageNet normalization since MobileNet and ViT both expect it
preprocess_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# Load Haar cascade once at module level
CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
_face_cascade = None

def _get_face_cascade():
    """Lazily load the Haar cascade classifier."""
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
        if _face_cascade.empty():
            logger.error("Failed to load Haar cascade classifier.")
            raise FileNotFoundError("Haar cascade XML file not found.")
    return _face_cascade


def extract_face_from_image(pil_image):
    """
    Detect and crop the largest face from a PIL Image.
    Returns (cropped_face_pil, face_found_bool).
    If no face is found, returns a center crop.
    """
    img_array = np.array(pil_image)
    gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    
    cascade = _get_face_cascade()
    faces = cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60)
    )
    
    if len(faces) == 0:
        # No face found — center crop as fallback
        h, w = img_array.shape[:2]
        size = min(h, w)
        y1, x1 = (h - size) // 2, (w - size) // 2
        cropped = img_array[y1:y1+size, x1:x1+size]
        return Image.fromarray(cropped), False
    
    # Find the largest face by area
    largest_face = max(faces, key=lambda rect: rect[2] * rect[3])
    x, y, w, h = largest_face
    
    # Add 20% margin around the face for context
    margin_w, margin_h = int(w * 0.2), int(h * 0.2)
    x1 = max(0, x - margin_w)
    y1 = max(0, y - margin_h)
    x2 = min(img_array.shape[1], x + w + margin_w)
    y2 = min(img_array.shape[0], y + h + margin_h)
    
    cropped = img_array[y1:y2, x1:x2]
    return Image.fromarray(cropped), True


def compute_frequency_score(pil_image):
    """
    Compute a frequency-domain anomaly score using DCT analysis.
    AI-generated images (GANs, diffusion models) have distinct frequency
    patterns: typically lower high-frequency energy and smoother spectral decay.
    
    Returns a score from 0.0 (looks natural/real) to 1.0 (looks AI-generated).
    """
    try:
        # Convert to grayscale numpy array
        img_gray = np.array(pil_image.convert('L').resize((256, 256)), dtype=np.float64)
        
        # Apply 2D Discrete Cosine Transform
        dct_coeffs = dctn(img_gray, norm='ortho')
        
        # Compute power spectrum (log magnitude)
        power = np.abs(dct_coeffs)
        
        h, w = power.shape
        
        # Split into frequency bands
        # Low frequency: top-left quadrant
        low_freq = power[:h//4, :w//4]
        # Mid frequency: between low and high
        mid_freq_mask = np.zeros_like(power, dtype=bool)
        mid_freq_mask[h//4:h//2, :w//2] = True
        mid_freq_mask[:h//2, w//4:w//2] = True
        mid_freq = power[mid_freq_mask]
        # High frequency: bottom-right region
        high_freq = power[h//2:, w//2:]
        
        total_energy = np.sum(power) + 1e-10
        low_energy = np.sum(low_freq) / total_energy
        mid_energy = np.sum(mid_freq) / total_energy
        high_energy = np.sum(high_freq) / total_energy
        
        # AI-generated images typically have:
        # 1. Lower high-frequency energy ratio (smoother textures)
        # 2. More concentrated energy in low frequencies
        # 3. Smoother spectral decay
        
        # Compute spectral decay rate (row-wise average of DCT magnitudes)
        row_energies = np.mean(power, axis=1)
        if len(row_energies) > 1:
            # Fit log decay — steeper = more AI-like
            log_energies = np.log(row_energies + 1e-10)
            x = np.arange(len(log_energies))
            # Simple linear fit
            slope = np.polyfit(x, log_energies, 1)[0]
        else:
            slope = 0
        
        # Scoring heuristic (calibrated empirically):
        # High-freq ratio < 0.08 is suspicious (AI-like)
        # Spectral slope < -0.06 is suspicious
        score = 0.0
        
        if high_energy < 0.05:
            score += 0.4
        elif high_energy < 0.08:
            score += 0.2
            
        # Removed the penalty for high energy, as modern diffusion models (like Midjourney/Gemini) 
        # can produce exceptionally sharp images, causing false negatives.
        
        if slope < -0.08:
            score += 0.3
        elif slope < -0.06:
            score += 0.15
        
        if low_energy > 0.85:
            score += 0.3
        elif low_energy > 0.75:
            score += 0.15
        
        # Clamp to [0, 1]
        score = max(0.0, min(1.0, score))
        
        return score
        
    except Exception as e:
        logger.warning(f"Frequency analysis failed: {e}")
        return 0.0  # Default to no signal


def detect_non_photographic(pil_image):
    """
    Detect if an image is non-photographic (anime, cartoon, illustration, etc.)
    by analyzing color distribution, edge density and texture characteristics.
    
    Returns (is_non_photo: bool, confidence: float)
    """
    try:
        img_array = np.array(pil_image.convert('RGB').resize((256, 256)))
        
        # 1. Color histogram analysis — cartoons/anime have fewer unique colors
        #    and more saturated, uniform color blocks
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        saturation = hsv[:, :, 1]
        
        # Mean and std of saturation
        sat_mean = np.mean(saturation)
        sat_std = np.std(saturation)
        
        # 2. Edge density — cartoons have sharp, clean edges with flat areas
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)
        
        # 3. Unique color count (quantized)
        quantized = (img_array // 32) * 32  # Reduce to ~8 levels per channel
        flat = quantized.reshape(-1, 3)
        unique_colors = len(np.unique(flat, axis=0))
        
        # 4. Texture analysis — real photos have more texture variation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_var = laplacian.var()
        
        # Scoring: anime/cartoons typically have:
        # - High saturation mean (>80) with low std
        # - Moderate edge density with very flat inter-edge regions
        # - Fewer unique quantized colors (<200)
        # - Lower texture variance
        
        anime_score = 0.0
        
        if sat_mean > 90 and sat_std < 50:
            anime_score += 0.3
        
        if unique_colors < 150:
            anime_score += 0.3
        elif unique_colors < 250:
            anime_score += 0.15
        
        if texture_var < 500:
            anime_score += 0.2
        
        if edge_density > 0.05 and texture_var < 800:
            anime_score += 0.2
        
        is_non_photo = anime_score >= 0.5
        
        return is_non_photo, round(anime_score, 2)
        
    except Exception as e:
        logger.warning(f"Non-photographic detection failed: {e}")
        return False, 0.0


def check_ai_signatures(image_path, pil_image):
    """
    Check filename and image metadata for obvious AI generation signatures.
    Many modern generators leave EXIF data, software tags, or recognizable filenames.
    """
    try:
        # 1. Check filename
        basename = os.path.basename(image_path).lower()
        ai_keywords = ['gemini_generated', 'dall_e', 'dalle', 'midjourney', 'stable_diffusion', 'ai_generated', 'journey']
        if any(k in basename for k in ai_keywords):
            return 1.0, "AI signature found in filename."
            
        # 2. Check metadata / EXIF
        info_str = str(pil_image.info).lower()
        if 'google' in info_str and 'gemini' in info_str:
            return 1.0, "AI signature found in metadata (Gemini)."
        elif 'dall-e' in info_str or 'midjourney' in info_str or 'stable diffusion' in info_str:
            return 1.0, "AI generation software found in metadata."
            
    except Exception as e:
        logger.warning(f"Signature check failed: {e}")
        pass
        
    return 0.0, ""


def predict_image(model, image_path, device='cpu'):
    """
    Predicts whether a single image is REAL or FAKE using the given STCA-Net model.
    Combines neural network prediction with frequency-domain analysis and signature checks.
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found at {image_path}")
        
    try:
        img = Image.open(image_path).convert('RGB')
        
        # Check for obvious AI signatures (Watermark/Metadata/Filename)
        sig_score, sig_reason = check_ai_signatures(image_path, img)
        
        # Check for non-photographic content (anime, cartoon, etc.)
        is_non_photo, non_photo_score = detect_non_photographic(img)
        
        # Extract face from image (consistent with video pipeline)
        face_img, face_found = extract_face_from_image(img)
        
        # Compute frequency-domain analysis score
        freq_score = compute_frequency_score(img)
        
        # Neural network prediction on face-cropped image
        input_tensor = preprocess_transform(face_img).unsqueeze(0).to(device)
        
        model.eval()
        with torch.no_grad():
            output, attn_weights = model(input_tensor)
            
            # Assuming output is shape (1, 2)
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            
            # Class 0: Fake, Class 1: Real
            nn_fake_prob = probabilities[0].item()
            nn_real_prob = probabilities[1].item()
        
        # ============================================================
        # COMBINED SCORING: Blend neural network + frequency analysis + signatures
        # ============================================================
        
        if sig_score == 1.0:
            # If an explicit AI signature is found, override the prediction
            combined_fake_prob = 0.95
            combined_real_prob = 0.05
        else:
            # Neural network weight: 0.7, Frequency analysis weight: 0.3
            nn_weight = 0.70
            freq_weight = 0.30
            
            # freq_score is 0=real, 1=AI-generated → treat as fake probability
            combined_fake_prob = (nn_fake_prob * nn_weight) + (freq_score * freq_weight)
            combined_real_prob = (nn_real_prob * nn_weight) + ((1 - freq_score) * freq_weight)
            
            # Normalize
            total = combined_fake_prob + combined_real_prob
            combined_fake_prob /= total
            combined_real_prob /= total
            
        fake_prob = combined_fake_prob * 100
        real_prob = combined_real_prob * 100
        
        prediction = "REAL" if real_prob > fake_prob else "FAKE"
        confidence = real_prob if prediction == "REAL" else fake_prob
        
        result = {
            'prediction': prediction,
            'confidence': round(confidence, 2),
            'fake_probability': round(fake_prob, 2),
            'real_probability': round(real_prob, 2),
            'attention_map': attn_weights.cpu().numpy(),
            'face_detected': face_found,
            'frequency_score': round(freq_score * 100, 2),
            'signature_found': sig_score == 1.0,
            'signature_reason': sig_reason
        }
        
        # Add non-photographic warning if detected
        if is_non_photo:
            result['warning'] = (
                "This appears to be non-photographic content (anime, cartoon, or illustration). "
                "Deepfake detection is designed for real photographs and may not be reliable for this type of image."
            )
            result['is_non_photographic'] = True
        else:
            result['is_non_photographic'] = False
        
        logger.info(
            f"Image prediction: {prediction} ({confidence:.1f}%) | "
            f"NN: fake={nn_fake_prob:.3f} real={nn_real_prob:.3f} | "
            f"Freq: {freq_score:.3f} | Sig: {sig_score} | Face: {face_found} | NonPhoto: {is_non_photo}"
        )
        
        return result
        
    except Exception as e:
        raise Exception(f"Failed to process image: {str(e)}")


def predict_video_frames(model, frames, device='cpu'):
    """
    Takes a list of PIL Images (extracted frames), predicts each one, 
    and aggregates the overall prediction.
    """
    if not frames:
        raise ValueError("No frames provided for prediction.")
        
    fake_scores = []
    real_scores = []
    freq_scores = []
    
    model.eval()
    with torch.no_grad():
        for frame in frames:
            # frame is a PIL Image (already face-cropped from video_processing)
            input_tensor = preprocess_transform(frame).unsqueeze(0).to(device)
            output, _ = model(input_tensor)
            
            probabilities = torch.nn.functional.softmax(output[0], dim=0)
            fake_scores.append(probabilities[0].item())
            real_scores.append(probabilities[1].item())
            
            # Also compute frequency score for each frame
            freq_scores.append(compute_frequency_score(frame))
            
    # Aggregate neural network scores (Average)
    avg_nn_fake = sum(fake_scores) / len(fake_scores)
    avg_nn_real = sum(real_scores) / len(real_scores)
    avg_freq = sum(freq_scores) / len(freq_scores)
    
    # Combine with frequency analysis (same weighting as image)
    nn_weight = 0.70
    freq_weight = 0.30
    
    combined_fake = (avg_nn_fake * nn_weight) + (avg_freq * freq_weight)
    combined_real = (avg_nn_real * nn_weight) + ((1 - avg_freq) * freq_weight)
    
    total = combined_fake + combined_real
    combined_fake /= total
    combined_real /= total
    
    avg_fake_prob = combined_fake * 100
    avg_real_prob = combined_real * 100
    
    prediction = "REAL" if avg_real_prob > avg_fake_prob else "FAKE"
    confidence = avg_real_prob if prediction == "REAL" else avg_fake_prob
    
    return {
        'prediction': prediction,
        'confidence': round(confidence, 2),
        'frames_analyzed': len(frames),
        'fake_probability': round(avg_fake_prob, 2),
        'real_probability': round(avg_real_prob, 2),
        'frequency_score': round(avg_freq * 100, 2),
    }
