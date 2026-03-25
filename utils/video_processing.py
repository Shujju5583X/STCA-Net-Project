import cv2
import os
import logging
from PIL import Image

from utils.face_detection import get_face_cascade, extract_face

logger = logging.getLogger(__name__)

def extract_frames_from_video(video_path, max_frames=15, output_dir=None):
    """
    Extracts a sequence of face-cropped frames from a video file.
    Args:
        video_path: Path to the video file
        max_frames: Number of evenly spaced frames to extract
        output_dir: If provided, saves the extracted frames to disk
    Returns:
        List of PIL Images containing the cropped faces
    """
    if not os.path.exists(video_path):
        logger.error(f"Video file not found: {video_path}")
        return []
        
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logger.error(f"Failed to open video: {video_path}")
        return []
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 0 or fps <= 0:
        logger.error("Invalid video format or unreadable frames.")
        cap.release()
        return []
        
    # Calculate interval to get evenly spaced frames
    interval = max(1, total_frames // max_frames)
    
    extracted_images = []
    cascade = get_face_cascade()
    
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    for i in range(max_frames):
        start_frame = i * interval
        end_frame = min((i + 1) * interval, total_frames)
        
        best_frame = None
        max_sharpness = -1.0
        
        # Sample up to 'sample_limit' frames within this interval to find the sharpest one
        # Avoid processing every single frame for long videos to maintain performance
        sample_limit = min(5, end_frame - start_frame)
        if sample_limit < 1:
            continue
            
        step = max(1, (end_frame - start_frame) // sample_limit)
        
        for f in range(start_frame, end_frame, step):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f)
            ret, frame = cap.read()
            if not ret:
                break
                
            # Convert to gray to check sharpness using Variance of Laplacian
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            if sharpness > max_sharpness:
                max_sharpness = sharpness
                best_frame = frame
                
        if best_frame is not None:
            # We found the sharpest frame in this interval, now extract the face
            rgb_frame = cv2.cvtColor(best_frame, cv2.COLOR_BGR2RGB)
            face_img, _ = extract_face(rgb_frame, cascade)
            
            if face_img is not None:
                extracted_images.append(face_img)
                
                # Save to disk if output_dir is provided
                if output_dir:
                    save_path = os.path.join(output_dir, f"frame_{len(extracted_images):04d}.jpg")
                    face_img.save(save_path)
        
    cap.release()
    logger.info(f"Extracted {len(extracted_images)} smart-selected face frames from video.")
    return extracted_images
