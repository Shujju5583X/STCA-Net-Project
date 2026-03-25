import os
import time
import logging
from flask import Flask, render_template, request, jsonify, redirect, url_for
from werkzeug.utils import secure_filename
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import torch

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from models.stca_net import STCANet
from utils.video_processing import extract_frames_from_video
from utils.prediction import predict_image as model_predict_image
from utils.prediction import predict_video_frames

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-fallback-change-in-production')

# Enable CORS for API access from separate frontends
CORS(app)

# Rate limiting to prevent abuse
limiter = Limiter(get_remote_address, app=app, default_limits=['200 per day', '50 per hour'])
app.config['UPLOAD_FOLDER'] = 'Uploaded_Files'
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500 MB max upload

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('static', exist_ok=True) # Ensure static dir exists

# Global Model Definition (Lazy Load ready)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_PATH = "model/stca_net_weights.pt"

logger.info(f"Flask App Initializing. Target Device: {device}")

# Pre-instantiate the model structure
stca_model = STCANet().to(device)

def load_stca_weights():
    """Loads weights if they exist, otherwise runs with random init for testing"""
    if os.path.exists(MODEL_PATH):
        try:
            stca_model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
            logger.info(f"Successfully loaded STCA-Net weights from {MODEL_PATH}")
        except Exception as e:
            logger.error(f"Failed to load weights: {e}")
    else:
        logger.warning(f"Weights not found at {MODEL_PATH}. Using untrained STCA-Net for demonstration.")

# Attempt to load weights on startup
load_stca_weights()

@app.route('/health')
def health_check():
    """Health check endpoint for Docker/Render monitoring."""
    return jsonify({'status': 'healthy', 'model_loaded': os.path.exists(MODEL_PATH)}), 200

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/detect', methods=['GET', 'POST'])
def detect_video():
    if request.method == 'GET':
        return render_template('detect.html')
        
    if 'video' not in request.files:
        return render_template('detect.html', error="No video file uploaded")
        
    video = request.files['video']
    if video.filename == '':
        return render_template('detect.html', error="No video selected")
        
    if not video.filename.lower().endswith(('.mp4', '.avi', '.mov')):
        return render_template('detect.html', error="Invalid format. Please use MP4, AVI, or MOV.")
        
    filename = secure_filename(video.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    video.save(filepath)
    
    try:
        start_time = time.time()
        logger.info(f"Processing VIDEO: {filename}")
        
        # 1. Extract Frames
        frames = extract_frames_from_video(filepath, max_frames=15)
        
        if not frames:
            raise Exception("No readable frames or faces found in video.")
            
        # 2. Predict
        result = predict_video_frames(stca_model, frames, device=device)
        processing_time = round(time.time() - start_time, 2)
        
        result['processing_time'] = processing_time
        result['model_used'] = "STCA-Net (Spatiotemporal Cross-Attention)"
        
        logger.info(f"Video Result: {result['prediction']} ({result['confidence']}%) in {processing_time}s")
        
        # Clean up video
        os.remove(filepath)
        
        return render_template('detect.html', data=result)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Video processing failed: {str(e)}")
        return render_template('detect.html', error=str(e))

@app.route('/image-detect', methods=['GET', 'POST'])
def detect_image():
    if request.method == 'GET':
        return render_template('image.html')
        
    if 'image' not in request.files:
        return render_template('image.html', error="No image file uploaded")
        
    image = request.files['image']
    if image.filename == '':
        return render_template('image.html', error="No image selected")
        
    if not image.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return render_template('image.html', error="Invalid format. Please use JPG or PNG.")
        
    filename = secure_filename(image.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    image.save(filepath)
    
    try:
        start_time = time.time()
        logger.info(f"Processing IMAGE: {filename}")
        
        # 1. Predict
        result = model_predict_image(stca_model, filepath, device=device)
        processing_time = round(time.time() - start_time, 2)
        
        result['processing_time'] = processing_time
        result['model_used'] = "STCA-Net (Spatiotemporal Cross-Attention)"
        
        logger.info(f"Image Result: {result['prediction']} ({result['confidence']}%) in {processing_time}s")
        
        # Clean up image
        os.remove(filepath)
        
        return render_template('image.html', data=result)
        
    except Exception as e:
        if os.path.exists(filepath):
            os.remove(filepath)
        logger.error(f"Image processing failed: {str(e)}")
        return render_template('image.html', error=str(e))

if __name__ == '__main__':
    # Force single-threaded execution to prevent memory spikes with ML models
    app.run(host='0.0.0.0', port=5000, threaded=False, debug=True)
