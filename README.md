# 🛡️ STCA-Net Deepfake Detection Project

A novel, lightweight deepfake detection system utilizing a **Spatio-Temporal Cross-Attention (STCA)** architecture. This project efficiently combines the spatial feature extraction of MobileNetV3 with the global context encoding of a Vision Transformer (ViT) to detect pixel-level and context-level artifacts in both images and videos.

## ✨ Features

- **Advanced Image & Video Analysis**: Uses OpenCV/Haar Cascades to reliably detect and crop faces before feeding them into the neural network, drastically improving accuracy by ignoring irrelevant backgrounds.
- **Frequency-Domain AI Detection**: Identifies AI-generated images (GANs, diffusion) by combining cross-attention neural network scores with Discrete Cosine Transform (DCT) frequency spectrum analysis.
- **Non-Photographic Recognition**: Prevents false predictions on cartoons, anime, and illustrations by evaluating color variance and edge densities.
- **Optimized for CPU & GPU**: Designed for researchers and developers with limited local hardware, while fully supporting GPU acceleration (e.g., Google Colab). The STCA-Net architecture boasts under 10MB of trainable parameters.
- **Modern User Interface**: A responsive, Flask-powered web dashboard that visualizes prediction confidence, processing time, and frequency analysis outcomes.

## 🌟 Recent Updates

We have significantly improved the project architecture, security, and developer experience:
- **Google Colab Support**: Added scripts (`download_faceforensics.py`, `download_benchmark_datasets.py`) for easy dataset downloading and model training on Google Colab GPUs.
- **Python 3.12+ Compatibility**: Upgraded core dependencies (PyTorch, torchvision, NumPy, SciPy) to ensure compatibility with modern Python environments.
- **Environment-based Configuration**: Implemented `.env` support for secure configuration management.
- **Robust Testing Infrastructure**: Added comprehensive unit tests using `pytest` to ensure code reliability and prevent regressions.
- **Deployment Ready**: Included `Dockerfile` and `render.yaml` for seamless deployment to cloud platforms like Render.
- **Enhanced Security & Error Handling**: Improved security practices, added `SECURITY.md`, and refined robust error handling throughout the application.
- **Code Quality**: Refactored the codebase to de-duplicate utility functions and remove unused dependencies for better maintainability.

## 📁 Repository Structure

```
STCA-Net-Project/
├── app.py                      # Main Flask web server
├── server.py                   # Production server entry point
├── train_stca_net.py           # Script to train/finetune the model
├── requirements.txt            # Python dependencies
├── .env.example                # Example environment configuration
├── Dockerfile & render.yaml    # Deployment configuration
├── models/
│   └── stca_net.py             # The PyTorch model architecture
├── tests/                      # Unit tests (pytest)
├── utils/                      # Core analysis and processing utilities
├── static/ & templates/        # Modern UI/UX frontend assets
├── download_faceforensics.py   # Dataset acquisition script
└── *.md                        # Comprehensive documentation (Guides, Security)
```

> **Note**: This repository contains only the code. Large datasets, heavy virtual environments, and the compiled model weights (`.pt` files) are intentionally excluded via `.gitignore` to keep the repository lightweight.

## 🚀 Getting Started

### 1. Installation

Requires Python 3.10 to 3.12+.

```bash
# Clone the repository
git clone https://github.com/YOUR_USERNAME/STCA-Net-Project.git
cd STCA-Net-Project

# Create and activate a virtual environment
python -m venv venv
# Windows:
venv\Scripts\activate
# Mac/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration & Dataset Setup

1. Copy `.env.example` to `.env` and set any required environment variables.
2. For training, you can use our automated scripts to fetch datasets:
   ```bash
   python test_dataset/download_benchmark_datasets.py
   # Or for original FaceForensics++:
   python download_faceforensics.py
   ```
   *See [DATASET_SETUP.md](DATASET_SETUP.md) and [DATASET_GUIDE.md](DATASET_GUIDE.md) for detailed instructions on acquiring and preparing benchmark datasets.*

### 3. Training the Model (Local or Google Colab)

```bash
# Train the model (automatically saves to models/stca_net_weights.pt)
python train_stca_net.py --dataset path/to/dataset --epochs 15 --samples 10000 --unfreeze-layers 3
```

*For Google Colab execution, please refer to our dataset scripts and Colab guides to leverage free GPU resources.*

### 4. Running the Web App

#### Local Development
```bash
python app.py
```
Open your browser to `http://127.0.0.1:5000` to access the Deepfake Scanner Dashboard.

#### Production Deployment
You can use Docker or simply run:
```bash
gunicorn server:app
```
Deployment configurations for Render (via `render.yaml`) are included out-of-the-box.

## 🧪 Running Tests

To run the unit tests, ensure you have the `pytest` dependency installed (or install from `requirements.txt`), then execute from the root directory:
```bash
pytest
```

## 🧠 Architectural Overview

Our STCA-Net utilizes a hybrid approach:
1. **Local Extractor**: A pre-trained MobileNetV3-Small backbone extracts spatial features (edges, textures, pixel blending artifacts).
2. **Global Encoder**: A 2-layer Vision Transformer (ViT) processes the spatial map as a sequence to understand global context and inconsistencies across the face.
3. **Cross-Attention Fusion**: Attention mechanisms merge the global anomalies with specific spatial regions for final classification.
4. **Frequency Fallback**: Pixel-based neural networks struggle with the smooth spectral decay of Latent Diffusion models. STCA-Net combines NN scores with a traditional Discrete Cosine Transform evaluating high-frequency energy ratios.

## 📄 License & Security
MIT License. Feel free to fork and modify for research or educational purposes.

Please review our [SECURITY.md](SECURITY.md) for reporting vulnerabilities.
