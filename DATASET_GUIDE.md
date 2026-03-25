# STCA-Net Dataset & Training Guide

## 1. Acquiring High-Quality Datasets
To achieve state-of-the-art accuracy, do not rely on toy datasets. You must request and download established benchmarks:
- **FaceForensics++ (FF++)**: The industry standard. Contains various compressions (c23, c40). 
- **Celeb-DF**: High-quality, highly realistic fakes.
- **DFDC**: Huge dataset by Meta with extreme perturbations.

*Note: You must request access via the creators' GitHub pages by signing an academic release form. They will email you temporary python download scripts.*

## 2. Smart Frame Extraction
We have updated `utils/video_processing.py` to use **Variance of Laplacian**. This automatically calculates the "blurriness" of frames and extracts only the sharpest faces from your videos. 
When you download the raw videos from FF++ or Celeb-DF, run them through our updated video processing script to extract frames into your `dataset/real` and `dataset/fake` folders.

## 3. Data Augmentations & Architecture Applied
Our training script (`train_stca_net.py`) now includes state-of-the-art techniques for Deepfake detection:
- **Aggressive Augmentation**: JPEG Compression, Gaussian Noise/Blur, Color Jitter, and Random Resized Cropping help the model generalize to any camera or platform.
- **Frequency Domain Analysis**: STCA-Net now calculates the 2D Fast Fourier Transform (FFT) of the input to inject frequency artifacts into the Vision Transformer's Cross-Attention mechanism.
- **Hard Negative Mining**: We replaced standard loss with **Focal Loss**, which dynamically rescales the loss to focus heavily on confusing examples.

## 4. Training Command
Once your data is extracted:
```bash
python train_stca_net.py --dataset dataset/benchmark_data --epochs 30 --batch-size 32
```
