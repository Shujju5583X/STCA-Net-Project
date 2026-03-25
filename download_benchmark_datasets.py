import os
import argparse

def create_dataset_structure(base_dir):
    os.makedirs(os.path.join(base_dir, 'real'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'fake'), exist_ok=True)
    print(f"Created dataset structure at: {base_dir}")

def download_sample_data(base_dir):
    """
    Downloads a tiny placeholder sample dataset to test the pipeline.
    Note: Real benchmark datasets (FaceForensics++, Celeb-DF) are 100s of GBs
    and require you to fill out academic request forms to get the download links.
    """
    print("=========================================================")
    print("WARNING: Real deepfake datasets require academic permission.")
    print("Please visit the following links to request access:")
    print("1. FaceForensics++: https://github.com/ondyari/FaceForensics")
    print("2. Celeb-DF: https://github.com/yuezunli/celeb-deepfakeforensics")
    print("3. DFDC: https://ai.meta.com/datasets/dfdc/")
    print("=========================================================\n")
    
    # Create the structure anyway for testing
    create_dataset_structure(base_dir)
    print("Dataset folders are ready. Once you download the official benchmark videos,")
    print("use `utils/video_processing.py` to extract sharp frames into:")
    print(f" - {os.path.abspath(os.path.join(base_dir, 'real'))}")
    print(f" - {os.path.abspath(os.path.join(base_dir, 'fake'))}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deepfake Dataset Setup Helper")
    parser.add_argument('--dir', type=str, default='dataset/140k', help='Target directory')
    args = parser.parse_args()
    
    download_sample_data(args.dir)
