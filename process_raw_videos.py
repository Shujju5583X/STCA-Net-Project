import os
import argparse
from tqdm import tqdm
from utils.video_processing import extract_frames_from_video

def process_videos_in_directory(input_dir, output_dir, max_frames=15):
    """
    Finds all videos in input_dir, extracts the sharpest facials frames using 
    Laplacian variance, and saves them safely to output_dir without overwriting.
    """
    if not os.path.exists(input_dir):
        print(f"Directory missing: {input_dir}")
        print("Please place your raw benchmark videos here first.")
        return
        
    os.makedirs(output_dir, exist_ok=True)
    
    valid_exts = ('.mp4', '.avi', '.mov', '.mkv', '.webm')
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(valid_exts)]
    
    if not video_files:
        print(f"No video files found in {input_dir}.")
        return
        
    print(f"\nProcessing {len(video_files)} videos from '{input_dir}' -> into -> '{output_dir}'")
    total_extracted = 0
    
    for video_file in tqdm(video_files, desc="Processing videos"):
        video_path = os.path.join(input_dir, video_file)
        vid_name = os.path.splitext(video_file)[0]
        
        # We pass output_dir=None so the function just returns the PIL Images in memory
        extracted_images = extract_frames_from_video(video_path, max_frames=max_frames, output_dir=None)
        
        # Save them here with unique filenames based on the source video name 
        for frame_idx, img in enumerate(extracted_images):
            save_path = os.path.join(output_dir, f"{vid_name}_frame_{frame_idx:04d}.jpg")
            img.save(save_path)
            total_extracted += 1
            
    print(f"Done! Successfully generated {total_extracted} high-quality dataset images from {input_dir}.")


def main():
    parser = argparse.ArgumentParser(description="Build Deepfake Dataset from Raw Videos")
    parser.add_argument('--raw-real-dir', type=str, default='raw_videos/real', help='Folder containing raw REAL videos')
    parser.add_argument('--raw-fake-dir', type=str, default='raw_videos/fake', help='Folder containing raw FAKE videos')
    parser.add_argument('--out-real-dir', type=str, default='dataset/benchmark_data/real', help='Destination for REAL images')
    parser.add_argument('--out-fake-dir', type=str, default='dataset/benchmark_data/fake', help='Destination for FAKE images')
    parser.add_argument('--frames-per-video', type=int, default=15, help='Max smart frames to extract per video')
    parser.add_argument('--skip-real', action='store_true', help='Skip processing real videos (useful when appending more fake types)')
    
    args = parser.parse_args()
    
    # 1. Process Real Videos (unless --skip-real is set)
    if not args.skip_real:
        process_videos_in_directory(args.raw_real_dir, args.out_real_dir, max_frames=args.frames_per_video)
    else:
        print("Skipping real video processing (--skip-real flag set).")
    
    # 2. Process Fake Videos
    process_videos_in_directory(args.raw_fake_dir, args.out_fake_dir, max_frames=args.frames_per_video)
    
    print("\n==================================")
    print("Dataset generation is fully complete!")
    print(f"You can now begin training STCA-Net with:")
    print(f"python train_stca_net.py --dataset {os.path.dirname(args.out_real_dir)}")
    print("==================================")

if __name__ == "__main__":
    main()
