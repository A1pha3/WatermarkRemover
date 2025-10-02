"""
ProPainter Video Watermark Removal Script
For removing moving watermarks from videos on Mac

Prerequisites:
1. Install ProPainter: git clone https://github.com/sczhou/ProPainter.git
2. Create environment: conda create -n propainter python=3.8 -y
3. Install dependencies: pip3 install -r requirements.txt
"""

import os
import subprocess
import argparse

def create_mask_instructions():
    """Print instructions for creating masks for moving watermarks"""
    instructions = """
    === Creating Masks for Moving Watermarks ===
    
    For moving watermarks, you need to create frame-by-frame masks:
    
    Method 1: Manual mask creation
    1. Extract frames from your video:
       ffmpeg -i input_video.mp4 frames/%05d.jpg
    
    2. Create mask images where:
       - White (255) = watermark area to remove
       - Black (0) = area to keep
       
    3. Save masks with same names as frames in a separate folder
    
    Method 2: Use a single mask (for static watermarks)
    - Create one mask image (PNG format)
    - ProPainter will apply it to all frames
    
    Method 3: Use tracking software
    - Use tools like DaVinci Resolve or Blender to track watermark position
    - Export mask sequence
    """
    print(instructions)

def run_propainter(video_path, mask_path, output_dir="results", 
                   width=None, height=None, use_fp16=True,
                   neighbor_length=5, ref_stride=15, subvideo_length=50):
    """
    Run ProPainter to remove watermarks from video
    
    Args:
        video_path: Path to input video or folder with frames
        mask_path: Path to mask image/video or folder with mask frames
        output_dir: Output directory
        width: Output width (None to keep original)
        height: Output height (None to keep original)
        use_fp16: Use half precision for memory efficiency
        neighbor_length: Number of local neighbors (lower = less memory)
        ref_stride: Stride for global references (higher = less memory)
        subvideo_length: Frames per sub-video (lower = less memory)
    """
    
    # Build command
    cmd = [
        "python", "inference_propainter.py",
        "--video", video_path,
        "--mask", mask_path,
        "--neighbor_length", str(neighbor_length),
        "--ref_stride", str(ref_stride),
        "--subvideo_length", str(subvideo_length)
    ]
    
    # Add optional parameters
    if width and height:
        cmd.extend(["--width", str(width), "--height", str(height)])
    
    if use_fp16:
        cmd.append("--fp16")
    
    # Run command
    print(f"Running ProPainter with command:\n{' '.join(cmd)}\n")
    try:
        subprocess.run(cmd, check=True)
        print(f"\nSuccess! Output saved to {output_dir}/")
    except subprocess.CalledProcessError as e:
        print(f"Error running ProPainter: {e}")
        print("\nTip: If you get memory errors, try:")
        print("  - Lower resolution: --width 640 --height 480")
        print("  - Fewer neighbors: --neighbor_length 3")
        print("  - More stride: --ref_stride 20")
        print("  - Shorter subvideos: --subvideo_length 30")

def main():
    parser = argparse.ArgumentParser(
        description="Remove watermarks from videos using ProPainter"
    )
    parser.add_argument("--video", required=True, 
                       help="Path to input video or folder with frames")
    parser.add_argument("--mask", required=True,
                       help="Path to mask image/video or folder with masks")
    parser.add_argument("--width", type=int, default=None,
                       help="Output width (e.g., 640)")
    parser.add_argument("--height", type=int, default=None,
                       help="Output height (e.g., 480)")
    parser.add_argument("--no-fp16", action="store_true",
                       help="Disable fp16 mode (uses more memory)")
    parser.add_argument("--help-masks", action="store_true",
                       help="Show instructions for creating masks")
    
    args = parser.parse_args()
    
    if args.help_masks:
        create_mask_instructions()
        return
    
    # Check if files exist
    if not os.path.exists(args.video):
        print(f"Error: Video path '{args.video}' not found")
        return
    
    if not os.path.exists(args.mask):
        print(f"Error: Mask path '{args.mask}' not found")
        return
    
    # Run ProPainter
    run_propainter(
        video_path=args.video,
        mask_path=args.mask,
        width=args.width,
        height=args.height,
        use_fp16=not args.no_fp16
    )

if __name__ == "__main__":
    # Example usage:
    # python propainter_watermark_removal.py --video input.mp4 --mask watermark_mask.png
    # python propainter_watermark_removal.py --video input.mp4 --mask watermark_mask.png --width 640 --height 480
    # python propainter_watermark_removal.py --help-masks
    
    main()