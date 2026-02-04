#!/usr/bin/env python3
"""
Room Reconstruction - Command Line Interface

Process room photographs from the command line.

Usage:
    python run_cli.py image1.jpg image2.jpg image3.jpg [options]

Examples:
    # Basic usage with 4 images
    python run_cli.py photos/*.jpg
    
    # Specify room width for better scaling
    python run_cli.py img1.jpg img2.jpg img3.jpg --room-width 5.0
    
    # Open 3D visualization after processing
    python run_cli.py images/*.jpg --visualize
"""

import argparse
import glob
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    parser = argparse.ArgumentParser(
        description="🏠 Room Reconstruction from Photos",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s room_photo1.jpg room_photo2.jpg room_photo3.jpg
  %(prog)s photos/*.jpg --room-width 5.0
  %(prog)s images/*.jpg --visualize

Photo Tips:
  - Take 4-5 photos from different corners of the room
  - Include floor and walls in each shot
  - Use good lighting for best results
"""
    )
    
    parser.add_argument(
        "images", 
        nargs="+", 
        help="Paths to room images (supports glob patterns like *.jpg)"
    )
    parser.add_argument(
        "--room-width", 
        type=float, 
        default=4.0,
        help="Assumed room width in meters (default: 4.0)"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Open interactive 3D visualization after processing"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ./outputs)"
    )
    
    args = parser.parse_args()
    
    # Expand glob patterns
    image_paths = []
    for pattern in args.images:
        expanded = glob.glob(pattern)
        if expanded:
            image_paths.extend(expanded)
        elif os.path.isfile(pattern):
            image_paths.append(pattern)
    
    # Filter to only valid image files
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    image_paths = [
        p for p in image_paths 
        if os.path.splitext(p.lower())[1] in valid_extensions
    ]
    
    if not image_paths:
        print("\n❌ ERROR: No valid image files found!")
        print("\nPlease provide image files as arguments.")
        print("Supported formats: JPG, JPEG, PNG, BMP, WEBP")
        print("\nExample: python run_cli.py photo1.jpg photo2.jpg photo3.jpg")
        sys.exit(1)
    
    if len(image_paths) < 2:
        print("\n⚠️ WARNING: Only 1 image provided.")
        print("For better results, use 4-5 images from different angles.")
        print("")
    
    print("\n" + "="*60)
    print("🏠 Room Reconstruction from Photos")
    print("="*60)
    print(f"\n📷 Found {len(image_paths)} image(s):")
    for i, path in enumerate(image_paths, 1):
        print(f"   {i}. {os.path.basename(path)}")
    print(f"\n📏 Assumed room width: {args.room_width} meters")
    print("\n" + "-"*60)
    
    # Import here to delay model loading
    from modules.room_reconstructor import RoomReconstructor
    from config import OUTPUT_DIR
    
    # Set output directory if specified
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        import config
        config.OUTPUT_DIR = args.output_dir
    
    # Create reconstructor and run
    try:
        reconstructor = RoomReconstructor(assumed_room_width=args.room_width)
        result = reconstructor.reconstruct(image_paths)
        
        print("\n" + "="*60)
        print("✅ RECONSTRUCTION COMPLETE!")
        print("="*60)
        
        print("\n💾 Output files:")
        for name, path in result['outputs'].items():
            print(f"   • {name}: {path}")
        
        # Open visualization if requested
        if args.visualize:
            print("\n🎮 Opening 3D visualization...")
            reconstructor.visualizer.visualize_open3d(
                result['data']['points'],
                result['data']['colors'],
                window_name="Room Reconstruction - Close window to exit"
            )
        
        print("\n👍 Done!\n")
        return 0
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
