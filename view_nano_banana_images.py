#!/usr/bin/env python3
"""
Helper script to view the latest Nano Banana generated images
"""
import os
import glob
from PIL import Image

def view_latest_images():
    temp_dir = "temp_nano_banana_images"
    
    if not os.path.exists(temp_dir):
        print(f"❌ Directory {temp_dir} not found. Run some tests first!")
        return
    
    # Get all PNG files sorted by modification time (newest first)
    image_files = glob.glob(os.path.join(temp_dir, "*.png"))
    
    if not image_files:
        print(f"❌ No images found in {temp_dir}")
        return
    
    # Sort by modification time, newest first
    image_files.sort(key=os.path.getmtime, reverse=True)
    
    print(f"📁 Found {len(image_files)} Nano Banana generated images:")
    print("=" * 60)
    
    for i, image_path in enumerate(image_files, 1):
        filename = os.path.basename(image_path)
        file_size = os.path.getsize(image_path)
        file_size_mb = file_size / (1024 * 1024)
        
        # Parse filename to get info
        parts = filename.replace('.png', '').split('_')
        if len(parts) >= 4:
            task_type = parts[2]  # txt2img or img2img
            timestamp = parts[3]
            image_num = parts[4] if len(parts) > 4 else "1"
            
            task_label = "Text-to-Image" if task_type == "txt2img" else "Image-to-Image"
            
            print(f"{i:2d}. {task_label} - {filename}")
            print(f"    📁 Path: {image_path}")
            print(f"    📊 Size: {file_size_mb:.1f} MB")
            
            # Try to get image dimensions
            try:
                with Image.open(image_path) as img:
                    print(f"    🖼️  Dimensions: {img.size[0]}x{img.size[1]} ({img.mode})")
            except Exception as e:
                print(f"    ⚠️  Could not read image info: {e}")
            
            print()
    
    print("💡 To view images:")
    print("   - On macOS: open temp_nano_banana_images/")
    print("   - On Linux: nautilus temp_nano_banana_images/ (or your file manager)")
    print("   - On Windows: explorer temp_nano_banana_images\\")

if __name__ == "__main__":
    view_latest_images()