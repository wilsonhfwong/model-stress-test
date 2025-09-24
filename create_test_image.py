#!/usr/bin/env python3
"""
Create a simple test image for image-to-image testing
"""
from PIL import Image, ImageDraw, ImageFont
import os

def create_test_image():
    """Create a simple test image"""
    # Create a 1024x1024 image with a simple scene
    width, height = 1024, 1024
    
    # Create a new image with gradient background
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw a simple landscape scene
    # Sky gradient (blue to light blue)
    for y in range(height // 2):
        color_intensity = int(135 + (120 * y / (height // 2)))
        draw.line([(0, y), (width, y)], fill=(color_intensity, color_intensity + 50, 255))
    
    # Ground (green)
    ground_color = (34, 139, 34)
    draw.rectangle([(0, height // 2), (width, height)], fill=ground_color)
    
    # Draw some simple mountains
    mountain_points = [
        (0, height // 2),
        (width // 4, height // 3),
        (width // 2, height // 2.5),
        (3 * width // 4, height // 3.5),
        (width, height // 2)
    ]
    draw.polygon(mountain_points, fill=(105, 105, 105))
    
    # Draw sun
    sun_center = (width - 150, 150)
    sun_radius = 60
    draw.ellipse([
        sun_center[0] - sun_radius, 
        sun_center[1] - sun_radius,
        sun_center[0] + sun_radius, 
        sun_center[1] + sun_radius
    ], fill=(255, 255, 0))
    
    # Add some simple trees
    for tree_x in [200, 400, 600, 800]:
        tree_y = height // 2 + 50
        # Tree trunk
        draw.rectangle([tree_x - 10, tree_y, tree_x + 10, tree_y + 100], fill=(101, 67, 33))
        # Tree crown
        draw.ellipse([tree_x - 40, tree_y - 80, tree_x + 40, tree_y + 20], fill=(0, 128, 0))
    
    # Save the image
    os.makedirs('test_images', exist_ok=True)
    output_path = 'test_images/landscape_input.jpg'
    img.save(output_path, 'JPEG', quality=95)
    print(f"Test image created: {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_test_image()