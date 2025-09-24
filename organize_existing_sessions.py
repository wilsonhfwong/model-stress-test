#!/usr/bin/env python3
"""
Helper script to organize existing test results into session folders
"""
import os
import shutil
import glob
import re

def organize_existing_results():
    # Create test_sessions directory
    sessions_dir = "test_sessions"
    if not os.path.exists(sessions_dir):
        os.makedirs(sessions_dir)
    
    # Find all existing test result files
    json_files = glob.glob("test_results/*.json")
    txt_files = glob.glob("test_results/*.txt")
    
    # Find existing nano banana images
    nano_images = glob.glob("temp_nano_banana_images/*.png")
    
    print(f"📁 Found {len(json_files)} JSON files, {len(txt_files)} analysis files, {len(nano_images)} nano banana images")
    
    # Group files by timestamp
    sessions = {}
    
    # Process JSON and TXT files
    for file_path in json_files + txt_files:
        filename = os.path.basename(file_path)
        # Extract timestamp from filename like test_plan_20250924_124055.json
        match = re.search(r'(\d{8}_\d{6})', filename)
        if match:
            session_id = match.group(1)
            if session_id not in sessions:
                sessions[session_id] = {'json': [], 'txt': [], 'images': []}
            
            if filename.endswith('.json'):
                sessions[session_id]['json'].append(file_path)
            elif filename.endswith('.txt'):
                sessions[session_id]['txt'].append(file_path)
    
    # Process nano banana images (match by timestamp)
    for image_path in nano_images:
        filename = os.path.basename(image_path)
        # Extract timestamp from filename like nano_banana_txt2img_1758688828025_1.png
        match = re.search(r'_(\d{13})_', filename)
        if match:
            # Convert millisecond timestamp to session format
            timestamp_ms = int(match.group(1))
            timestamp_s = timestamp_ms / 1000
            from datetime import datetime
            dt = datetime.fromtimestamp(timestamp_s)
            session_id = dt.strftime("%Y%m%d_%H%M%S")
            
            # Find closest session (within same minute)
            best_session = None
            for existing_session in sessions.keys():
                if existing_session[:13] == session_id[:13]:  # Same day, hour, minute
                    best_session = existing_session
                    break
            
            if best_session:
                sessions[best_session]['images'].append(image_path)
            else:
                # Create new session for orphaned images
                if session_id not in sessions:
                    sessions[session_id] = {'json': [], 'txt': [], 'images': []}
                sessions[session_id]['images'].append(image_path)
    
    # Move files to session directories
    for session_id, files in sessions.items():
        session_dir = os.path.join(sessions_dir, session_id)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        print(f"\n📂 Session {session_id}:")
        
        # Move JSON and TXT files
        for file_path in files['json'] + files['txt']:
            dest = os.path.join(session_dir, os.path.basename(file_path))
            shutil.move(file_path, dest)
            print(f"  📄 {os.path.basename(file_path)} → {dest}")
        
        # Move images to nano_banana_images subdirectory
        if files['images']:
            images_dir = os.path.join(session_dir, "nano_banana_images")
            if not os.path.exists(images_dir):
                os.makedirs(images_dir)
            
            for image_path in files['images']:
                dest = os.path.join(images_dir, os.path.basename(image_path))
                shutil.move(image_path, dest)
                print(f"  🖼️  {os.path.basename(image_path)} → {dest}")
    
    print(f"\n✅ Organized {len(sessions)} sessions into test_sessions/")
    print("📁 You can now find all session results in test_sessions/[session_id]/")

if __name__ == "__main__":
    organize_existing_results()