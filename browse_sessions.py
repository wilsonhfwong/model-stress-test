#!/usr/bin/env python3
"""
Browse test sessions and their contents
"""
import os
import glob
import json
from datetime import datetime

def browse_sessions():
    sessions_dir = "test_sessions"
    
    if not os.path.exists(sessions_dir):
        print("❌ No test_sessions directory found. Run some tests first!")
        return
    
    # Get all session directories
    session_dirs = [d for d in os.listdir(sessions_dir) if os.path.isdir(os.path.join(sessions_dir, d))]
    session_dirs.sort(reverse=True)  # Newest first
    
    if not session_dirs:
        print("❌ No sessions found in test_sessions/")
        return
    
    print(f"📁 Found {len(session_dirs)} test sessions:")
    print("=" * 80)
    
    for i, session_id in enumerate(session_dirs, 1):
        session_path = os.path.join(sessions_dir, session_id)
        
        # Parse session timestamp
        try:
            dt = datetime.strptime(session_id, "%Y%m%d_%H%M%S")
            time_str = dt.strftime("%Y-%m-%d %H:%M:%S")
        except:
            time_str = session_id
        
        print(f"{i:2d}. Session {session_id} ({time_str})")
        
        # Count files
        json_files = glob.glob(os.path.join(session_path, "*.json"))
        txt_files = glob.glob(os.path.join(session_path, "*_analysis.txt"))
        
        nano_images_dir = os.path.join(session_path, "nano_banana_images")
        nano_images = []
        if os.path.exists(nano_images_dir):
            nano_images = glob.glob(os.path.join(nano_images_dir, "*.png"))
        
        print(f"    📄 {len(json_files)} JSON file{'s' if len(json_files) != 1 else ''}")
        print(f"    📊 {len(txt_files)} analysis file{'s' if len(txt_files) != 1 else ''}")
        print(f"    🖼️  {len(nano_images)} Nano Banana image{'s' if len(nano_images) != 1 else ''}")
        
        # Try to read test info from JSON
        if json_files:
            try:
                with open(json_files[0], 'r') as f:
                    data = json.load(f)
                    
                plan_a_count = len(data.get('test_plan_a', []))
                plan_b_count = len(data.get('test_plan_b', []))
                duration = data.get('total_duration', 0)
                
                print(f"    ⏱️  Duration: {duration:.1f}s")
                if plan_a_count > 0:
                    print(f"    🅰️  Plan A: {plan_a_count} test{'s' if plan_a_count != 1 else ''}")
                if plan_b_count > 0:
                    print(f"    🅱️  Plan B: {plan_b_count} test{'s' if plan_b_count != 1 else ''}")
                    
            except Exception as e:
                print(f"    ⚠️  Could not read test info: {e}")
        
        print(f"    📁 Path: {session_path}")
        print()
    
    print("💡 To open a session folder:")
    print("   - macOS: open test_sessions/[session_id]/")
    print("   - Linux: nautilus test_sessions/[session_id]/")
    print("   - Windows: explorer test_sessions\\[session_id]\\")

if __name__ == "__main__":
    browse_sessions()