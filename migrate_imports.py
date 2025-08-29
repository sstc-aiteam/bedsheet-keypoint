#!/usr/bin/env python3
"""
Migration script to update old imports to use the new src structure.
"""

import os
import re
from pathlib import Path

def update_imports_in_file(file_path):
    """Update imports in a single file."""
    
    print(f"📝 Updating imports in: {file_path}")
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Update imports
    replacements = [
        # Old imports to new src structure
        (r'from models\.utils import \*', 'from src.utils import *'),
        (r'from models\.utils import', 'from src.utils import'),
        (r'from models\.yolo_vit import', 'from src.models import'),
        (r'from models\.quantization_utils import', 'from src.utils import'),
        (r'from models\.losses import', 'from src.utils import'),
        (r'from models\.unet import', '# from models.unet import  # Migrated to src structure'),
        (r'from models\.yolo_cnn import', '# from models.yolo_cnn import  # Migrated to src structure'),
        
        # Add src to path if needed
        (r'import sys', 'import sys\nimport os\nsys.path.append(os.path.join(os.path.dirname(__file__), "src"))'),
    ]
    
    for old_pattern, new_pattern in replacements:
        content = re.sub(old_pattern, new_pattern, content)
    
    # Only write if content changed
    if content != original_content:
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"  ✅ Updated: {file_path}")
        return True
    else:
        print(f"  ⏭️ No changes needed: {file_path}")
        return False

def migrate_all_files():
    """Migrate all Python files to use the new src structure."""
    
    print("🔄 Starting import migration to src structure...")
    print("=" * 60)
    
    # Files to migrate
    files_to_migrate = [
        'depth_keypoint_model_training.py',
        'realsense_bedsheet_detect.py', 
        'test_improved_masking.py',
        'test_keypoint_matching.py',
        'test_training_process.py',
        'run_optimal_training.py'
    ]
    
    updated_count = 0
    
    for file_name in files_to_migrate:
        if os.path.exists(file_name):
            if update_imports_in_file(file_name):
                updated_count += 1
        else:
            print(f"  ⚠️ File not found: {file_name}")
    
    print("\n" + "=" * 60)
    print(f"🎉 Migration completed! Updated {updated_count} files.")
    print("=" * 60)
    
    print("\n📋 Next Steps:")
    print("1. Test the updated files to ensure they work")
    print("2. Consider removing old model files if no longer needed")
    print("3. Update any remaining Jupyter notebooks manually")
    
    return updated_count

def list_unused_files():
    """List files that might be safe to remove."""
    
    print("\n🗑️ Files that might be safe to remove:")
    print("=" * 40)
    
    potentially_unused = [
        'models/yolo_cnn.py',           # Commented out in notebooks
        'models/unet.py',               # Commented out in notebooks  
        'models/flash_attn_test.py',    # Test file only
        'models/losses.py',             # Already removed from src
    ]
    
    for file_path in potentially_unused:
        if os.path.exists(file_path):
            print(f"  📁 {file_path}")
        else:
            print(f"  ❌ {file_path} (already removed)")
    
    print("\n⚠️ Note: Check these files manually before removing!")

if __name__ == "__main__":
    print("🚀 Import Migration Tool")
    print("=" * 60)
    
    # Run migration
    updated_count = migrate_all_files()
    
    # List potentially unused files
    list_unused_files()
    
    if updated_count > 0:
        print(f"\n✅ Successfully migrated {updated_count} files!")
        print("🧪 Please test the updated files to ensure they work correctly.")
    else:
        print("\n⏭️ No files needed migration.")
