#!/usr/bin/env python3
"""
Test script for NBA analysis using sample data.
This script temporarily renames sample files to test the main analysis.
"""

import os
import shutil
import sys

# Backup original files if they exist
backups = {}
files_to_backup = ['Regular_Season.csv', 'Playoffs.csv', 'nba.csv']

print("Setting up test environment...")
for filename in files_to_backup:
    if os.path.exists(filename):
        backup_name = f"{filename}.backup"
        shutil.copy2(filename, backup_name)
        backups[filename] = backup_name
        print(f"  Backed up {filename} to {backup_name}")

# Copy sample files to test locations
print("\nCopying sample data files...")
shutil.copy2('sample_Regular_Season.csv', 'Regular_Season.csv')
shutil.copy2('sample_Playoffs.csv', 'Playoffs.csv')
shutil.copy2('sample_nba.csv', 'nba.csv')
print("  Sample files copied")

# Run the main analysis
print("\n" + "="*70)
print("RUNNING NBA ANALYSIS WITH SAMPLE DATA")
print("="*70 + "\n")

try:
    # Import and run main analysis
    import nba_analysis
    nba_analysis.main()
    print("\nTest completed successfully!")

except Exception as e:
    print(f"\nError during analysis: {e}")
    import traceback
    traceback.print_exc()

finally:
    # Restore original files
    print("\n" + "="*70)
    print("Restoring original files...")
    for filename, backup_name in backups.items():
        if os.path.exists(backup_name):
            shutil.copy2(backup_name, filename)
            os.remove(backup_name)
            print(f"  Restored {filename}")

    # Clean up test files if originals didn't exist
    for filename in files_to_backup:
        if filename not in backups and os.path.exists(filename):
            # This was created by test, check if we should remove it
            # For now, keep them as they might be useful
            pass

    print("Cleanup complete")
    print("="*70)
