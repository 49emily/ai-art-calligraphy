import os
import shutil
from pathlib import Path


def rename_files():
    # Paths to the original directories
    dir_a = Path("datasets/jiannobg/train/A")
    dir_b = Path("datasets/jiannobg/train/B")

    # Paths to the new directories
    new_dir_a = Path("datasets/jiannobg_renamed/train/A")
    new_dir_b = Path("datasets/jiannobg_renamed/train/B")

    # Create new directories if they don't exist
    new_dir_a.mkdir(parents=True, exist_ok=True)
    new_dir_b.mkdir(parents=True, exist_ok=True)

    # Get the list of files in both directories
    files_a = [f.name for f in dir_a.iterdir() if f.is_file()]
    files_b = [f.name for f in dir_b.iterdir() if f.is_file()]

    # Create a set of all unique filenames
    all_files = sorted(set(files_a + files_b))

    # Create a mapping from old filenames to new filenames
    file_mapping = {filename: f"{i:04d}.png" for i, filename in enumerate(all_files)}

    # Copy and rename files in directory A
    for old_name in files_a:
        if old_name in file_mapping:
            new_name = file_mapping[old_name]
            old_path = dir_a / old_name
            new_path = new_dir_a / new_name

            # Copy the file with the new name
            shutil.copy2(old_path, new_path)
            print(f"Copied to new directory A: {old_name} -> {new_name}")

    # Copy and rename files in directory B
    for old_name in files_b:
        if old_name in file_mapping:
            new_name = file_mapping[old_name]
            old_path = dir_b / old_name
            new_path = new_dir_b / new_name

            # Copy the file with the new name
            shutil.copy2(old_path, new_path)
            print(f"Copied to new directory B: {old_name} -> {new_name}")

    print(f"Total files processed: {len(all_files)}")
    print(f"Files copied to new directory A: {len(files_a)}")
    print(f"Files copied to new directory B: {len(files_b)}")


if __name__ == "__main__":
    rename_files()
