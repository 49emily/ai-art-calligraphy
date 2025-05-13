import os
import sys


def main():
    # Define the paths to the A and B directories
    path_a = "datasets/jian2/train/A"
    path_b = "datasets/jian2/train/B"

    # Get the list of filenames in both directories
    files_a = set(os.listdir(path_a))
    files_b = set(os.listdir(path_b))

    # Filter out hidden files (like .DS_Store)
    files_a = {f for f in files_a if not f.startswith(".")}
    files_b = {f for f in files_b if not f.startswith(".")}

    # Find files in A that don't have a match in B
    files_to_remove = files_a - files_b

    # Count files
    total_files_a = len(files_a)
    matched_files = len(files_a) - len(files_to_remove)

    print(f"Total files in A: {total_files_a}")
    print(f"Files in A with matching names in B: {matched_files}")
    print(f"Files to remove from A: {len(files_to_remove)}")

    # Remove files from A that don't have matching names in B
    for filename in files_to_remove:
        file_path = os.path.join(path_a, filename)
        try:
            os.remove(file_path)
            print(f"Removed: {filename}")
        except Exception as e:
            print(f"Error removing {filename}: {e}")

    print(
        f"Finished. Kept {matched_files} files in directory A that have matching names in B."
    )


if __name__ == "__main__":
    main()
