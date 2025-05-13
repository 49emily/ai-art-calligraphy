#!/usr/bin/env python3
# Script to create the index.pkl file for the reference directory

import os
import pickle
from glob import glob


def create_reference_index(reference_dir="./assets/reference"):
    """
    Create an index.pkl file for the reference directory.

    The index is structured as a list of lists, where each sublist contains
    the filenames of characters for a specific writer.
    """
    # Get all PNG files in the reference directory
    ref_files = glob(os.path.join(reference_dir, "*.png"))

    # Create a list of relative paths for all reference files
    relative_paths = [os.path.basename(f) for f in ref_files]

    # Since we don't have actual writer info, we'll treat all files as coming from one writer
    # In a real dataset, this would be organized by writer
    index = [relative_paths]

    # Save the index
    with open(os.path.join(reference_dir, "index.pkl"), "wb") as f:
        pickle.dump(index, f)

    print(f"Created index.pkl with {len(relative_paths)} reference images")
    return index


if __name__ == "__main__":
    create_reference_index()
