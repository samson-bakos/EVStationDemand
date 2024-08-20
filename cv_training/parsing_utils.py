import pandas as pd
import os
import glob
import tarfile
import shutil


def parse_annotation_line(line):
    """Parse a single line of annotations"""
    parts = line.strip().split()

    # If too short fill missing parts with None
    if len(parts) < 14:
        parts.extend([None] * (14 - len(parts)))

    # Parse components if available
    center_x = float(parts[0]) if parts[0] is not None else None
    center_y = float(parts[1]) if parts[1] is not None else None
    orientation = float(parts[2]) if parts[2] is not None else None
    class_label = int(parts[3]) if parts[3] is not None else None
    occluded = int(parts[4]) if parts[4] is not None else None
    fully_contained = int(parts[5]) if parts[5] is not None else None
    corners = [int(p) if p is not None else None for p in parts[6:14]]

    return {
        "center_x": center_x,
        "center_y": center_y,
        "orientation": orientation,
        "class_label": class_label,
        "occluded": occluded,
        "fully_contained": fully_contained,
        "corner1_x": corners[0],
        "corner1_y": corners[1],
        "corner2_x": corners[2],
        "corner2_y": corners[3],
        "corner3_x": corners[4],
        "corner3_y": corners[5],
        "corner4_x": corners[6],
        "corner4_y": corners[7],
    }


def lines_to_df(annotation_lines):
    """Parse Multiple Lines into a DataFrame"""
    return pd.DataFrame([parse_annotation_line(line) for line in annotation_lines])


def parse_annotations_to_dict(annotations_dir):
    """
    Parse all annotation files in the specified directory into a dictionary.

    Args:
    - annotations_dir (str): Directory path containing annotation files.

    Returns:
    - annotations_dict (dict): Dictionary where keys are image numbers and values
                              are dictionaries containing 'annotation_df' (DataFrame)
                              and 'df_length' (length of DataFrame).
    """
    annotations_dict = {}

    for filename in os.listdir(annotations_dir):
        if filename.startswith("0000") and filename.endswith(".txt"):
            filepath = os.path.join(annotations_dir, filename)

            image_number = filename.split(".")[0]

            with open(filepath, "r") as f:
                annotation_lines = f.readlines()

            df_annotations = lines_to_df(annotation_lines)

            annotations_dict[image_number] = {
                "annotation_df": df_annotations,
                "df_length": len(df_annotations),
            }

    return annotations_dict


def combine_split_files(source_dir, base_filename, combined_filename):
    """Combine split tar files into a single tar file."""
    combined_filepath = os.path.join(source_dir, combined_filename)
    with open(combined_filepath, "wb") as outfile:
        part_files = sorted(
            glob.glob(os.path.join(source_dir, f"{base_filename}.tar.*"))
        )
        for part_file in part_files:
            print(f"Combining {part_file}")
            with open(part_file, "rb") as infile:
                shutil.copyfileobj(infile, outfile)
    return combined_filepath


def extract_tar_file(tar_path, extract_path):
    """Extracts a tar file to a specified directory without overwriting existing files."""
    try:
        with tarfile.open(tar_path, "r") as tar:
            members = tar.getmembers()
            for member in members:
                # Construct the full path for the extracted file
                extracted_file_path = os.path.join(extract_path, member.name)
                # Check if the file already exists
                if not os.path.exists(extracted_file_path):
                    tar.extract(member, extract_path)
                    print(f"Extracted {member.name} to {extract_path}")
                else:
                    print(f"Skipped {member.name} as it already exists.")
    except tarfile.TarError as e:
        print(f"Failed to extract {tar_path}: {e}")


def handle_split_tars(source_dir, dest_dir, base_filename, combined_filename):
    """Handle the process of combining and extracting split tar files."""
    combined_tar_path = combine_split_files(
        source_dir, base_filename, combined_filename
    )
    extract_tar_file(combined_tar_path, dest_dir)
