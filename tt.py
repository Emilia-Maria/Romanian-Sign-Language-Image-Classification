import os

def rename_files(base_dir):
    """
    Renames files in the dataset directories (train, validation, test) to the pattern 
    'folder_name' followed by an increasing number.

    Parameters:
    base_dir (str): Base directory containing the 'train', 'validation', and 'test' subdirectories.
    """
    for split in ['train', 'validation', 'test']:
        split_dir = os.path.join(base_dir, split)
        if not os.path.exists(split_dir):
            print(f"Directory {split_dir} does not exist.")
            continue

        for folder_name in os.listdir(split_dir):
            folder_path = os.path.join(split_dir, folder_name)
            if not os.path.isdir(folder_path):
                continue

            files = os.listdir(folder_path)
            files.sort()  # Ensure consistent ordering
            for i, filename in enumerate(files):
                file_extension = os.path.splitext(filename)[1]
                new_name = f"{folder_name}{i+1}{file_extension}"
                src = os.path.join(folder_path, filename)
                dst = os.path.join(folder_path, new_name)
                os.rename(src, dst)
                print(f"Renamed {src} to {dst}")

# Example usage:
base_directory = "Ro_Sign_language_Dataset"  # Update this path to your dataset directory
rename_files(base_directory)
