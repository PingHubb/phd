import os
from pathlib import Path

gesture_number = "cylinder_1"
RESOURCE_ROOT = Path(__file__).resolve().parents[1] / "resource"

# Define the folder path
folder_path = str(RESOURCE_ROOT / "ai" / "data" / "offset" / f"gesture_{gesture_number}")


# Function to find .txt files with fewer than 10 lines
def find_files_with_few_lines(folder):
    # List to store files with fewer than 10 lines
    files_with_few_lines = []

    # Iterate over all files in the folder
    for file_name in os.listdir(folder):
        # Check if the file is a .txt file
        if file_name.endswith(".txt"):
            file_path = os.path.join(folder, file_name)

            # Count the number of lines in the file
            with open(file_path, 'r') as file:
                lines = file.readlines()

                # If the file has fewer than 10 lines, add it to the list
                if len(lines) < 30:
                    files_with_few_lines.append(file_name)

    return files_with_few_lines


# Find and print .txt files with fewer than 10 lines
txt_files = find_files_with_few_lines(folder_path)
if txt_files:
    print("Files with fewer than X lines:")
    for txt_file in txt_files:
        print(txt_file)
else:
    print("No files with fewer than X lines found.")
