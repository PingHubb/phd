import os
import re

gesture_number = 3

def rename_files_in_folder(folder_path):
    # List all .txt files in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

    # Create a list of tuples (extracted_number, filename)
    file_tuples = []
    for file in files:
        match = re.search(r'(\d+)', file)
        if match:
            number = int(match.group(1))
            file_tuples.append((number, file))
        else:
            print(f"No number found in filename: {file}")

    # Sort the list based on the extracted number (smallest to largest)
    file_tuples.sort(key=lambda x: x[0])

    # Rename files sequentially
    for index, (_, original_file) in enumerate(file_tuples, start=1):
        new_name = f"{index}.txt"
        src = os.path.join(folder_path, original_file)
        dst = os.path.join(folder_path, new_name)
        print(f"Renaming '{original_file}' to '{new_name}'")
        os.rename(src, dst)


if __name__ == "__main__":
    # Define the folder path using your provided format
    folder_path = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{gesture_number}"

    if os.path.isdir(folder_path):
        rename_files_in_folder(folder_path)
    else:
        print(f"Folder '{folder_path}' does not exist.")
