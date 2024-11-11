import os

gesture_number = 15
# Folder path where the text files are stored
folder_path = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/diff/gesture_{gesture_number}"

# Loop over the original filenames from 181 to 200
for old_number in range(351, 451):
    # Define the old and new file names
    old_filename = f"{old_number}.txt"
    new_number = old_number - 350
    new_filename = f"{new_number}.txt"

    # Full path of the old and new files
    old_file_path = os.path.join(folder_path, old_filename)
    new_file_path = os.path.join(folder_path, new_filename)

    # Rename the file
    if os.path.exists(old_file_path):
        os.rename(old_file_path, new_file_path)
        print(f"Renamed {old_filename} to {new_filename}")
    else:
        print(f"{old_filename} not found!")

print("Renaming complete.")
