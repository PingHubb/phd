import os

number = 3
# Define your input and output folder paths
gesture_diff_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/offset/gesture_{number}"
output_dir = f"/home/ping2/ros2_ws/src/phd/phd/resource/ai/data/discrete/gesture_{number}"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Function to process a file
def process_file(file_path, output_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    # Open output file to write transformed data
    with open(output_path, 'w') as output_file:
        for line in lines:
            # Split the line into individual elements
            elements = line.strip().split()
            # Convert elements to float, apply transformation (1 if < -1, otherwise 0), and convert back to string
            transformed_elements = [
                '2' if float(el) > 2 else
                '1' if float(el) < -1 else
                '0.2' if float(el) < -0.2 else
                '0' if float(el) >= -0.2 else el
                for el in elements
            ]
            # Join the transformed elements and write to output file
            output_file.write(' '.join(transformed_elements) + '\n')


# Iterate through all the txt files in the input directory
for file_name in os.listdir(gesture_diff_dir):
    if file_name.endswith(".txt"):
        input_file_path = os.path.join(gesture_diff_dir, file_name)
        output_file_path = os.path.join(output_dir, file_name)
        # Process and transform the file
        process_file(input_file_path, output_file_path)

print("All files have been processed and saved to the output directory.")

