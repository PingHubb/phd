# import numpy as np
#
# # Parameters for the half-cylinder
# radius = 42.5  # Radius of the cylinder in mm (half of the diameter 85mm)
# height = 100.0  # Height of the cylinder in mm
# N_theta = 80  # Number of divisions along the theta (angular) direction
# N_y = 75  # Number of divisions along the y (height) direction
#
# # Create arrays for theta and y
# theta = np.linspace(0, np.pi, N_theta)  # Theta from 0 to pi (half-cylinder)
# y = np.linspace(-height / 2, height / 2, N_y)  # y from -h/2 to h/2
#
# # Create a meshgrid for theta and y
# theta_grid, y_grid = np.meshgrid(theta, y)
#
# # Compute x, y, z coordinates
# x_grid = radius * np.cos(theta_grid)
# z_grid = radius * np.sin(theta_grid)
# y_grid = y_grid  # y remains the same
#
# # Compute normals (they are radial and same as the x and z components of the position, normalized)
# normals_x = np.cos(theta_grid)
# normals_y = np.zeros_like(y_grid)
# normals_z = np.sin(theta_grid)
#
# # Flatten the arrays to create lists of vertices and normals
# vertices = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
# normals = np.column_stack((normals_x.flatten(), normals_y.flatten(), normals_z.flatten()))
#
# # Write to .obj file
# with open('half_cylinder.obj', 'w') as file:
#     # Write vertices
#     for v in vertices:
#         file.write(f"v {v[0]} {v[1]} {v[2]}\n")
#
#     # Write vertex normals
#     for n in normals:
#         file.write(f"vn {n[0]} {n[1]} {n[2]}\n")
#
#     # Write faces
#     for i in range(N_y - 1):
#         for j in range(N_theta - 1):
#             # Calculate vertex indices for the two triangles of each quad
#             v1 = i * N_theta + j + 1
#             v2 = v1 + N_theta
#             v3 = v2 + 1
#             v4 = v1 + 1
#
#             # First triangle
#             file.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
#             # Second triangle
#             file.write(f"f {v1}//{v1} {v3}//{v3} {v4}//{v4}\n")
#
# # Now, generate the group numbers
# # Define the group boundaries along y and theta
#
# # Divide the surface into 10 divisions along the y-axis (height)
# y_group_boundaries = [0, 7, 14, 21, 28, 35, 43, 51, 59, 67, 75]  # 10 intervals
#
# # Divide the surface into 11 divisions along the theta-axis (angular direction)
# theta_group_boundaries = [0, 7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 80]  # 11 intervals
#
# # Initialize an array to hold group numbers
# group_numbers = np.zeros((N_y, N_theta), dtype=int)
#
# # Assign group numbers to each grid cell
# for y_group_index in range(10):  # y_group_index from 0 to 9
#     y_start = y_group_boundaries[y_group_index]
#     y_end = y_group_boundaries[y_group_index + 1]
#     for theta_group_index in range(11):  # theta_group_index from 0 to 10
#         theta_start = theta_group_boundaries[theta_group_index]
#         theta_end = theta_group_boundaries[theta_group_index + 1]
#         # Compute group number
#         group_number = y_group_index * 11 + theta_group_index  # group numbers from 1 to 110
#         # Assign group number to the appropriate indices
#         group_numbers[y_start:y_end, theta_start:theta_end] = group_number
#
# # Flatten the group_numbers array to match the vertices array
# group_numbers_flat = group_numbers.flatten()
#
# # Write group numbers to txt file
# with open('vertex_groups.txt', 'w') as f:
#     for group_num in group_numbers_flat:
#         f.write(f"{group_num}\n")




import numpy as np

# Parameters for the half-cylinder
radius = 42.5/500  # Radius of the cylinder in mm (half of the diameter 85mm)
height = 100.0/500  # Height of the cylinder in mm
N_theta = 80  # Number of divisions along the theta (angular) direction
N_y = 75  # Number of divisions along the y (height) direction

# Create arrays for theta and y
theta = np.linspace(0, np.pi, N_theta)  # Theta from 0 to pi (half-cylinder)
y = np.linspace(-height / 2, height / 2, N_y)  # y from -h/2 to h/2

# Create a meshgrid for theta and y
theta_grid, y_grid = np.meshgrid(theta, y)

# Compute x, y, z coordinates
x_grid = radius * np.cos(theta_grid)
z_grid = radius * np.sin(theta_grid)
y_grid = y_grid  # y remains the same

# Compute normals (they are radial and same as the x and z components of the position, normalized)
normals_x = np.cos(theta_grid)
normals_y = np.zeros_like(y_grid)
normals_z = np.sin(theta_grid)

# Flatten the arrays to create lists of vertices and normals
vertices = np.column_stack((x_grid.flatten(), y_grid.flatten(), z_grid.flatten()))
normals = np.column_stack((normals_x.flatten(), normals_y.flatten(), normals_z.flatten()))

# Write to .obj file
with open('../resource/sensor/half_cylinder_surface/half_cylinder_2.obj', 'w') as file:
    # Write vertices
    for v in vertices:
        file.write(f"v {v[0]} {v[1]} {v[2]}\n")

    # Write vertex normals
    for n in normals:
        file.write(f"vn {n[0]} {n[1]} {n[2]}\n")

    # Write faces
    for i in range(N_y - 1):
        for j in range(N_theta - 1):
            # Calculate vertex indices for the two triangles of each quad
            v1 = i * N_theta + j + 1
            v2 = v1 + N_theta
            v3 = v2 + 1
            v4 = v1 + 1

            # First triangle
            file.write(f"f {v1}//{v1} {v2}//{v2} {v3}//{v3}\n")
            # Second triangle
            file.write(f"f {v1}//{v1} {v3}//{v3} {v4}//{v4}\n")

# Now, generate the group numbers
# Divide the surface into 10 divisions along the y-axis (height)
y_divisions = 9
y_group_size = N_y // y_divisions
y_remainder = N_y % y_divisions
y_group_boundaries = [0]

# Adjust boundaries to distribute any remainder
for i in range(y_divisions):
    additional_row = 1 if i < y_remainder else 0
    y_group_boundaries.append(y_group_boundaries[-1] + y_group_size + additional_row)

# Divide the surface into 10 divisions along the theta-axis (angular direction)
theta_divisions = 10
theta_group_size = N_theta // theta_divisions
theta_remainder = N_theta % theta_divisions
theta_group_boundaries = [0]

# Adjust boundaries to distribute any remainder
for i in range(theta_divisions):
    additional_col = 1 if i < theta_remainder else 0
    theta_group_boundaries.append(theta_group_boundaries[-1] + theta_group_size + additional_col)

# Initialize an array to hold group numbers
group_numbers = np.zeros((N_y, N_theta), dtype=int)

# Assign group numbers to each grid cell
group_number = 0  # Start group numbering from 0
for y_group_index in range(y_divisions):
    y_start = y_group_boundaries[y_group_index]
    y_end = y_group_boundaries[y_group_index + 1]
    for theta_group_index in range(theta_divisions):
        theta_start = theta_group_boundaries[theta_group_index]
        theta_end = theta_group_boundaries[theta_group_index + 1]
        # Assign group number to the appropriate indices
        group_numbers[y_start:y_end, theta_start:theta_end] = group_number
        group_number += 1  # Increment group number

# Flatten the group_numbers array to match the vertices array
group_numbers_flat = group_numbers.flatten()

# Write group numbers to txt file
with open('../resource/sensor/half_cylinder_surface/vertex_groups_2.txt', 'w') as f:
    for group_num in group_numbers_flat:
        f.write(f"{group_num}\n")
