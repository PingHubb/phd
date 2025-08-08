# import numpy as np
#
# # Parameters for the half-cylinder
# radius = 0.2125  # Radius of the cylinder in mm (half of the diameter 85mm)
# height = 0.5  # Height of the cylinder in mm
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



import numpy as np

# Parameters for the half-cylinder
radius = 42.5 / 500  # Radius of the cylinder in meters
height = 100.0 / 500  # Height of the cylinder in meters
N_theta = 80  # Number of divisions along the theta (angular) direction
N_y = 75      # Number of divisions along the y (height) direction

# Create arrays for theta and y
theta = np.linspace(0, np.pi, N_theta)  # Theta from 0 to pi (half-cylinder)
y = np.linspace(-height / 2, height / 2, N_y)  # y from -h/2 to h/2

# Create a meshgrid for theta and y
theta_grid, y_grid = np.meshgrid(theta, y)

# Compute x, y, z coordinates
x_grid = radius * np.cos(theta_grid)
z_grid = radius * np.sin(theta_grid)
y_grid = y_grid  # y remains the same

# Compute normals (reverse the direction)
normals_x = -np.cos(theta_grid)
normals_y = -np.zeros_like(y_grid)
normals_z = -np.sin(theta_grid)

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
            # Calculate vertex indices for the four corners of each quad
            v1 = i * N_theta + j + 1         # Original Top-Left
            v2 = v1 + N_theta                # Original Bottom-Left
            v3 = v2 + 1                      # Original Bottom-Right
            v4 = v1 + 1                      # Original Top-Right

            # Swap top-left (v1) with bottom-right (v3)
            v1_new = v3  # New Top-Left (was Bottom-Right)
            v2_new = v2  # Bottom-Left remains the same
            v3_new = v1  # New Bottom-Right (was Top-Left)
            v4_new = v4  # Top-Right remains the same

            # First triangle (adjusted winding order)
            file.write(f"f {v1_new}//{v1_new} {v2_new}//{v2_new} {v3_new}//{v3_new}\n")
            # Second triangle (adjusted winding order)
            file.write(f"f {v1_new}//{v1_new} {v4_new}//{v4_new} {v2_new}//{v2_new}\n")

