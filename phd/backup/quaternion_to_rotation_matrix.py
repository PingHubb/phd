import numpy as np


def quaternion_to_rotation_matrix(w, x, y, z):
    # Compute the rotation matrix from quaternion
    R = np.array([
        [1 - 2 * y ** 2 - 2 * z ** 2, 2 * x * y - 2 * z * w, 2 * x * z + 2 * y * w],
        [2 * x * y + 2 * z * w, 1 - 2 * x ** 2 - 2 * z ** 2, 2 * y * z - 2 * x * w],
        [2 * x * z - 2 * y * w, 2 * y * z + 2 * x * w, 1 - 2 * x ** 2 - 2 * y ** 2]
    ])
    return R


def create_homogeneous_matrix(px, py, pz, w, x, y, z):
    # Get rotation matrix from quaternion
    R = quaternion_to_rotation_matrix(w, x, y, z)

    # Create the homogeneous transformation matrix
    T = np.zeros((4, 4))
    T[:3, :3] = R
    T[:3, 3] = [px, py, pz]
    T[3, 3] = 1
    return T


def main():
    # Define all the positions and orientations
    positions_orientations = [
        ("3D Printed part - Button Left", (-0.294, 0.546, 0.137, 0.000, -1.000, -0.000, -0.000)),
        ("3D Printed part - Button Right", (-0.146, 0.690, 0.134, 0.000, -1.000, -0.000, -0.000)),
        ("3D Printed part - Top Right", (-0.285, 0.838, 0.134, 0.000, -1.000, -0.000, -0.000)),
        ("3D Printed part - Top Left", (-0.434, 0.695, 0.137, 0.000, -1.000, -0.000, -0.000)),
        ("Metal part - Button Left", (-0.294, 0.546, 0.144, 0.000, -1.000, -0.000, -0.000)),
        ("Metal part - Button Right", (-0.146, 0.690, 0.142, 0.000, -1.000, -0.000, -0.000)),
        ("Metal part - Top Right", (-0.285, 0.838, 0.142, 0.000, -1.000, -0.000, -0.000)),
        ("Metal part - Top Left", (-0.434, 0.695, 0.144, 0.000, -1.000, -0.000, -0.000)),
        ("Starting position", (-0.285, 0.700, 0.249, 0.000, -1.000, -0.000, 0.000)),
        ("Ending position", (-0.300, 0.750, 0.202, 0.000, 1.000, -0.000, 0.000))
    ]

    # File to save the transformation matrices
    with open("/phd/resource/singa_request/dualC/matrix.txt", 'w') as file:
        for label, (px, py, pz, w, x, y, z) in positions_orientations:
            T = create_homogeneous_matrix(px, py, pz, w, x, y, z)
            file.write(f"{label}:\n{np.array2string(T, precision=3, suppress_small=True)}\n\n")


if __name__ == "__main__":
    main()
