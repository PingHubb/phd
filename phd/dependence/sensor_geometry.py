from __future__ import annotations

import numpy as np


def normalize_point_array(points, expected_count=None):
    """Return a finite ``(N, 3)`` point array, or ``None`` when invalid."""
    try:
        array = np.asarray(points, dtype=float)
    except Exception:
        return None
    if array.ndim != 2 or array.shape[1] != 3 or not np.all(np.isfinite(array)):
        return None
    if expected_count is not None and array.shape[0] != int(expected_count):
        return None
    return np.array(array, dtype=float, copy=True)


def grid_selection_weights(center_index, n_row, n_col, radius=0, soft=True):
    """Select a circular taxel neighborhood in column-major grid order."""
    n_row = int(n_row)
    n_col = int(n_col)
    center_index = int(center_index)
    radius = max(0, int(radius))
    if n_row <= 0 or n_col <= 0 or not 0 <= center_index < n_row * n_col:
        return np.empty(0, dtype=int), np.empty(0, dtype=float)

    center_col, center_row = divmod(center_index, n_row)
    indices = []
    weights = []
    for col in range(max(0, center_col - radius), min(n_col, center_col + radius + 1)):
        for row in range(max(0, center_row - radius), min(n_row, center_row + radius + 1)):
            distance = float(np.hypot(col - center_col, row - center_row))
            if distance > radius + 1e-9:
                continue
            indices.append(col * n_row + row)
            if soft and radius > 0:
                weights.append(max(0.15, 1.0 - distance / float(radius + 1)))
            else:
                weights.append(1.0)
    return np.asarray(indices, dtype=int), np.asarray(weights, dtype=float)


def compute_grid_normals(points, n_row, n_col, normal_flip=False):
    """Estimate consistently oriented normals from a deformed taxel grid."""
    n_row = int(n_row)
    n_col = int(n_col)
    point_array = normalize_point_array(points, n_row * n_col)
    if point_array is None:
        return None

    normals = np.zeros_like(point_array)

    def point(col, row):
        return point_array[col * n_row + row]

    for col in range(n_col):
        for row in range(n_row):
            if n_col <= 1:
                tangent_col = np.array([1.0, 0.0, 0.0])
            elif col == 0:
                tangent_col = point(1, row) - point(0, row)
            elif col == n_col - 1:
                tangent_col = point(col, row) - point(col - 1, row)
            else:
                tangent_col = point(col + 1, row) - point(col - 1, row)

            if n_row <= 1:
                tangent_row = np.array([0.0, 1.0, 0.0])
            elif row == 0:
                tangent_row = point(col, 1) - point(col, 0)
            elif row == n_row - 1:
                tangent_row = point(col, row) - point(col, row - 1)
            else:
                tangent_row = point(col, row + 1) - point(col, row - 1)

            normal = np.cross(tangent_col, tangent_row)
            norm = float(np.linalg.norm(normal))
            if not np.isfinite(norm) or norm <= 1e-12:
                normal = np.array([0.0, 0.0, 1.0])
            else:
                normal /= norm
            normals[col * n_row + row] = normal

    finite_mean = np.mean(normals[np.all(np.isfinite(normals), axis=1)], axis=0)
    if finite_mean.shape == (3,) and finite_mean[2] < 0.0:
        normals *= -1.0
    if bool(normal_flip):
        normals *= -1.0
    return normals


def structured_grid_edge_pairs(n_row, n_col):
    """Return unique endpoint pairs for a column-major structured grid."""
    n_row = int(n_row)
    n_col = int(n_col)
    pairs = []
    for col in range(n_col):
        for row in range(n_row - 1):
            first = col * n_row + row
            pairs.append((first, first + 1))
    for col in range(n_col - 1):
        for row in range(n_row):
            first = col * n_row + row
            pairs.append((first, first + n_row))
    return np.asarray(pairs, dtype=int)


def structured_grid_edges(n_row, n_col):
    """Return VTK line connectivity for a column-major structured grid."""
    pairs = structured_grid_edge_pairs(n_row, n_col)
    edges = []
    for first, second in pairs:
        edges.extend([2, int(first), int(second)])
    return np.asarray(edges, dtype=np.int_)


def heatmap_surface_from_corner_lattice(
    corner_points,
    corner_n_row,
    corner_n_col,
    edge_offsets=None,
    curve_resolution=1,
):
    """Build watertight taxel patches and shared edge samples."""
    corner_n_row = int(corner_n_row)
    corner_n_col = int(corner_n_col)
    corners = normalize_point_array(
        corner_points,
        corner_n_row * corner_n_col,
    )
    if corners is None or corner_n_row < 2 or corner_n_col < 2:
        return None

    edge_pairs = structured_grid_edge_pairs(corner_n_row, corner_n_col)
    offsets = normalize_point_array(edge_offsets, len(edge_pairs))
    if offsets is None:
        offsets = np.zeros((len(edge_pairs), 3), dtype=float)
    resolution = max(1, int(curve_resolution))
    control_points = (
        0.5 * (corners[edge_pairs[:, 0]] + corners[edge_pairs[:, 1]])
        + offsets
    )
    controls = {
        tuple(sorted((int(first), int(second)))): control
        for (first, second), control in zip(edge_pairs, control_points)
    }

    def curve(first, second, values):
        first = int(first)
        second = int(second)
        values_np = np.asarray(values, dtype=float).reshape(-1, 1)
        start = corners[first]
        end = corners[second]
        control = controls[tuple(sorted((first, second)))]
        one_minus = 1.0 - values_np
        return (
            one_minus * one_minus * start
            + 2.0 * one_minus * values_np * control
            + values_np * values_np * end
        )

    parameters = np.linspace(0.0, 1.0, resolution + 1)
    edge_samples = [
        curve(first, second, parameters)
        for first, second in edge_pairs
    ]
    tile_points = []
    faces = []
    points_per_tile = (resolution + 1) ** 2
    taxel_n_row = corner_n_row - 1
    taxel_n_col = corner_n_col - 1
    tile_vertices = np.empty(
        (taxel_n_row, taxel_n_col, points_per_tile), dtype=int
    )

    for col in range(taxel_n_col):
        for row in range(taxel_n_row):
            p00 = col * corner_n_row + row
            p10 = (col + 1) * corner_n_row + row
            p11 = (col + 1) * corner_n_row + row + 1
            p01 = col * corner_n_row + row + 1
            bottom = curve(p00, p10, parameters)
            top = curve(p01, p11, parameters)
            left = curve(p00, p01, parameters)
            right = curve(p10, p11, parameters)
            first_vertex = len(tile_points)
            local_points = []
            for v_index, v_value in enumerate(parameters):
                for u_index, u_value in enumerate(parameters):
                    blended_boundaries = (
                        (1.0 - v_value) * bottom[u_index]
                        + v_value * top[u_index]
                        + (1.0 - u_value) * left[v_index]
                        + u_value * right[v_index]
                    )
                    bilinear_corners = (
                        (1.0 - u_value) * (1.0 - v_value) * corners[p00]
                        + u_value * (1.0 - v_value) * corners[p10]
                        + u_value * v_value * corners[p11]
                        + (1.0 - u_value) * v_value * corners[p01]
                    )
                    local_points.append(blended_boundaries - bilinear_corners)
            tile_points.extend(local_points)
            vertex_ids = np.arange(
                first_vertex,
                first_vertex + points_per_tile,
                dtype=int,
            )
            tile_vertices[row, col] = vertex_ids
            for v_index in range(resolution):
                for u_index in range(resolution):
                    lower_left = first_vertex + v_index * (resolution + 1) + u_index
                    lower_right = lower_left + 1
                    upper_left = lower_left + resolution + 1
                    upper_right = upper_left + 1
                    faces.extend(
                        [
                            4,
                            lower_left,
                            lower_right,
                            upper_right,
                            upper_left,
                        ]
                    )

    return {
        "points": np.asarray(tile_points, dtype=float),
        "faces": np.asarray(faces, dtype=np.int_),
        "tile_vertices": tile_vertices,
        "edge_pairs": edge_pairs,
        "edge_samples": edge_samples,
        "control_points": control_points,
    }


def heatmap_corner_points_from_centres(
    points,
    normals,
    n_row,
    n_col,
    surface_lift_ratio=0.002,
):
    """Create a connected heatmap corner lattice from taxel centres."""
    n_row = int(n_row)
    n_col = int(n_col)
    centres = normalize_point_array(points, n_row * n_col)
    centre_normals = normalize_point_array(normals, n_row * n_col)
    if centres is None or centre_normals is None:
        return None

    point_grid = np.empty((n_row, n_col, 3), dtype=float)
    normal_grid = np.empty_like(point_grid)
    for col in range(n_col):
        for row in range(n_row):
            index = col * n_row + row
            point_grid[row, col] = centres[index]
            normal_grid[row, col] = centre_normals[index]

    sensor_span = float(
        np.linalg.norm(np.max(centres, axis=0) - np.min(centres, axis=0))
    )
    fallback_half_size = max(sensor_span * 0.025, 0.001)
    surface_lift = max(sensor_span * float(surface_lift_ratio), 0.0001)
    corner_sum = np.zeros((n_row + 1, n_col + 1, 3), dtype=float)
    corner_count = np.zeros((n_row + 1, n_col + 1), dtype=float)

    def normalized(vector):
        vector_np = np.asarray(vector, dtype=float)
        magnitude = float(np.linalg.norm(vector_np))
        if not np.isfinite(magnitude) or magnitude <= 1e-12:
            return np.array([0.0, 0.0, 1.0], dtype=float)
        return vector_np / magnitude

    def axis_half_vector(row, col, axis):
        if axis == 0 and n_row > 1:
            if 0 < row < n_row - 1:
                return (point_grid[row + 1, col] - point_grid[row - 1, col]) * 0.25
            neighbour = row + 1 if row == 0 else row - 1
            direction = 1.0 if row == 0 else -1.0
            return (point_grid[neighbour, col] - point_grid[row, col]) * 0.5 * direction
        if axis == 1 and n_col > 1:
            if 0 < col < n_col - 1:
                return (point_grid[row, col + 1] - point_grid[row, col - 1]) * 0.25
            neighbour = col + 1 if col == 0 else col - 1
            direction = 1.0 if col == 0 else -1.0
            return (point_grid[row, neighbour] - point_grid[row, col]) * 0.5 * direction
        return np.zeros(3, dtype=float)

    def tangent(vector, normal):
        tangent_vector = vector - normal * float(np.dot(vector, normal))
        if np.all(np.isfinite(tangent_vector)):
            return tangent_vector
        return np.zeros(3, dtype=float)

    for col in range(n_col):
        for row in range(n_row):
            centre = point_grid[row, col]
            normal = normalized(normal_grid[row, col])
            row_half = tangent(axis_half_vector(row, col, axis=0), normal)
            col_half = tangent(axis_half_vector(row, col, axis=1), normal)
            row_norm = float(np.linalg.norm(row_half))
            col_norm = float(np.linalg.norm(col_half))
            if row_norm <= 1e-9 and col_norm > 1e-9:
                row_half = normalized(np.cross(normal, col_half)) * max(
                    col_norm, fallback_half_size
                )
            elif col_norm <= 1e-9 and row_norm > 1e-9:
                col_half = normalized(np.cross(row_half, normal)) * max(
                    row_norm, fallback_half_size
                )
            elif row_norm <= 1e-9 and col_norm <= 1e-9:
                reference = np.array([0.0, 0.0, 1.0], dtype=float)
                if abs(float(np.dot(normal, reference))) > 0.9:
                    reference = np.array([0.0, 1.0, 0.0], dtype=float)
                col_half = normalized(np.cross(reference, normal)) * fallback_half_size
                row_half = normalized(np.cross(normal, col_half)) * fallback_half_size

            lifted_centre = centre + normal * surface_lift
            corners = (
                lifted_centre - row_half - col_half,
                lifted_centre - row_half + col_half,
                lifted_centre + row_half + col_half,
                lifted_centre + row_half - col_half,
            )
            indices = (
                (row, col),
                (row, col + 1),
                (row + 1, col + 1),
                (row + 1, col),
            )
            for corner, (corner_row, corner_col) in zip(corners, indices):
                corner_sum[corner_row, corner_col] += corner
                corner_count[corner_row, corner_col] += 1.0

    corner_grid = corner_sum / corner_count[:, :, None]
    return np.asarray(
        [
            corner_grid[row, col]
            for col in range(n_col + 1)
            for row in range(n_row + 1)
        ],
        dtype=float,
    )


def interpolate_coarse_deformation(
    base_coarse_points,
    deformed_coarse_points,
    fine_points,
    neighbor_count=4,
):
    """Blend coarse point displacement onto a dense point cloud."""
    base = normalize_point_array(base_coarse_points)
    deformed = normalize_point_array(deformed_coarse_points)
    fine = normalize_point_array(fine_points)
    if (
        base is None
        or deformed is None
        or fine is None
        or base.shape != deformed.shape
        or base.shape[0] == 0
    ):
        return fine

    displacement = deformed - base
    distances_sq = np.sum((fine[:, None, :] - base[None, :, :]) ** 2, axis=2)
    neighbor_count = max(1, min(int(neighbor_count), base.shape[0]))
    nearest = np.argpartition(
        distances_sq, kth=neighbor_count - 1, axis=1
    )[:, :neighbor_count]
    nearest_distances = np.take_along_axis(distances_sq, nearest, axis=1)
    weights = 1.0 / np.maximum(nearest_distances, 1e-12)
    weights /= np.sum(weights, axis=1, keepdims=True)
    blended = np.sum(displacement[nearest] * weights[:, :, None], axis=1)
    return fine + blended


def smooth_grid_points(points, selected_indices, n_row, n_col, strength=0.5):
    """Laplacian-smooth selected taxels using their four grid neighbors."""
    n_row = int(n_row)
    n_col = int(n_col)
    point_array = normalize_point_array(points, n_row * n_col)
    if point_array is None:
        return None
    result = np.array(point_array, copy=True)
    strength = float(np.clip(float(strength), 0.0, 1.0))

    for raw_index in np.asarray(selected_indices, dtype=int).reshape(-1):
        index = int(raw_index)
        if not 0 <= index < n_row * n_col:
            continue
        col, row = divmod(index, n_row)
        neighbors = []
        if col > 0:
            neighbors.append(point_array[(col - 1) * n_row + row])
        if col + 1 < n_col:
            neighbors.append(point_array[(col + 1) * n_row + row])
        if row > 0:
            neighbors.append(point_array[col * n_row + row - 1])
        if row + 1 < n_row:
            neighbors.append(point_array[col * n_row + row + 1])
        if neighbors:
            neighbor_mean = np.mean(neighbors, axis=0)
            result[index] = (
                (1.0 - strength) * point_array[index]
                + strength * neighbor_mean
            )
    return result
