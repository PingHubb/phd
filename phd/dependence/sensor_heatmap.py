import numpy as np


DEFAULT_HEATMAP_SATURATION_PCT = 5.0
DEFAULT_HEATMAP_NOISE_FLOOR_PCT = 0.5
HEATMAP_RESPONSE_LINEAR_RELATIVE = "linear_relative"
HEATMAP_RESPONSE_PROXIMITY_ENHANCED = "proximity_enhanced"
DEFAULT_HEATMAP_RESPONSE_MODE = HEATMAP_RESPONSE_LINEAR_RELATIVE
DEFAULT_PROXIMITY_NOISE_FLOOR = 20.0
DEFAULT_PROXIMITY_KNEE = 100.0
DEFAULT_PROXIMITY_SATURATION = 1000.0
PROXIMITY_KNEE_COLOR_STRENGTH = 0.55
DEFAULT_HEATMAP_3D_COLOR_GAIN = 1.5
HEATMAP_3D_PALETTE_WHITE_RED = "white_red"
HEATMAP_3D_PALETTE_WHITE_BLUE_RED = "white_blue_red"
HEATMAP_3D_PALETTE_LIGHT_DEEP_BLUE = "light_deep_blue"
DEFAULT_HEATMAP_3D_PALETTE = HEATMAP_3D_PALETTE_WHITE_RED
WHITE_BLUE_RED_STOPS = np.asarray(
    [0.0, 0.08, 0.25, 0.45, 0.65, 0.82, 1.0], dtype=float
)
WHITE_BLUE_RED_COLORS = np.asarray(
    [
        [255, 255, 255],
        [225, 235, 255],
        [120, 160, 235],
        [35, 80, 205],
        [92, 45, 170],
        [190, 25, 70],
        [130, 0, 25],
    ],
    dtype=float,
)
LIGHT_DEEP_BLUE_STOPS = np.asarray(
    [0.0, 0.08, 0.25, 0.45, 0.65, 0.82, 1.0], dtype=float
)
LIGHT_DEEP_BLUE_COLORS = np.asarray(
    [
        [245, 250, 255],
        [222, 239, 252],
        [172, 215, 244],
        [104, 174, 225],
        [50, 115, 190],
        [20, 70, 145],
        [5, 25, 80],
    ],
    dtype=float,
)


def normalize_heatmap_response_mode(mode):
    mode = str(mode or DEFAULT_HEATMAP_RESPONSE_MODE)
    if mode == HEATMAP_RESPONSE_PROXIMITY_ENHANCED:
        return mode
    return HEATMAP_RESPONSE_LINEAR_RELATIVE


def normalize_heatmap_3d_palette(palette):
    palette = str(palette or DEFAULT_HEATMAP_3D_PALETTE)
    if palette in {
        HEATMAP_3D_PALETTE_WHITE_BLUE_RED,
        HEATMAP_3D_PALETTE_LIGHT_DEEP_BLUE,
    }:
        return palette
    return HEATMAP_3D_PALETTE_WHITE_RED


def heatmap_strength(
    values,
    response_mode=DEFAULT_HEATMAP_RESPONSE_MODE,
    saturation_pct=DEFAULT_HEATMAP_SATURATION_PCT,
    noise_floor_pct=DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    proximity_noise_floor=DEFAULT_PROXIMITY_NOISE_FLOOR,
    proximity_knee=DEFAULT_PROXIMITY_KNEE,
    proximity_saturation=DEFAULT_PROXIMITY_SATURATION,
):
    values_np = np.abs(
        np.nan_to_num(np.asarray(values, dtype=float), nan=0.0, posinf=0.0, neginf=0.0)
    )
    mode = normalize_heatmap_response_mode(response_mode)
    if mode == HEATMAP_RESPONSE_PROXIMITY_ENHANCED:
        noise_floor = max(0.0, float(proximity_noise_floor))
        knee = max(noise_floor + 1e-6, float(proximity_knee))
        saturation = max(knee + 1e-6, float(proximity_saturation))

        proximity_progress = np.clip(
            (values_np - noise_floor) / (knee - noise_floor), 0.0, 1.0
        )
        proximity_strength = PROXIMITY_KNEE_COLOR_STRENGTH * np.sqrt(
            proximity_progress
        )

        contact_progress = np.clip(
            (values_np - knee) / (saturation - knee), 0.0, 1.0
        )
        compressed_contact = np.log1p(9.0 * contact_progress) / np.log(10.0)
        contact_strength = PROXIMITY_KNEE_COLOR_STRENGTH + (
            1.0 - PROXIMITY_KNEE_COLOR_STRENGTH
        ) * compressed_contact
        strength = np.where(values_np <= knee, proximity_strength, contact_strength)
        return np.where(values_np <= noise_floor, 0.0, np.clip(strength, 0.0, 1.0))

    saturation = max(1e-6, float(saturation_pct))
    noise_floor = max(0.0, float(noise_floor_pct))
    usable_span = max(1e-6, saturation - noise_floor)
    strength = np.clip((values_np - noise_floor) / usable_span, 0.0, 1.0)
    return np.where(values_np <= noise_floor, 0.0, strength)


def heatmap_rgb(
    values,
    response_mode=DEFAULT_HEATMAP_RESPONSE_MODE,
    saturation_pct=DEFAULT_HEATMAP_SATURATION_PCT,
    noise_floor_pct=DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    proximity_noise_floor=DEFAULT_PROXIMITY_NOISE_FLOOR,
    proximity_knee=DEFAULT_PROXIMITY_KNEE,
    proximity_saturation=DEFAULT_PROXIMITY_SATURATION,
):
    """Map relative sensor changes to the shared white-to-red heatmap."""
    strength = heatmap_strength(
        values,
        response_mode=response_mode,
        saturation_pct=saturation_pct,
        noise_floor_pct=noise_floor_pct,
        proximity_noise_floor=proximity_noise_floor,
        proximity_knee=proximity_knee,
        proximity_saturation=proximity_saturation,
    )

    green_blue = np.rint(255.0 * (1.0 - strength)).astype(np.uint8)
    rgb = np.empty(np.asarray(strength).shape + (3,), dtype=np.uint8)
    rgb[..., 0] = 255
    rgb[..., 1] = green_blue
    rgb[..., 2] = green_blue
    return rgb


def apply_3d_heatmap_color_gain(rgb, gain=DEFAULT_HEATMAP_3D_COLOR_GAIN):
    """Increase intermediate red contrast for small 3D tiles.

    White, full red, and the underlying signal thresholds stay unchanged. A
    gain of 1.0 is the exact 2D colour; values above 1.0 compensate for the
    weaker appearance of pale colours on a curved 3D field.
    """
    colors = np.asarray(rgb, dtype=np.uint8)
    gain_value = max(1.0, float(gain))
    if gain_value <= 1.0:
        return np.array(colors, copy=True)

    output = np.array(colors, copy=True)
    remaining_white = colors[..., 1:3].astype(float) / 255.0
    output[..., 1:3] = np.rint(
        255.0 * np.power(remaining_white, gain_value)
    ).astype(np.uint8)
    return output


def signed_heatmap_rgb(
    values,
    response_mode=DEFAULT_HEATMAP_RESPONSE_MODE,
    saturation_pct=DEFAULT_HEATMAP_SATURATION_PCT,
    noise_floor_pct=DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    proximity_noise_floor=DEFAULT_PROXIMITY_NOISE_FLOOR,
    proximity_knee=DEFAULT_PROXIMITY_KNEE,
    proximity_saturation=DEFAULT_PROXIMITY_SATURATION,
    color_gain=DEFAULT_HEATMAP_3D_COLOR_GAIN,
):
    """Map signed response to blue-negative, white-zero, red-positive RGB."""
    signed_values = np.nan_to_num(
        np.asarray(values, dtype=float),
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    strength = heatmap_strength(
        signed_values,
        response_mode=response_mode,
        saturation_pct=saturation_pct,
        noise_floor_pct=noise_floor_pct,
        proximity_noise_floor=proximity_noise_floor,
        proximity_knee=proximity_knee,
        proximity_saturation=proximity_saturation,
    )
    gain_value = max(1.0, float(color_gain))
    if gain_value > 1.0:
        strength = 1.0 - np.power(1.0 - strength, gain_value)

    remaining_white = np.rint(255.0 * (1.0 - strength)).astype(np.uint8)
    rgb = np.full(signed_values.shape + (3,), 255, dtype=np.uint8)
    positive = signed_values > 0.0
    negative = signed_values < 0.0
    rgb[..., 1][positive] = remaining_white[positive]
    rgb[..., 2][positive] = remaining_white[positive]
    rgb[..., 0][negative] = remaining_white[negative]
    rgb[..., 1][negative] = remaining_white[negative]
    return rgb


def heatmap_3d_rgb(
    values,
    palette=DEFAULT_HEATMAP_3D_PALETTE,
    response_mode=DEFAULT_HEATMAP_RESPONSE_MODE,
    saturation_pct=DEFAULT_HEATMAP_SATURATION_PCT,
    noise_floor_pct=DEFAULT_HEATMAP_NOISE_FLOOR_PCT,
    proximity_noise_floor=DEFAULT_PROXIMITY_NOISE_FLOOR,
    proximity_knee=DEFAULT_PROXIMITY_KNEE,
    proximity_saturation=DEFAULT_PROXIMITY_SATURATION,
    color_gain=DEFAULT_HEATMAP_3D_COLOR_GAIN,
    use_absolute_signal=True,
):
    """Map sensor response using the selected 3D-only colour palette."""
    if not bool(use_absolute_signal):
        return signed_heatmap_rgb(
            values,
            response_mode=response_mode,
            saturation_pct=saturation_pct,
            noise_floor_pct=noise_floor_pct,
            proximity_noise_floor=proximity_noise_floor,
            proximity_knee=proximity_knee,
            proximity_saturation=proximity_saturation,
            color_gain=color_gain,
        )

    palette = normalize_heatmap_3d_palette(palette)
    if palette == HEATMAP_3D_PALETTE_WHITE_RED:
        rgb = heatmap_rgb(
            values,
            response_mode=response_mode,
            saturation_pct=saturation_pct,
            noise_floor_pct=noise_floor_pct,
            proximity_noise_floor=proximity_noise_floor,
            proximity_knee=proximity_knee,
            proximity_saturation=proximity_saturation,
        )
        return apply_3d_heatmap_color_gain(rgb, gain=color_gain)

    strength = heatmap_strength(
        values,
        response_mode=response_mode,
        saturation_pct=saturation_pct,
        noise_floor_pct=noise_floor_pct,
        proximity_noise_floor=proximity_noise_floor,
        proximity_knee=proximity_knee,
        proximity_saturation=proximity_saturation,
    )
    gain_value = max(1.0, float(color_gain))
    if gain_value > 1.0:
        strength = 1.0 - np.power(1.0 - strength, gain_value)

    if palette == HEATMAP_3D_PALETTE_LIGHT_DEEP_BLUE:
        palette_stops = LIGHT_DEEP_BLUE_STOPS
        palette_colors = LIGHT_DEEP_BLUE_COLORS
    else:
        palette_stops = WHITE_BLUE_RED_STOPS
        palette_colors = WHITE_BLUE_RED_COLORS

    flat_strength = np.asarray(strength, dtype=float).reshape(-1)
    flat_rgb = np.column_stack(
        [
            np.interp(
                flat_strength,
                palette_stops,
                palette_colors[:, channel],
            )
            for channel in range(3)
        ]
    )
    return np.rint(flat_rgb).astype(np.uint8).reshape(
        np.asarray(strength).shape + (3,)
    )
