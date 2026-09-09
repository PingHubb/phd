# Sensor Data Layout and Compatibility

Last reviewed: 2026-08-27

This document records how a tactile sensor frame moves through PingLab, how
rows and columns are interpreted by each major feature, and why several
legacy conversions are intentionally being left unchanged for now.

## Current decision

Keep the existing processing and trained-model layouts unchanged.

- The Signal Viewer and all main 3D visualization modes now use the same
  top-down row/column layout.
- Rule-based DFM and the other averaged-signal controls remain compatible.
- Existing AI DFM and AI proximity models continue to receive the same input
  layout used during their training.
- Do not remove the central `flipud()` or silently change AI channel layouts.
- Three known legacy inconsistencies are documented below. They should be
  addressed only through local fixes and a versioned AI migration.

## Terms

`row` and `column` always refer to a logical sensor cell.

Top-down means that row 0 is shown at the top of a table or image and row
numbers increase downward:

```text
row 0  top
row 1
row 2
...
row R-1  bottom
```

Column-major means that a flat list contains every row of column 0 first,
then every row of column 1, and so on:

```text
flat_index = column * number_of_rows + row
```

The internal matrix shape is normally `(number_of_rows, number_of_columns)`
and is accessed as `matrix[row, column]`.

## End-to-end frame flow

```text
Serial/USB response
    -> numeric API payload list
    -> optional removal of one extra packet column
    -> column-major reshape into source matrix
    -> calibration and instantaneous differences
    -> rolling average
    -> vertical flip into top-down averaged matrix
    -> visualization, rule-based control, or legacy AI conversion
```

### 1. API payload

The serial API removes the response framing values and returns the sensor
payload as a Python list. The direct Goodix backend also returns the historic
column-major taxel order.

If `Raw packet includes +1 column` is enabled, the last `number_of_rows`
values are discarded before the frame is reshaped. This supports controllers
that transmit one extra non-sensor column.

### 2. Source matrix

The payload is converted with:

```python
source = np.asarray(payload).reshape(number_of_columns, number_of_rows).T
```

For a 2x3 sensor:

```text
payload = [1, 2, 3, 4, 5, 6]

source matrix =
[[1, 3, 5],
 [2, 4, 6]]
```

These matrices retain source row orientation:

| Name | Meaning |
|---|---|
| `calData` | Calibration baseline |
| `rawData` | Latest instantaneous raw frame |
| `diffData` | `rawData - calData` |
| `diffPerData` | `100 * diffData / calData` |

### 3. Top-down averaged matrix

The rolling-window results are vertically flipped:

```python
diffPerDataAve = np.flipud(np.mean(diffPerDataWin, axis=0))
```

The 2x3 example therefore becomes:

```text
top-down averaged matrix =
[[2, 4, 6],
 [1, 3, 5]]
```

These matrices use the top-down layout:

| Name | Meaning |
|---|---|
| `rawDataAve` | Moving-average raw frame |
| `diffDataAve` | Moving-average absolute difference |
| `diffPerDataAve` | Moving-average percentage difference |

## Exact display mapping

For a display cell `(display_row, column)`, the corresponding API payload
index is:

```text
API index = column * number_of_rows
            + (number_of_rows - 1 - display_row)
```

For a 10x10 sensor:

| Display cell | Point label | API index |
|---|---:|---:|
| `(0, 0)` | P0 | 9 |
| `(1, 0)` | P1 | 8 |
| `(9, 0)` | P9 | 0 |
| `(0, 1)` | P10 | 19 |
| `(9, 9)` | P99 | 90 |

Point labels are logical display identifiers:

```text
point_id = column * number_of_rows + display_row
```

For 10x10, the displayed labels are:

```text
top row:    P0, P10, P20, ..., P90
second row: P1, P11, P21, ..., P91
bottom row: P9, P19, P29, ..., P99
```

The point label and the API index are therefore not generally the same after
the top-down conversion.

## Layout used by each major feature

| Feature | Data/layout used |
|---|---|
| Signal Viewer raw values | `flipud(rawData)`, top-down |
| Signal Viewer calibration | `flipud(calData)`, top-down |
| Signal Viewer differences | Computed from top-down raw/calibration lists |
| Point Grid | `diffPerDataAve[row, column]`, top-down |
| Stereo Field | `diffPerDataAve[row, column]`, top-down |
| 3D Heatmap, linear mode | `diffPerDataAve[row, column]`, top-down |
| 3D Heatmap, proximity mode | `flipud(diffData)`, top-down |
| Green selected-cell marker | Direct viewer `(row, column)` |
| Visual zero mask and labels | Direct viewer `(row, column)` |
| Contact-force display | Top-down averaged matrix |
| Contact-normal vector | Top-down averaged matrix |
| Robot-model sensor colours | Top-down averaged matrix |
| Rule-based DFM | Column-major flatten of `diffPerDataAve` |
| Console sensor control | Column-major flatten of `diffPerDataAve` |
| Three-level recognition | Column-major flatten of `diffPerDataAve` |
| Rule-based proximity control | Absolute value of `diffPerDataAve` |
| AI DFM record/execute | Historic transposed AI matrices |
| AI proximity | Historic transposed AI matrices |
| Raw-data log | Source matrix flattened back to API order |
| Averaged-data log | Top-down averaged matrix in column-major order |

For DFM and other top-down motion tracking:

```text
delta_row < 0  movement toward the viewer top
delta_row > 0  movement toward the viewer bottom
delta_col < 0  movement toward the viewer left
delta_col > 0  movement toward the viewer right
```

## Geometry reordering

Sensor reorder settings change geometry arrays, not the sensor data matrix.
They remap:

- XYZ points
- Surface normals
- Mesh-to-taxel assignments
- Grid edges

The signal is still read as `matrix[row, column]`, but the physical XYZ point
assigned to that logical cell may change. Cylinder bending, custom shape
editing, scaling, and XYZ rotation also change physical positions without
renaming the logical cell.

At the time of this review, the saved `2d_10x10` configuration uses
`row_to_col`, which transposes the original geometry indexing. The saved 7x7,
10x16, 10x8, and 8x10 2D configurations use their factory reorder setting.
These values are user settings and may change later.

## Legacy AI layout

AI DFM recording and execution use the historic matrix view:

```python
ai_matrix = sensor_matrix.T
```

The model therefore receives `(number_of_columns, number_of_rows)`. A 10x10
sensor still appears to be 10x10, so this transpose is not visible from tensor
shape alone.

AI DFM normally builds channels from:

```text
diffPerData
diffPerDataAve
frameDiff
touchMask
```

AI proximity normally builds channels from:

```text
diffPerData
diffPerDataAve
frameDiff
```

The checkpoint controls the exact AI proximity channel list. Training and
live execution preserve the same historic transpose, so existing checkpoints
remain compatible.

## Known legacy inconsistencies

### 1. Zero-mask preprocessing

The mask UI and visual mask use top-down rows. The current preprocessing code
applies that mask directly to source-oriented `rawData` before averaging.

For four rows:

```text
viewer row: 0  1  2  3
source row: 3  2  1  0
```

Masking viewer row 0 can therefore force source row 0, which later appears at
viewer row 3, to calibration. The visually hidden cell and the cell forced to
zero can differ.

Future local fix:

```python
source_mask = np.flipud(viewer_mask)
raw_matrix[source_mask] = calibration[source_mask]
```

Impact: no change when no cells are masked. With a mask, downstream rule-based
and AI functions would receive zero at the intended cell instead of the
opposite row.

### 2. Mixed instantaneous and averaged AI channels

`diffPerData` is source-oriented, while `diffPerDataAve` is top-down. The same
physical touch can therefore appear at opposite rows in those two channels
before both are transposed for AI.

Training and execution currently reproduce the same layout, so existing AI
models can operate. Correcting the channel alignment would nevertheless
change their learned input representation.

Future versioned fix:

```python
instantaneous_top_down = np.flipud(diffPerData)
averaged_top_down = diffPerDataAve
```

Use a checkpoint metadata field such as `sensor_layout_version`:

```text
legacy_v1  current trained-model behavior
top_down_v2  aligned channels for new recordings and models
```

Do not silently apply this conversion to an existing checkpoint. Record new
data and retrain, or explicitly retain a legacy adapter.

### 3. Pressure-admittance fallback

Pressure admittance checks instantaneous contact with source-oriented
`diffPerData`, then normally estimates location using top-down
`diffPerDataAve`. If the averaged estimate is unavailable, it falls back to
the source-oriented location.

That fallback can vertically reverse the selected contact point, surface
normal, and robot motion direction.

Future local fix:

```python
instantaneous_top_down = np.flipud(diffPerData)
averaged_top_down = diffPerDataAve
```

Both estimates should then use the same top-down coordinates. This change is
local to pressure admittance and does not require AI retraining.

## Compatibility impact of future changes

| Proposed change | Existing rule-based DFM | Existing AI models | Admittance |
|---|---|---|---|
| Correct zero-mask source conversion | Only changes masked cells | Only changes masked cells | Only changes masked cells |
| Align instantaneous pressure matrix | No effect | No effect | Corrects location/fallback |
| Align AI instantaneous channels | No effect | Requires legacy mode or retraining | No effect |
| Remove central averaged `flipud()` | Broad breaking change | Breaks input layout | Broad behavior change |

## Safe future migration plan

1. Keep the current averaged top-down convention.
2. Add explicit source-to-top-down helper functions rather than adding inline
   flips in consumers.
3. Correct zero-mask application locally and test top, bottom, left, and right
   cells.
4. Correct pressure admittance locally and verify robot motion in dry run.
5. Add `sensor_layout_version` to AI recordings and checkpoints.
6. Preserve `legacy_v1` inference for current models.
7. Record and train new models with aligned `top_down_v2` channels.
8. Compare legacy and v2 models before making v2 the default.

## Quick validation procedure

Use one isolated touch at each corner and verify that all views identify the
same logical cell:

```text
top-left
top-right
bottom-left
bottom-right
```

For every touch, compare:

- Signal Viewer value and colour
- Green selected point
- Point Grid displacement
- Stereo Field response
- 3D Heatmap colour
- Point label
- Rule-based DFM centroid
- AI localization output, when applicable
- Admittance contact centre in dry-run mode

The camera or sensor XYZ rotation can change how the surface appears on the
screen, but it must not change the logical `(row, column)` assigned to a
taxel.
