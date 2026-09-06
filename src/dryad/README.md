# Dryad staged reconstruction tools

This package converts the immutable `published` source tree into standardized, aligned, vector-geometry, mesh, and conservative load artifacts under `data/`. It does **not** modify files inside the source tree.

## Build

```bash
uv run python src/dryad/build_dataset.py \
  --source published \
  --output data
```

Existing large position outputs are validated and left unchanged; lightweight source-path indexes are refreshed. Use `--overwrite` to rebuild all position artifacts, or select positions with `--positions pos01 pos02`.

Validate existing outputs without writing:

```bash
uv run python src/dryad/build_dataset.py \
  --source published \
  --output data \
  --validate-only
```

## Source-to-output mapping

| Source | Standardized output |
|---|---|
| `ExperimentalSettings.txt` and root `README.txt` | `metadata.yaml` |
| all source filenames | `source_index.yaml` |
| `DIC_c1_64w0_16d0.mat` | `displacement/published_dic.npz` |
| `tractions_DIC_c1_64w0.mat` | `traction/published.npz` |
| `Centroids_to_Voronoi.mat` | `geometry/centroids.npz` |
| `KNN_results_JN_refine.mat` | `geometry/trajectories.npz` |
| `domain.tif` | `geometry/domain.tif` |
| `Label_Image.tif` | `geometry/labels.tif` |
| `c1.tif`, `c1_ref.tif`, `c2.tif`, `nuc.tif` | paths only in `images/source_paths.yaml` |

The dataset README explicitly identifies `c1` as loaded beads, `c1_ref` as reference beads, `c2` as cell body, and `nuc` as nuclei. `source_paths.yaml` records this as `verified_from_dataset_readme` and also preserves the raw names.

## Arrays and units

For `pos01`, published DIC fields `u` and `v` have shape `(126, 127, 97)`, with `x`, `y`, and `c_peak` on `(126, 127)`. Published traction fields `u`, `v`, `tx`, and `ty` have shape `(118, 119, 97)`, with `x` and `y` on `(118, 119)`. Shapes are validated per position rather than forced to these dimensions.

MATLAB numeric arrays retain their original shape, dtype, values, and source units. Variable-length MATLAB cells in the centroid file are encoded without pickle as `<name>_values`, `<name>_offsets`, `<name>_shapes`, `<name>_ndims`, and `<name>_cell_shape`. Cell `i` is reconstructed from `values[offsets[i]:offsets[i+1]]` and its recorded shape.

`metadata.yaml` preserves raw settings entries and records normalized SI values. The settings establish a 0.65 µm pixel size, 6 kPa substrate modulus, 0.48 substrate Poisson ratio, 75 µm substrate thickness, 0.3 monolayer Poisson ratio, and 5 µm monolayer thickness. The README establishes 97 frames at 15-minute intervals. Units of the MATLAB coordinate, displacement, and traction arrays are not documented and therefore remain `source_preserved`/`unknown` rather than being inferred.

Image coordinates use a top-left origin. A later FEM representation is expected to use a bottom-left origin, so a y-axis transform is required; this preprocessing stage does not apply one.

## Later stages

The commands above describe Stage 0 only. Subsequent modules build alignment diagnostics, vector geometry, constrained meshes, and Stage 5 conservative loads. Build or validate Stage 5 with:

```bash
uv run python src/dryad/build_loads.py --data data
uv run python src/dryad/build_loads.py --data data --validate-only
```

Stage 5 infers Pa from the SI reconstruction context but records that this is not explicit in the source README. It also preserves both possible action/reaction signs in metadata because “cell-substrate traction” is semantically unresolved. Optical flow, traction inversion, FEM solving, stiffness assembly, constitutive assumptions, boundary constraints, automatic equilibrium correction, and material inference remain unimplemented.
