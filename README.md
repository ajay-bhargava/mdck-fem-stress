# MDCK Cell Mechanics

A scroll-driven explanation of how forces beneath a sheet of **MDCK cells** become a finite element model (FEM) of tissue stress.

Watch one field of view across **97 frames and 24 hours**. Reveal nuclei, nuclear segmentation, a triangular mesh, applied nodal forces, force cancellation, and predicted stress.

> [!NOTE]
> 🔬 Images and traction provide experimental information. Tissue stress is a **model prediction**, not a direct measurement.

## How the model works

1. **Observe the cells.** Fluorescent nuclei show how the cell sheet changes over time.
2. **Use measured traction.** Traction-force microscopy estimates force per unit area from deformation of the underlying gel. This project starts with existing traction estimates.
3. **Build a mesh.** Triangles cover the reference tissue outline. Their corners are calculation nodes, not individual cells.
4. **Assemble loads.** Integrate traction over the triangles to obtain applied nodal forces in newtons.
5. **Solve equilibrium.** With explicit material and boundary assumptions, FEM calculates an auxiliary displacement field and recovers strain and stress.
6. **Show stress over time.** Repeat with each frame's loads on the same reference mesh. Bright regions indicate larger predicted stress magnitude.

> [!IMPORTANT]
> 🧩 Applied nodal forces are **inputs** to the solve. Stress is an **output**. A strong applied force does not automatically mean high internal stress at that location.

## Explore thickness

Choose **2.5, 5, or 10 µm** in the stress view. For the same loads in this model:

| Assumed tissue thickness | Stress relative to 5 µm |
| --- | --- |
| 2.5 µm | Twice as large |
| 5 µm | Reference |
| 10 µm | Half as large |

All options share a fixed color scale across time and thickness. Values above its limit saturate. The displayed quantity is the **largest absolute principal stress**, in Pa.

> [!TIP]
> 📏 This is an assumption-sensitivity exercise. A thinner sheet carries the same membrane force with higher stress. It does not predict how living cells would respond to becoming thinner.

## Caveats

> [!WARNING]
> ⚠️ These reconstructions are educational, not validated measurements of tissue rheology.
>
> - Nuclear outlines are **not cell boundaries**.
> - The mesh stays fixed while the cells move.
> - Traction units are assumed to be Pa; the source units need confirmation.
> - Traction action/reaction direction remains unresolved. The magnitude view cannot distinguish tension from compression.
> - Measured loads do not balance exactly. The solve uses an explicit correction, stored separately from the raw loads.
> - Material and boundary assumptions affect the result. Small numerical residuals do not establish biological accuracy.
> - Auxiliary FEM displacement and strain are **not measured cell motion**. Some auxiliary strains are too large to interpret as small physical deformation.
> - No viscosity, relaxation time, or active stress has been identified.

<details>
<summary>🔧 What assumptions does the solve use?</summary>

The current model is a homogeneous, isotropic, small-strain, plane-stress sheet with free boundaries, Poisson ratio 0.3, and an auxiliary Young's modulus of 1 kPa. This modulus is not an inferred tissue property.

Rigid-motion gauges remove numerical translation and rotation modes. A mesh-dependent nodal-load projection removes incompatible net force and moment for each connected component. These gauges are not physical clamps, and the correction is not a measured force.

Stress in Pa multiplied by uniform thickness in metres gives membrane stress resultant in N/m. Changing thickness uses exact linear-model scaling, not duplicated solves. The membrane resultant stays unchanged.

Synthetic tests check uniform stress recovery, rigid rotation, disconnected components, load correction, modulus scaling, and thickness scaling. Experimental frames also pass independent nodal equilibrium checks. Broader convergence, uncertainty, and assumption-sensitivity studies remain necessary.

</details>

<details>
<summary>🌐 Why pre-render the educational website?</summary>

The goal is a lightweight Cloudflare-compatible site that serves prepared images and small metadata files. Scientific computation belongs in an offline build, not in a visitor's browser or a request handler.

Stress images are already pre-rendered for all frames and thicknesses. Conversion of the rest of the experience to static assets is ongoing. The current local viewer still uses a Python backend.

Keep image dimensions and color scales consistent, preload nearby frames, and publish only the assets the lesson needs. Raw microscopy and full solver arrays do not belong in the website bundle.

</details>

<details>
<summary>💻 Local development</summary>

Requires Python 3.12 or newer and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
./scripts/run_web.sh
```

The local viewer requires separately prepared data. This repository intentionally excludes experimental inputs, solver results, generated media, and credentials. A fresh clone is not a standalone dataset or a finished static deployment.

Run the synthetic solver tests without experimental data:

```sh
uv run python -m unittest discover -s tests -p test_equilibrium.py -v
```

Code lives in `src/`, build helpers in `scripts/`, and tests in `tests/`. Preserve source inputs unchanged when preparing data locally. Verify data permissions and required attribution before publishing media.

</details>
