# First forward-solve implementation (experimental)

`equilibrium.py` implements a homogeneous, isotropic, small-strain P1 plane-stress
sheet. This is an **auxiliary elastic closure**, not a rheological inference.
Equilibrium does not uniquely determine a 2D symmetric stress tensor from traction.
Boundary, closure, geometry, thickness, sign and correction assumptions matter.

## Validation

```sh
uv run python -m unittest discover -s tests -p test_equilibrium.py -v
```

Tests cover analytically integrated uniform boundary tractions (including shear),
patch consistency across three mesh densities, rigid rotation, disconnected bodies,
explicit rejection/correction of incompatible loads, and modulus scaling.
These are initial verification tests, **not** a completed nonuniform convergence,
experimental validation, uncertainty, or boundary/closure sensitivity study.

## Experimental entry point

```sh
uv run python -m dryad.build_stress --help
```

This requires an explicit position/frame, auxiliary E, nu, thickness, assumed
traction sign, acknowledgement of inferred Pa, and a new output directory.
It does not select these physical inputs for the user. It does not process all
frames automatically or add stress to the public viewer.

Boundary policy is a free sheet. Three rigid-motion gauges per connected component
remove translational/rotational null modes, not supply physical supports. Default
`--equilibrium-policy reject` refuses loads incompatible with this boundary model.
Optional `project` explicitly subtracts the Euclidean projection onto each
component's rigid modes. This preserves raw loads in output and never modifies
Stage 5 artifacts. This projection is mesh-dependent, not a uniquely justified
experimental correction; a nonzero correction must not be mistaken for measured
boundary traction or an active stress. No clamped or mixed physical supports have
been implemented.

Outputs: `stress.npz`, `provenance.json`. Stress is elementwise, symmetric, in Pa;
membrane stress resultant is h times stress in N/m. Arrays retain coordinates,
connectivity, area, frame, position, relative physical time, component identity,
validity, raw/applied loads, correction, residual, auxiliary displacement and
strain. Provenance stores source hashes, source mapping metadata and assumptions.
Constraint multipliers are numerical gauge quantities, not measured support reactions.

## Remaining before scientific use

- Resolve or bracket action/reaction sign and confirm traction units.
- Approve boundary/closure and equilibrium policy; quantify their sensitivity.
- Nonuniform manufactured solution convergence and experimental mesh sensitivity.
- Independent artifact validator, uncertainty propagation and solver performance
  validation on the canonical meshes.
- Moving-domain/reference-geometry assessment for the time series.
- Later: independently reconstructed, regularized nuclear velocity fields, D,
  divergence and vorticity with trajectory/timing/gradient validation.
- Later: held-out constitutive model comparisons with derivative objectivity and
  identifiability limits (15-minute samples over 24 hours).

Do not use auxiliary elastic displacement as experimental velocity or use the
elastic reconstruction law as evidence that the tissue is elastic. No viscosity,
relaxation time, or active stress is estimated by this implementation.
