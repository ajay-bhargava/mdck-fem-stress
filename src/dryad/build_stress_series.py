"""Build all pos01 frames with one factorization and one shared geometry artifact."""
from __future__ import annotations
import argparse
import json
import shutil
import tempfile
import time
from pathlib import Path
import numpy as np
import yaml
from dryad.build_stress import digest
from dryad.equilibrium import SheetSystem
from dryad.stress_validate import validate


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data',type=Path,default=Path('data'))
    args=parser.parse_args()
    position=args.data.resolve()/'pos01'
    out=position/'fem/stress/time_series'
    if out.exists():raise ValueError('Series exists; will not overwrite')
    meshpath=position/'fem/mesh/mesh.npz';loadpath=position/'fem/loads/nodal_forces.npz'
    configpath=position/'fem/loads/mapping_config.yaml'
    config=yaml.safe_load(configpath.read_text())
    meta=yaml.safe_load((position/'metadata.yaml').read_text())
    # Carry forward the approved, explicitly conditional pilot choices.
    pilot=json.loads((position/'fem/stress/pilot_frame000_sign_plus/provenance.json').read_text())
    assumptions=pilot['assumptions']
    with np.load(meshpath,allow_pickle=False) as d:nodes=d['nodes_xy_um']*1e-6;tri=d['triangles']
    with np.load(loadpath,allow_pickle=False) as d:
        frames=d['frame_indices'];fx=d['fx_node'];fy=d['fy_node']
        np.testing.assert_allclose(nodes,d['nodes_xy_m'],atol=1e-15,rtol=0)
    if not np.array_equal(frames,np.arange(meta['imaging']['n_frames'])):raise ValueError('Incomplete source frame set')
    interval=meta['imaging']['frame_interval_min']*60
    hashes={str(p):digest(p) for p in (meshpath,loadpath,configpath)}
    started=time.perf_counter()
    system=SheetSystem(nodes,tri,assumptions['young_pa'],assumptions['poisson'],assumptions['thickness_m'])
    temporary=Path(tempfile.mkdtemp(prefix='.series-',dir=out.parent))
    rows=[]
    try:
        np.savez_compressed(temporary/'geometry.npz',nodes_xy_m=nodes,triangles=tri,
            element_centroid_xy_m=nodes[tri].mean(axis=1),element_area_m2=system.area,
            component_id_node=system.component,valid_element=np.ones(len(tri),dtype=bool))
        for i,frame in enumerate(frames):
            frame=int(frame);directory=temporary/f'frame{frame:03d}';directory.mkdir()
            result=system.solve(np.column_stack((fx[i],fy[i]))*assumptions['assumed_traction_sign'],assumptions['equilibrium_policy'])
            residual=result.pop('normalized_equilibrium_residual');fraction=result.pop('rigid_load_projection_fraction')
            np.savez_compressed(directory/'stress.npz',**result,position=np.asarray('pos01'),frame=np.asarray(frame),
                time_since_first_frame_s=np.asarray(frame*interval),stress_units=np.asarray('Pa'),
                membrane_resultant_units=np.asarray('N/m'))
            provenance=dict(position='pos01',frame=frame,time_since_first_frame_s=frame*interval,
                geometry_source='../geometry.npz',traction_source=str(loadpath),source_mapping=config,
                assumptions=assumptions,hashes=hashes,normalized_equilibrium_residual=residual,
                rigid_load_projection_fraction=fraction,status='conditional_forward_reconstruction_not_validated_for_rheology')
            (directory/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
            report=validate(directory,write_vtu=frame in (0,48,96))
            magnitude=np.max(np.abs(np.linalg.eigvalsh(result['stress_pa'])),axis=1)
            rows.append(dict(frame=frame,time_s=frame*interval,
                equilibrium_error=report['independent_relative_nodal_equilibrium_error'],
                correction_fraction=fraction,stress_magnitude_p99_pa=float(np.percentile(magnitude,99)),
                stress_magnitude_max_pa=float(magnitude.max()),
                auxiliary_strain_max=report['auxiliary_strain_frobenius_max']))
            print(f'frame {frame}: independent residual {rows[-1]["equilibrium_error"]:.3e}',flush=True)
        summary=dict(position='pos01',complete=True,frame_count=len(frames),frame_interval_s=interval,
            assumptions=assumptions,source_hashes=hashes,frames=rows,
            color_max_pa=max(r['stress_magnitude_p99_pa'] for r in rows),
            color_definition='Fixed across all frames: maximum of per-frame 99th element percentiles; values above limit saturate.',
            stress_field='maximum absolute principal stress',units='Pa',
            traction_sign_alternative='Reverse all stress, load, displacement and strain tensors/vectors for opposite sign; scalar magnitude unchanged.',
            geometry='Single shared static reference mesh, NOT time-evolving tissue geometry.',
            numerical_equilibrium_passed=True,scientifically_validated_for_rheology=False,
            warning='Auxiliary elastic strains are not measured motion; model/geometry/correction sensitivity remains unvalidated.',
            runtime_s=time.perf_counter()-started)
        (temporary/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        temporary.rename(out)
        print(out)
    except BaseException:
        shutil.rmtree(temporary)
        raise

if __name__=='__main__':main()
