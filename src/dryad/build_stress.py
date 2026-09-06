"""Opt-in experimental stress reconstruction. All physical choices are explicit."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import yaml
from dryad.equilibrium import SheetSystem


def digest(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''): h.update(b)
    return h.hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data',type=Path,default=Path('data'))
    p.add_argument('--position',required=True)
    p.add_argument('--frame',type=int,required=True)
    p.add_argument('--young-pa',type=float,required=True,help='Auxiliary elastic closure; NOT inferred tissue modulus')
    p.add_argument('--poisson',type=float,required=True)
    p.add_argument('--thickness-m',type=float,required=True)
    p.add_argument('--traction-sign',type=int,choices=[-1,1],required=True,help='Assumed substrate-on-tissue sign relative to stored loads')
    p.add_argument('--equilibrium-policy',choices=['reject','project'],default='reject')
    p.add_argument('--acknowledge-inferred-pa',action='store_true',required=True)
    p.add_argument('--output',type=Path,required=True,help='New experiment directory, outside published/')
    a=p.parse_args()
    position=a.data.resolve()/a.position
    if position.parent!=a.data.resolve(): p.error('Invalid position')
    out=a.output.resolve()
    published=a.data.resolve().parent/'published'
    if out==published or published in out.parents: p.error('published/ is immutable')
    if out.exists(): p.error('Output must be a new directory')
    meshpath=position/'fem/mesh/mesh.npz'; loadpath=position/'fem/loads/nodal_forces.npz'
    configpath=position/'fem/loads/mapping_config.yaml'
    config=yaml.safe_load(configpath.read_text())
    meta=yaml.safe_load((position/'metadata.yaml').read_text())
    with np.load(meshpath,allow_pickle=False) as d: nodes=d['nodes_xy_um']*1e-6;tri=d['triangles']
    with np.load(loadpath,allow_pickle=False) as d:
        match=np.flatnonzero(d['frame_indices']==a.frame)
        if not len(match): p.error('Frame unavailable')
        i=match[0];loads=np.column_stack((d['fx_node'][i],d['fy_node'][i]))
        if not np.allclose(nodes,d['nodes_xy_m'],atol=1e-15,rtol=0):p.error('Mesh/load coordinates mismatch')
    system=SheetSystem(nodes,tri,a.young_pa,a.poisson,a.thickness_m)
    result=system.solve(loads*a.traction_sign,a.equilibrium_policy)
    interval=meta['imaging'].get('frame_interval_min')
    time_s=a.frame*interval*60 if interval is not None else None
    assumptions=dict(model='small-strain homogeneous isotropic plane-stress auxiliary elastic closure',
        young_pa=a.young_pa,poisson=a.poisson,thickness_m=a.thickness_m,
        boundary='free boundary; independent rigid-motion gauges for each connected component',
        equilibrium_policy=a.equilibrium_policy,
        correction_definition='project: Euclidean nodal-load projection off per-component rigid modes; mesh dependent, not an experimental correction estimate',
        assumed_traction_sign=a.traction_sign,action_reaction_resolved=False,
        auxiliary_displacement_is_measured_motion=False,
        stress_is_unique_from_traction_alone=False,
        geometry='fixed reference mesh; not updated with moving tissue',
        uncertainty='not quantified; sign, closure, boundary, correction and geometry sensitivity required before inference')
    provenance=dict(position=a.position,frame=a.frame,time_since_first_frame_s=time_s,
        traction_source=str(loadpath),source_mapping=config,assumptions=assumptions,
        hashes={str(q):digest(q) for q in [meshpath,loadpath,configpath]},
        normalized_equilibrium_residual=result.pop('normalized_equilibrium_residual'),
        rigid_load_projection_fraction=result.pop('rigid_load_projection_fraction'),
        status='conditional_forward_reconstruction_not_validated_for_rheology')
    out.mkdir(parents=True)
    np.savez_compressed(out/'stress.npz',**result,nodes_xy_m=nodes,triangles=tri,
        element_centroid_xy_m=nodes[tri].mean(axis=1),element_area_m2=system.area,
        component_id_node=system.component,valid_element=np.ones(len(tri),dtype=bool),
        position=np.asarray(a.position),frame=np.asarray(a.frame),
        time_since_first_frame_s=np.asarray(time_s if time_s is not None else np.nan),
        stress_units=np.asarray('Pa'),membrane_resultant_units=np.asarray('N/m'))
    (out/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')
    print(out)


if __name__=='__main__': main()
