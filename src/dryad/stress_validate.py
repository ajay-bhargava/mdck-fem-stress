"""Independent element-stress equilibrium checks for a conditional pilot solve."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
import meshio


def validate(directory: Path, write_vtu: bool = True):
    with np.load(directory/'stress.npz',allow_pickle=False) as data:
        d={k:data[k] for k in data.files}
    if 'nodes_xy_m' not in d:
        with np.load(directory.parent/'geometry.npz',allow_pickle=False) as geometry:
            d.update({k:geometry[k] for k in geometry.files})
    p=json.loads((directory/'provenance.json').read_text())
    for k in ('nodes_xy_m','stress_pa','membrane_resultant_n_per_m','applied_load_n','auxiliary_strain'):
        if not np.isfinite(d[k]).all(): raise ValueError(f'Nonfinite {k}')
    nodes=d['nodes_xy_m'];tri=d['triangles'];xy=nodes[tri]
    a=xy[:,1]-xy[:,0];b=xy[:,2]-xy[:,0]
    cross=a[:,0]*b[:,1]-a[:,1]*b[:,0]
    if np.any(cross<=0):raise ValueError('Invalid triangles')
    grad=np.stack((xy[:,[1,2,0],1]-xy[:,[2,0,1],1],xy[:,[2,0,1],0]-xy[:,[1,2,0],0]),axis=-1)/cross[:,None,None]
    local=np.einsum('eab,eib,e->eia',d['membrane_resultant_n_per_m'],grad,cross/2)
    internal=np.zeros_like(nodes)
    for axis in range(2):
        internal[:,axis]=np.bincount(tri.ravel(),weights=local[:,:,axis].ravel(),minlength=len(nodes))
    applied=d['applied_load_n'];raw=d['raw_load_n'];correction=d['equilibrium_correction_n']
    relative=float(np.linalg.norm(internal-applied)/max(np.linalg.norm(applied),np.finfo(float).tiny))
    if relative>1e-7:raise ValueError(f'Independent equilibrium residual {relative:g}')
    np.testing.assert_allclose(raw+correction,applied,rtol=1e-12,atol=1e-24)
    np.testing.assert_allclose(d['stress_pa'],d['stress_pa'].transpose(0,2,1),rtol=0,atol=1e-12)
    np.testing.assert_allclose(d['membrane_resultant_n_per_m'],d['stress_pa']*p['assumptions']['thickness_m'])
    principal=np.linalg.eigvalsh(d['stress_pa'])
    strain_norm=np.linalg.norm(d['auxiliary_strain'],axis=(1,2))
    reference=nodes.mean(axis=0);r=nodes-reference
    balances={}
    for name,f in [('raw',raw),('correction',correction),('applied',applied),('internal',internal)]:
        balances[name]={'sum_force_n':f.sum(axis=0).tolist(),
            'sum_moment_nm':float(np.sum(r[:,0]*f[:,1]-r[:,1]*f[:,0]))}
    summary={'position':str(d['position']),'frame':int(d['frame']),
        'independent_relative_nodal_equilibrium_error':relative,
        'balances_about_mean_node_position':balances,
        'correction_l2_over_raw_load_l2':float(np.linalg.norm(correction)/np.linalg.norm(raw)),
        'principal_stress_min_pa':float(principal.min()),'principal_stress_max_pa':float(principal.max()),
        'auxiliary_strain_frobenius_max':float(strain_norm.max()),
        'auxiliary_strain_frobenius_p95':float(np.percentile(strain_norm,95)),
        'warnings':['Conditional elastic reconstruction, not validated tissue rheology.',
                    'Mesh-dependent equilibrium projection; closure/boundary/correction sensitivity remains outstanding.',
                    'Auxiliary displacement and strain are not observed motion.'],
        'numerical_equilibrium_passed':True}
    if strain_norm.max()>.1:summary['warnings'].append('Auxiliary strain exceeds 0.1; do not interpret as a validated small physical deformation.')
    (directory/'validation.json').write_text(json.dumps(summary,indent=2)+'\n')
    if not write_vtu:
        return summary
    meshio.write(directory/'stress.vtu',meshio.Mesh(np.column_stack((nodes,np.zeros(len(nodes)))),[('triangle',tri)],
        point_data={'auxiliary_displacement_m':np.column_stack((d['auxiliary_displacement_m'],np.zeros(len(nodes)))),
                    'applied_load_n':np.column_stack((applied,np.zeros(len(nodes))))},
        cell_data={key:[value] for key,value in {
            'sigma_xx_pa':d['stress_pa'][:,0,0], 'sigma_yy_pa':d['stress_pa'][:,1,1],
            'sigma_xy_pa':d['stress_pa'][:,0,1], 'principal_min_pa':principal[:,0],
            'principal_max_pa':principal[:,1]}.items()}))
    return summary

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('directory',type=Path)
    print(json.dumps(validate(parser.parse_args().directory),indent=2))
