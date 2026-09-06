import unittest
import numpy as np
from dryad.equilibrium import SheetSystem


def square(n=4):
    nodes=np.array([(x,y) for y in np.linspace(0,1e-3,n) for x in np.linspace(0,1e-3,n)])
    tri=[]
    for j in range(n-1):
        for i in range(n-1):
            a=j*n+i;tri.extend([[a,a+1,a+n+1],[a,a+n+1,a+n]])
    return nodes,np.array(tri)


def boundary_load(nodes,tri,stress,h):
    edges={}
    for t in tri:
        for i,j in zip(t,np.roll(t,-1)):
            key=tuple(sorted((i,j)))
            if key in edges: del edges[key]
            else:edges[key]=(i,j)
    f=np.zeros_like(nodes)
    for i,j in edges.values():
        tangent=nodes[j]-nodes[i]
        integrated_normal=np.array([tangent[1],-tangent[0]])
        load=stress@integrated_normal*h/2
        f[i]+=load;f[j]+=load
    return f


class SheetTests(unittest.TestCase):
    def test_uniform_stress_analytic_boundary_loads_and_refinement(self):
        expected=np.array([[31.,7.],[7.,-12.]])
        for n in [3,5,9]:
            nodes,tri=square(n);s=SheetSystem(nodes,tri,1000.,.3,5e-6)
            r=s.solve(boundary_load(nodes,tri,expected,s.h))
            np.testing.assert_allclose(r['stress_pa'],np.broadcast_to(expected,r['stress_pa'].shape),atol=1e-9)
            np.testing.assert_allclose(r['membrane_resultant_n_per_m'],r['stress_pa']*s.h)
            self.assertLess(r['normalized_equilibrium_residual'],1e-10)

    def test_rigid_rotation_has_zero_strain(self):
        nodes,tri=square();s=SheetSystem(nodes,tri,1000.,.3,5e-6)
        u=np.column_stack((-nodes[:,1],nodes[:,0])).ravel()
        np.testing.assert_allclose(np.einsum('eij,ej->ei',s.B,u[s.dofs]),0,atol=1e-14)

    def test_unbalanced_load_rejected_and_projection_explicit(self):
        nodes,tri=square();s=SheetSystem(nodes,tri,1000.,.3,5e-6)
        f=np.zeros_like(nodes);f[0,0]=1e-9
        with self.assertRaises(ValueError):s.solve(f)
        r=s.solve(f,'project')
        np.testing.assert_allclose(r['raw_load_n'],f)
        np.testing.assert_allclose(r['applied_load_n'].sum(axis=0),0,atol=1e-23)
        self.assertGreater(np.linalg.norm(r['equilibrium_correction_n']),0)

    def test_disconnected_components(self):
        nodes,tri=square();nodes=np.vstack((nodes,nodes+[.002,0]));tri=np.vstack((tri,tri+len(nodes)//2))
        s=SheetSystem(nodes,tri,1000.,.3,5e-6)
        self.assertEqual(s.R.shape[1],6)
        expected=np.eye(2)*10
        r=s.solve(boundary_load(nodes,tri,expected,s.h))
        np.testing.assert_allclose(r['stress_pa'],np.broadcast_to(expected,r['stress_pa'].shape),atol=1e-9)

    def test_thickness_scaling_matches_independent_solves(self):
        nodes,tri=square()
        f=boundary_load(nodes,tri,np.array([[31.,7.],[7.,-12.]]),5e-6)
        reference=SheetSystem(nodes,tri,1000.,.3,5e-6).solve(f)
        for h in (2.5e-6,10e-6):
            result=SheetSystem(nodes,tri,1000.,.3,h).solve(f)
            np.testing.assert_allclose(result['stress_pa'],reference['stress_pa']*5e-6/h,atol=1e-9)
            np.testing.assert_allclose(result['membrane_resultant_n_per_m'],reference['membrane_resultant_n_per_m'],atol=1e-14)
            np.testing.assert_allclose(result['auxiliary_displacement_m'],reference['auxiliary_displacement_m']*5e-6/h,atol=1e-14)

    def test_modulus_scales_auxiliary_motion_not_uniform_stress(self):
        nodes,tri=square();f=boundary_load(nodes,tri,np.eye(2)*10,5e-6)
        a=SheetSystem(nodes,tri,1000.,.3,5e-6).solve(f)
        b=SheetSystem(nodes,tri,2000.,.3,5e-6).solve(f)
        np.testing.assert_allclose(a['stress_pa'],b['stress_pa'],atol=1e-10)
        np.testing.assert_allclose(a['auxiliary_displacement_m'],2*b['auxiliary_displacement_m'],atol=1e-14)

if __name__=='__main__':unittest.main()
