"""Assumption-explicit P1 plane-stress reconstruction; not measured motion/rheology."""
from __future__ import annotations
import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.sparse.linalg import splu


class SheetSystem:
    def __init__(self, nodes_m, triangles, young_pa, poisson, thickness_m):
        self.nodes = np.asarray(nodes_m, dtype=float)
        self.tri = np.asarray(triangles, dtype=np.int64)
        if not (np.isfinite([young_pa, poisson, thickness_m]).all() and young_pa > 0
                and thickness_m > 0 and -1 < poisson < .5):
            raise ValueError('Require E>0, h>0, -1<nu<0.5')
        self.E, self.h = young_pa, thickness_m
        p = self.nodes[self.tri]
        a, b = p[:, 1]-p[:, 0], p[:, 2]-p[:, 0]
        cross = a[:,0]*b[:,1]-a[:,1]*b[:,0]
        if np.any(cross <= 0):
            raise ValueError('Require counterclockwise, nondegenerate triangles')
        self.area = cross / 2
        grad = np.stack((p[:, [1,2,0], 1]-p[:, [2,0,1], 1],
                         p[:, [2,0,1], 0]-p[:, [1,2,0], 0]), axis=-1)/cross[:,None,None]
        self.B = np.zeros((len(p), 3, 6))
        self.B[:,0,0::2] = grad[:,:,0]
        self.B[:,1,1::2] = grad[:,:,1]
        self.B[:,2,0::2] = grad[:,:,1]
        self.B[:,2,1::2] = grad[:,:,0]
        self.C = young_pa/(1-poisson**2)*np.array([[1,poisson,0],[poisson,1,0],[0,0,(1-poisson)/2]])
        self.dofs = (2*self.tri[:,:,None]+np.arange(2)).reshape(-1,6)
        local = np.einsum('eai,ab,ebj,e->eij',self.B,self.C,self.B,self.area*thickness_m)
        n = 2*len(self.nodes)
        self.K = sparse.coo_matrix((local.ravel(),(np.broadcast_to(self.dofs[:,:,None],local.shape).ravel(),
                  np.broadcast_to(self.dofs[:,None,:],local.shape).ravel())),shape=(n,n)).tocsc()
        edges = self.tri[:,[0,1,1,2,2,0]].reshape(-1,2)
        graph = sparse.coo_matrix((np.ones(len(edges)),(edges[:,0],edges[:,1])),shape=(len(self.nodes),)*2)
        count, self.component = connected_components(graph,directed=False)
        rows,cols,values = [],[],[]
        for c in range(count):
            ids = np.flatnonzero(self.component==c)
            if len(ids)<3: raise ValueError('Isolated mesh nodes')
            xy = self.nodes[ids]-self.nodes[ids].mean(axis=0)
            modes = np.zeros((len(ids),2,3))
            modes[:,0,0]=1; modes[:,1,1]=1
            modes[:,0,2]=-xy[:,1]; modes[:,1,2]=xy[:,0]
            modes /= np.linalg.norm(modes.reshape(-1,3),axis=0)[None,None,:]
            for k in range(3):
                rows.extend((2*ids[:,None]+np.arange(2)).ravel())
                cols.extend([3*c+k]*(2*len(ids)))
                values.extend(modes[:,:,k].ravel())
        self.R = sparse.csc_matrix((values,(rows,cols)),shape=(n,3*count))
        # K scaled by E*h; Lagrange constraints remove rigid motion, not physical supports.
        self.factor = splu(sparse.bmat([[self.K/(young_pa*thickness_m),self.R],
                                      [self.R.T,None]],format='csc'))

    def solve(self, forces_n, correction='reject', tolerance=1e-10):
        f = np.asarray(forces_n,dtype=float).reshape(-1)
        if f.shape!=(len(self.nodes)*2,) or not np.isfinite(f).all(): raise ValueError('Invalid nodal loads')
        incompatible = np.asarray(self.R@(self.R.T@f)).ravel()
        fraction = np.linalg.norm(incompatible)/max(np.linalg.norm(f),np.finfo(float).tiny)
        if correction not in ('reject','project'): raise ValueError('Unknown equilibrium policy')
        if correction=='reject' and fraction>tolerance:
            raise ValueError(f'Loads incompatible with free boundaries: rigid-load projection fraction {fraction:g}; no automatic correction')
        delta = -incompatible if correction=='project' else np.zeros_like(f)
        used = f+delta
        solution = self.factor.solve(np.r_[used/(self.E*self.h), np.zeros(self.R.shape[1])])
        u = solution[:len(f)]
        residual = self.K@u-used
        relative = np.linalg.norm(residual)/max(np.linalg.norm(used),np.finfo(float).tiny)
        if relative>1e-7: raise ValueError(f'Equilibrium residual failed: {relative:g}')
        strain_eng = np.einsum('eij,ej->ei',self.B,u[self.dofs])
        stress = strain_eng@self.C.T
        tensor = np.zeros((len(self.tri),2,2));tensor[:,0,0]=stress[:,0];tensor[:,1,1]=stress[:,1]
        tensor[:,0,1]=tensor[:,1,0]=stress[:,2]
        strain = np.zeros_like(tensor);strain[:,0,0]=strain_eng[:,0];strain[:,1,1]=strain_eng[:,1]
        strain[:,0,1]=strain[:,1,0]=strain_eng[:,2]/2
        return dict(auxiliary_displacement_m=u.reshape(-1,2), auxiliary_strain=strain,
                    stress_pa=tensor, membrane_resultant_n_per_m=tensor*self.h,
                    raw_load_n=f.reshape(-1,2), applied_load_n=used.reshape(-1,2),
                    equilibrium_correction_n=delta.reshape(-1,2),
                    equilibrium_residual_n=residual.reshape(-1,2),
                    constraint_multiplier_n=solution[len(f):]*self.E*self.h,
                    normalized_equilibrium_residual=relative, rigid_load_projection_fraction=fraction)
