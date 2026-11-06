"""Real-space projection with full fields, physical operators and joint Born draws.

Tensor axes use C order; physical local bits use little endian within each
triangle. No positivity assumption is made on the reduced Hamiltonian.
"""
from itertools import combinations

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.linalg import LinearOperator, eigsh

from .triangle_projector import triangle_layers, LOCAL_SPINS


def apply_axis(tensor, matrix, axis):
    return np.moveaxis(np.tensordot(matrix, tensor, axes=(1, axis)), 0, axis)


class Operator:
    def __init__(self, dims, terms):
        self.dims, self.terms = tuple(dims), terms
        self.dimension = int(np.prod(dims))

    def action(self, vector):
        tensor = vector.reshape(self.dims)
        out = np.zeros_like(tensor)
        for coefficient, factors in self.terms:
            term = tensor
            for axis, matrix in factors:
                term = apply_axis(term, matrix, axis)
            out += coefficient * term
        return out.ravel()

    def solve(self, seed=0):
        linear = LinearOperator((self.dimension, self.dimension), matvec=self.action,
                                dtype=np.float64)
        values, vectors = eigsh(linear, k=2, which='SA', tol=1e-11,
                                maxiter=20000,
                                v0=np.random.default_rng(seed).normal(size=self.dimension))
        order = np.argsort(values)
        energy, vector = values[order[0]], vectors[:, order[0]]
        residual = np.linalg.norm(self.action(vector) - energy * vector)
        if residual > 1e-7:
            raise RuntimeError(f'Unresolved eigenpair: residual={residual}')
        return float(energy), vector, float(values[order[1]]-energy), float(residual)

    def dense(self):
        result = np.zeros((self.dimension, self.dimension))
        for coefficient, factors in self.terms:
            by_axis = dict(factors)
            term = np.ones((1, 1))
            for axis, dim in enumerate(self.dims):
                term = np.kron(term, by_axis.get(axis, np.eye(dim)))
            result += coefficient*term
        return result


def canonical_eigh(matrix):
    values, vectors = eigh(matrix)
    pivots = np.argmax(abs(vectors), axis=0)
    vectors *= np.where(vectors[pivots, np.arange(len(values))] < 0, -1., 1.)
    return values, vectors


def triangles(ham, gamma, m, orientation=0, lam=1.):
    sites = triangle_layers(ham)[orientation]
    owner = {site: (t, a) for t, block in enumerate(sites) for a, site in enumerate(block)}
    coupling = {tuple(sorted(edge)): -j for edge, j in zip(ham.bonds, ham.J)}
    bases, spectra, zops, terms = [], [], [], []
    for t, block in enumerate(sites):
        diagonal = -ham.h * LOCAL_SPINS.sum(1).astype(float)
        for a, b in combinations(range(3), 2):
            diagonal += coupling[tuple(sorted((block[a], block[b])))] * LOCAL_SPINS[:, a] * LOCAL_SPINS[:, b]
        matrix = np.diag(diagonal)
        for state in range(8):
            for a in range(3):
                matrix[state ^ (1 << a), state] = -gamma
        values, vectors = canonical_eigh(matrix)
        basis = vectors[:, :m]
        spectra.append(values)
        bases.append(basis)
        zops.append([basis.T @ (LOCAL_SPINS[:, a, None] * basis) for a in range(3)])
        terms.append((1., [(t, np.diag(values[:m]))]))
    for (i, j), coupling_j in zip(ham.bonds, ham.J):
        t, a = owner[int(i)]
        u, b = owner[int(j)]
        if t != u:
            terms.append((-lam * coupling_j, [(t, zops[t][a]), (u, zops[u][b])]))
    return dict(sites=sites, owner=owner, bases=bases, spectra=spectra,
                zops=zops, operator=Operator([m]*len(sites), terms))


def hierarchy(ham, gamma, m, chi, orientation=0, origin=(0, 0), lam=1.,
              eigensolver='cpu'):
    L = int(round(np.sqrt(ham.n_sites/3)))
    if L % 2 or L < 4:
        raise ValueError('Superblock benchmark requires even L>=4')
    first = triangles(ham, gamma, m, orientation, lam)
    # Every triangle contains exactly one sublattice-0 site; use its cell as
    # the anchor in either orientation. Verify block connectivity explicitly.
    anchored = {}
    for t, block in enumerate(first['sites']):
        anchor, = [i//3 for i in block if i % 3 == 0]
        anchored[(anchor % L, anchor//L)] = t
    groups = [[anchored[((2*x+origin[0]+r) % L, (2*y+origin[1]+s) % L)]
               for s in range(2) for r in range(2)]
              for y in range(L//2) for x in range(L//2)]
    assert sorted(t for group in groups for t in group) == list(range(L*L))
    membership = {t: (b, a) for b, group in enumerate(groups) for a, t in enumerate(group)}
    local_terms = [[] for _ in groups]
    crossing = []
    for coefficient, factors in first['operator'].terms:
        blocks = {membership[t][0] for t, matrix in factors}
        if len(blocks) == 1:
            b = next(iter(blocks))
            local_terms[b].append((coefficient, [(membership[t][1], matrix) for t, matrix in factors]))
        else:
            crossing.append((coefficient, factors))
    retained, super_spectra, top_terms, decoders = [], [], [], []
    for b, group in enumerate(groups):
        adjacency = {t: set() for t in group}
        for _, factors in first['operator'].terms:
            if len(factors) == 2:
                t, u = [a for a, _ in factors]
                if t in adjacency and u in adjacency:
                    adjacency[t].add(u); adjacency[u].add(t)
        reached, pending = set(), [group[0]]
        while pending:
            t = pending.pop()
            if t not in reached:
                reached.add(t); pending.extend(adjacency[t]-reached)
        assert len(reached) == 4, 'Disconnected superblock'
        local = Operator([m]*4, local_terms[b])
        # Internal matrices are at most 4096 square for m<=8.
        dense = local.dense()
        if eigensolver == 'cpu':
            values, vectors = canonical_eigh(dense)
        elif eigensolver == 'cuda':
            import torch
            tensor = torch.as_tensor(dense, dtype=torch.float64, device='cuda')
            eigenvalues, eigenvectors = torch.linalg.eigh(tensor)
            values, vectors = eigenvalues.cpu().numpy(), eigenvectors.cpu().numpy()
            pivots = np.argmax(abs(vectors), axis=0)
            vectors *= np.where(vectors[pivots, np.arange(len(values))] < 0, -1., 1.)
            del tensor, eigenvalues, eigenvectors
        else:
            raise ValueError(f'Unknown eigensolver: {eigensolver}')
        W = vectors[:, :chi].copy()
        retained.append(W); super_spectra.append(values)
        top_terms.append((1., [(b, np.diag(values[:chi]))]))
        decoder = W.reshape((m,)*4+(chi,))
        for a, t in enumerate(group):
            decoder = apply_axis(decoder, first['bases'][t], a)
        decoders.append(decoder.reshape(4096, chi))
    cache = {}
    for coefficient, factors in crossing:
        projected = []
        for t, matrix in factors:
            b, a = membership[t]
            key = (t, matrix.tobytes())
            if key not in cache:
                W = retained[b]
                cache[key] = W.T @ apply_axis(W.reshape((m,)*4+(chi,)), matrix, a).reshape(m**4, chi)
            projected.append((b, cache[key]))
        top_terms.append((coefficient, projected))
    physical_sites = [[site for t in group for site in first['sites'][t]] for group in groups]
    # Decoder row packs triangle digits in C order, each digit has 3 little-endian bits.
    digits = np.array(np.unravel_index(np.arange(4096), (8,)*4)).T
    spins = (1-2*((digits[:, :, None] >> np.arange(3)) & 1)).reshape(4096, 12).astype(np.int8)
    return dict(first=first, groups=groups, retained=retained,
                super_spectra=super_spectra, decoders=decoders,
                sites=physical_sites, spins=spins,
                operator=Operator([chi]*len(groups), top_terms))


def physical_correlations(model, vector):
    decoders, sites, spins = model['decoders'], model['sites'], model['spins']
    chi, count = decoders[0].shape[1], len(decoders)
    state = vector.reshape((chi,)*count)
    n = sum(map(len, sites))
    magnetization, zz = np.zeros(n), np.eye(n)
    inserted = []
    for b, (D, block_sites) in enumerate(zip(decoders, sites)):
        moved = np.moveaxis(state, b, 0).reshape(chi, -1)
        rho = moved @ moved.T
        ops = [D.T @ (spins[:, a, None]*D) for a in range(12)]
        inserted.append([apply_axis(state, matrix, b).ravel() for matrix in ops])
        for a, i in enumerate(block_sites):
            magnetization[i] = np.sum(rho*ops[a].T)
            for c in range(a):
                j = block_sites[c]
                product = D.T @ ((spins[:, a]*spins[:, c])[:, None]*D)
                zz[i, j] = zz[j, i] = np.sum(rho*product.T)
    for b in range(count):
        for c in range(b):
            cross = np.asarray(inserted[b]) @ np.asarray(inserted[c]).T
            zz[np.ix_(sites[b], sites[c])] = cross
            zz[np.ix_(sites[c], sites[b])] = cross.T
    return magnetization, zz


def joint_samples(model, vector, draws, seed):
    """Independent Born draws using prefix POVMs and conditional contractions.

    The top state is conditioned on each completed superblock outcome. Within
    a block a binary prefix tree samples the exact joint 4096-outcome POVM.
    No block marginal independence or sign assumption is used.
    """
    rng = np.random.default_rng(seed)
    decoders, chi = model['decoders'], model['decoders'][0].shape[1]
    count = len(decoders)
    trees = []
    for D in decoders:
        levels = [np.einsum('ri,rj->rij', D, D)]
        for _ in range(12):
            level = levels[-1]
            levels.append(level.reshape(-1, 2, chi, chi).sum(1))
        trees.append(levels)
    output = np.empty((draws, 12*count), dtype=np.int8)
    # Limit duplicated top-state memory to ~32 MiB.
    batch_size = max(1, min(256, 2**22//vector.size))
    for start in range(0, draws, batch_size):
        length = min(batch_size, draws-start)
        conditioned = np.repeat(vector.reshape((1,)+(chi,)*count), length, axis=0)
        for b in range(count):
            flat = conditioned.reshape(length, chi, -1)
            rho = np.einsum('bik,bjk->bij', flat, flat)
            prefix = np.zeros(length, dtype=int)
            parent = np.trace(rho, axis1=1, axis2=2)
            for depth in range(1, 13):
                left = np.einsum('bij,bji->b', rho, trees[b][12-depth][2*prefix])
                probability = np.clip(left/parent, 0, 1)
                right = rng.random(length) >= probability
                prefix = 2*prefix+right
                parent = np.where(right, parent-left, left)
            rows = decoders[b][prefix]
            conditioned = np.einsum('bi,bik->bk', rows, flat)
            conditioned /= np.linalg.norm(conditioned, axis=1)[:, None]
            conditioned = conditioned.reshape((length,)+(chi,)*(count-b-1))
            output[start:start+length, model['sites'][b]] = model['spins'][prefix]
    return output
