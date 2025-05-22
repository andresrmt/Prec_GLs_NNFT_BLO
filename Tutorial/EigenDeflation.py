# -*- coding: utf-8 -*-
# This code was built using `NumPy: 2.2.5`, `SciPy: 1.15.1`.
# Written by Andrés MT

import numpy as np
import scipy.sparse as sp


'''
    Unitary change of basis U such that U^\dagger A U is block diagonal
'''

# =====================
# ** Operator forms **

# Vectors
# x is the vector where we apply the transform, y is the eigenvector, and i is its first nonzero index
def Uv_op(x, y, i=0):
    '''
        Computes U(y,i)x = h
        ===============================================================
        Inputs:
        x:    array (numeric) of size (n,)
        y:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        h:    array (numeric) of size (n,)
    '''
    # First block
    h = x.copy()         # Copy x since its entries are preserved
    
    # Second block
    v = y[i:]            # We can just recycle the code from before by defining an auxiliary variable
    # Build a_v and b_v
    norms = np.cumsum(v * v.conjugate()).real**0.5
    b = np.r_[ norms[-1]**-1,  v[1:].conjugate() / (norms[:-1] * norms[1:])]
    a = np.r_[0, norms[:-1]/(norms[1:])]
    
    # Dense core: SCS↑
    w = b * x[i:]
    c = (w.sum() - np.cumsum(w));    c[-1] = 0.0;    c += w[0]
    # Weight vector
    c *= v
    # Affine translation
    c -= a * x[i:]
    
    # Assign entries
    h[i:] = c
    return h
def Uv_in(x, y, i=0):
    '''
        Computes the solution of U(y,i) h = x
        ===============================================================
        Inputs:
        x:    array (numeric) of size (n,)
        y:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        h:    array (numeric) of size (n,)
    '''
    # First block
    h = x.copy()         # Copy x since its entries are preserved
    
    # Second block
    v = y[i:]            # We can just recycle the code from before by defining an auxiliary variable
    # Build a_v and b_v
    norms = np.cumsum(v * v.conjugate()).real**0.5
    b = np.r_[ norms[-1]**-1,  v[1:] / (norms[:-1] * norms[1:])]
    a = np.r_[0, norms[:-1]/(norms[1:])]
    
    # Dense core: SCS↓
    w = v.conj() * x[i:]
    c = np.roll( np.cumsum(w),1)
    # Weight vector
    c *= b
    # Affine translation
    c -= a * x[i:]
    
    # Assign entries
    h[i:] = c
    return h

# Matrices
# X is the matrix where we apply the transform, y is the eigenvector, and i is its first nonzero index
def Um_op(X, y, i=0):
    '''
        Computes U(y,i)X = H
        ===============================================================
        Inputs:
        X:    array (numeric) of size (n,m)
        y:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        H:    array (numeric) of size (n,m)
    '''
    # First block
    H = X.copy()         # Copy X since its entries are preserved
    
    # Second block
    v = y[i:]            # subvector used to define transformation

    # Build a_v and b_v
    norms = np.cumsum(v * v.conjugate()).real**0.5
    b = np.r_[ norms[-1]**-1,  v[1:].conjugate() / (norms[:-1] * norms[1:])]
    a = np.r_[0, norms[:-1]/(norms[1:])]

    # Dense core: SCS↑
    W = b[:, None] * X[i:]  # shape (n-i, m)
    C = W.sum(0) - np.cumsum(W, axis = 0);    C[-1] = 0.0;    C += W[0]
    # Weight matrix
    C *= v[:, None]
    # Affine translation
    C -= a[:, None] * X[i:]

    H[i:] = C
    return H
def Um_in(X, y, i=0):
    '''
        Computes the solution of U(y,i) H = X
        ===============================================================
        Inputs:
        X:    array (numeric) of shape (n, m)
        y:    array (numeric) of shape (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        H:    array (numeric) of shape (n, m)
    '''
    # First block
    H = X.copy()         # Copy X since its entries are preserved
    
    # Second block
    v = y[i:]

    # Build a_v and b_v
    norms = np.cumsum(v * v.conjugate()).real**0.5
    b = np.r_[ norms[-1]**-1,  v[1:] / (norms[:-1] * norms[1:])]
    a = np.r_[0, norms[:-1]/(norms[1:])]

    # Dense core: SCS↓
    W = v.conj()[:, None] * X[i:]
    C = np.roll(np.cumsum(W, axis=0), 1, axis=0)
    # Weight matrix
    C *= b[:, None]
    # Affine correction
    C -= a[:, None] * X[i:]

    H[i:] = C
    return H

# =====================
# ** Matrix forms **
# x is the eigenvector, and i is its first nonzero index
def U_mat(x, i=0):
    '''
        Computes U(x,i)
        ===============================================================
        Inputs:
        x:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        U:    array (numeric) of size (n,n). Its inverse is given by U.conj().T
    '''
    # First block
    n_diag = x[:i].size
    Id = sp.eye(n_diag, format = 'csr', dtype = x.dtype)
    
    # Second block
    v = x[i:]            # We can just recycle the code from before by defining an auxiliary variable
    n = v.size
    # Core upper triangle
    U = np.zeros([n,n]);    U[:-1, 1:] = np.triu(np.ones([n-1,n-1]),0);    U[:,0] = 1.0
    # Compute weights
    norms = np.cumsum(v*v.conjugate()).real**0.5
    b = np.r_[ norms[-1]**-1,  v[1:].conjugate() / (norms[:-1] * norms[1:])]
    a = np.r_[0, norms[:-1]/(norms[1:])]
    
    # Include weights in dense core
    U = (v[::,np.newaxis] * U * b)
    # Correct diagonal
    np.fill_diagonal(U, U.diagonal() - a)
    
    # Optional: Convert to sparse
    U = sp.csr_array(U)
    
    return sp.block_diag((Id, U), format = 'csr')



'''
    Hermitian change of basis Q such that Q^{-1} A Q is block diagonal
'''

# =====================
# ** Operator forms **
# Vectors
# u is the vector where we apply the transform, v is the eigenvector, and i is its first nonzero index
def Qv_op(u, v, i=0):
    '''
        Computes Q(v,i)u = a
        ===============================================================
        Inputs:
        u:    array (numeric) of size (n,)
        v:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        a:    array (numeric) of size (n,)
    '''
    a = -v[i] * u;    a[i] = np.vdot(v,u);    a[i+1:] += u[i] * v[i+1:]
    return a
def Qv_in(u, v, i=0):
    '''
        Computes the solution of Q(v,i) a = u
        ===============================================================
        Inputs:
        u:    array (numeric) of size (n,)
        v:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        a:    array (numeric) of size (n,)
    '''
    # Projection
    a = np.vdot(v,u) * v / np.vdot(v,v)
    # Correction
    a[:i]   -= u[:i]
    a[i+1:] -= u[i+1:]
    # Scaling
    a /= v[i]
    return a

# Matrices
# U is the matrix where we apply the transform, v is the eigenvector, and i is its first nonzero index
def Qm_op(U, v, i=0):
    '''
        Computes Q(v,i)U = A
        ===============================================================
        Inputs:
        U:    array (numeric) of size (n,m)
        v:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        A:    array (numeric) of size (n,m)
    '''
    A = -v[i] * U;    A[i] = np.dot(v.conj(),U);    A[i+1:] += U[i] * v[i+1:, None]
    return A
def Qm_in(U, v, i=0):
    '''
        Computes the solution of Q(v,i) A = U
        ===============================================================
        Inputs:
        U:    array (numeric) of size (n,m)
        v:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        A:    array (numeric) of size (n,m)
    '''
    # Projection
    A = np.dot(v.conj(),U) * (v[:, None] / np.vdot(v,v))
    # Correction
    A[:i]   -= U[:i]
    A[i+1:] -= U[i+1:]
    # Scaling
    A /= v[i]
    
    return A

# =====================
# ** Matrix forms **
# x is the eigenvector, and i is its first nonzero index
def Q_mat(v, i = 0):
    '''
        Computes Q(v,i)
        ===============================================================
        Inputs:
        v:    array (numeric) of size (n,)
        i:    int in [0,n-1] is the index where v≠0 for the first time
        Output:
        Q:    array (numeric) of size (n,n)
    '''
    n = v.size
    # Main diagonal
    Q = sp.diags(np.broadcast_to(-v[i], (n,)), offsets = 0, shape = (n,n), format = 'dok', dtype = v.dtype)
    # First row
    Q[i] = v.conj()
    # First column
    Q[:,i] = v
    return Q.tocsr()
def Q_mat_inv(v, i = 0):
    '''
        Computes the inverse of Q(v,i)
        ==================================================================
        Inputs:
        v:       array (numeric) of size (n,)
        i:       int in [0,n-1] is the index where v≠0 for the first time
        Output:
        Q^-1:    array (numeric) of size (n,n)
    '''
    n = v.size
    # Projection
    Qi = sp.kron(v[:,np.newaxis]/( v[i] * np.vdot(v,v).real ), v.conj(), format = 'csr')
    # Anisotropic corrector
    d = np.ones(n, dtype=v.dtype);    d[i] = 0.0;    d /= v[i]
    Qi.setdiag( Qi.diagonal() - d )
    return Qi