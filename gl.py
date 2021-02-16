import numpy as np

# return the Legendre-Gauss-Lobatto nodes
# N should be the order of polynomial
# thus N1 should be the number of nodes or quads
def lglnodes(N):
    N1 = N+1
    x = np.cos( np.pi*np.arange(0,N+1)/float(N) )
    P = np.zeros((N1,N1))

    xold = 2

    while np.abs(x-xold).max() > np.finfo(float).eps:
        xold = x
        P[:,0] = 1
        P[:,1] = x

        for k in range(2,N+1):
            P[:,k] = ((2*k-1)*x*P[:,k-1] - (k-1)*P[:,k-2])/float(k)

        x = xold - (x*P[:,N] - P[:,N-1]) / (N1*P[:,N])


    w = 2.0/(N*N1*P[:,N]**2)
    return (0.5*(x[::-1]+1),0.5*w[::-1])

# Forms Lagrange interpolating polynomial defined at nodes x1, evaluates the
# polynomial (matrix G) and its derivative (matrix D) at points x2
def lagint(x1, x2):
    p1 = len(x1) - 1
    p2 = len(x2) - 1
    D = np.zeros((p1+1, p1+1))
    w = np.ones((p1+1,1))

    for j in range(p1+1):
        for k in range(p1+1):
            if j != k:
                w[j] *= x1[j] - x1[k]

    w = 1.0/w

    for i in range(p1+1):
        for j in range(p1+1):
            if i != j:
                D[i,j] = w[j]/w[i] / (x1[i] - x1[j])
        D[i,i] = 0.0
        for j in range(p1+1):
            if i != j:
                D[i,i] -= D[i,j]

    G = np.zeros((p2+1, p1+1))

    for i in range(p2+1):
        ell = 1.0
        for j in range(p1+1):
            ell *= x2[i] - x1[j]
        for j in range(p1+1):
            if x2[i] == x1[j]:
                G[i,j] = 1
            else:
                G[i,j] = ell*w[j]/(x2[i] - x1[j])

    GD = G.dot(D)
    return G, GD
