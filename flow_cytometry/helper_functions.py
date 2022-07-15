# Imports
import time
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy import stats, linalg, optimize, special
from sklearn.mixture import GaussianMixture

# 1. Functions for gating
# R


# Compute mahalanobis distance
def mahalanobis(U, v, cov):
    cov_inv = linalg.inv(cov)
    Uc = U - v
    dist = np.sqrt(np.sum(Uc @ cov_inv * Uc, axis=1))
    
    return dist


# Distinguish cells from debris
def gate_cells(df, nclust, random_state=None, p=0.99, savefig=False):
    
    X = df[['FSC-A', 'SSC-A']].values
    
    # Fit multivariate gaussians
    gm = GaussianMixture(nclust, random_state=random_state)    
    fit = gm.fit(X)
    means = fit.means_
    covs = fit.covariances_
    
    # Plot histogram, extract bin edges
    _, xbins, ybins, _ = plt.hist2d(X[:,0], X[:,1], bins=200, cmin=1, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.xlabel('hlog(FSC-A)')
    plt.ylabel('hlog(SSC-A)')
    
    # Derive bin centers
    xwidth = np.diff(xbins)[0]
    ywidth = np.diff(ybins)[0]
    xcenters = xbins[:-1] + xwidth/2
    ycenters = ybins[:-1] + ywidth/2

    # Make coordinate grid
    xmesh, ymesh = np.meshgrid(xcenters, ycenters)
    xygrid = np.column_stack((xmesh.flatten(), ymesh.flatten()))
    
    # Compute cluster probabilities on grid
    probs = gm.predict_proba(xygrid)
    
    # Plot decision boundaries
    for j in range(nclust):
        boundary = np.prod(probs[:, np.newaxis, j] >= probs, axis=1).reshape(200,200)
        # Plot 2d histogram of data and contours of fitted Gaussians
        plt.contour(xmesh, ymesh, boundary, levels=[0.5], colors='lime')
    
    # Identify cluster of interest (furthest from origin)
    cell_clust = np.argmax(np.sum(means**2, axis=1))
    
    # Predict cluster identity of each data point
    labels = gm.predict(X)
    
    # Compute mahalanobis distances of data points to cell cluster mean
    dist = mahalanobis(X, means[cell_clust], covs[cell_clust])
    dist[labels != cell_clust] = np.nan  # ignore other clusters
    
    # Compute distance inside which p*100 percent of the data points lie
    crit_dist = np.nanquantile(dist, p)
    
    # Plot contour of critical distance
    mahala_grid = mahalanobis(xygrid, means[cell_clust], covs[cell_clust]).reshape(200,200)
    plt.contour(xmesh, ymesh, mahala_grid, levels=[crit_dist], colors='red')
    if savefig:
        plt.savefig('cell_gate.pdf', pad_inches=0)
    plt.show()
    
    # Find cells within critical distance
    is_cell = dist < crit_dist

    return is_cell

# Distinguish singlets from doublets
def gate_singlets(df, cell_gate, p_left=1e-3, p_right=0.90, savefig=False):
    
    X = df[['FSC-A', 'FSC-H']].values
    
    # Filter previously gated data
    Y = X[cell_gate,:]
    
    # Compute least squares vector through data
    x = Y[:,0]
    y = Y[:,1]
    slope = np.sum(x*y) / np.sum(x*x)
    u = np.array((1, slope))
    u = u / np.sqrt(np.sum(u**2))  # normalize
    
    # Complement of u 
    v = np.array((1, -1/slope))
    v = v / np.sqrt(np.sum(v**2))  # normalize
    
    # Project filtered data on v
    Y_proj = Y @ v.reshape(2,1)
    plt.hist(Y_proj, bins=100, density=True)
    vlines = np.quantile(Y_proj, (p_left, p_right))
    plt.axvline(vlines[0], c='red')
    plt.axvline(vlines[1], c='red')
    plt.show()

    # Plot data with chosen boundaries
    plt.hist2d(x, y, bins=200, cmin=1, norm=mpl.colors.LogNorm())
    plt.colorbar()
    plt.axline(v*vlines[0], v*vlines[0]+u, c='red')
    plt.axline(v*vlines[1], v*vlines[1]+u, c='red')
    plt.xlabel('hlog(FSC-A)')
    plt.ylabel('hlog(FSC-H)')
    if savefig:
        plt.savefig('singlet_gate.pdf', pad_inches=0)
    plt.show()
    

    # Project all full data X on v and check conditions
    X_proj = X @ v
    is_singlet = (X_proj > vlines[0]) & (X_proj < vlines[1]) & cell_gate
    
    return is_singlet

# Distinguish alive from dead
def gate_alive(df, nclust, singlet_gate, p=1, random_state=None, savefig=False):

    X = df[['PE-Texas Red-A', 'PE-Texas Red-H']].values
    
    # Filter data with previous gate
    Y = X[singlet_gate,:]
    
    # Fit multivariate gaussian
    gm = GaussianMixture(nclust, random_state=random_state)    
    fit = gm.fit(Y)
    means = fit.means_
    covs = fit.covariances_
    
    # Identify the "alive" cluster as the one with mean nearest 0
    alive_clust = np.argmin(np.sum(means**2, axis=1))

    # Plot histogram, extract bin edges
    _, xbins, ybins, _ = plt.hist2d(Y[:,0], Y[:,1], bins=200, cmin=1)
    plt.colorbar()
    plt.xlabel('hlog(Red-A)')
    plt.ylabel('hlog(Red-H)')

    # Derive bin centers
    xwidth = np.diff(xbins)[0]
    ywidth = np.diff(ybins)[0]
    xcenters = xbins[:-1] + xwidth/2
    ycenters = ybins[:-1] + ywidth/2

    # Make coordinate grid
    xmesh, ymesh = np.meshgrid(xcenters, ycenters)
    xygrid = np.column_stack((xmesh.flatten(), ymesh.flatten()))
    
    # Compute cluster probabilities on grid
    probs = gm.predict_proba(xygrid)
    
    # Draw decision boundaries
    for j in range(nclust):
        boundary = np.prod(probs[:, np.newaxis, j] >= probs, axis=1).reshape(200,200)
        plt.contour(xmesh, ymesh, boundary, levels=[0.5], colors='lime')

    # Predict cluster identities of data points
    labels = gm.predict(Y)

    # Compute distance from alive cluster mean
    dist = mahalanobis(Y, means[alive_clust], covs[alive_clust])
    dist[labels != alive_clust] = np.nan  # ignore data from other clusters
    
    # Compute distance inside which p*100 percent of the data points lie
    crit_dist = np.nanquantile(dist, p)
    
    # Plot contour of critical distance
    mahala_grid = mahalanobis(xygrid, means[alive_clust], covs[alive_clust]).reshape(200,200)
    plt.contour(xmesh, ymesh, mahala_grid, levels=[crit_dist], colors='red')
    if savefig:
        plt.savefig('alive_gate.pdf', pad_inches=0)
    plt.show()
    
    # Gate
    is_alive = dist <= crit_dist
    alive_gate = np.full(X.shape[0], False)  # Make the gate!
    alive_gate[singlet_gate] = is_alive
    
    return alive_gate



# 2. Functions for estimating gamma mixture parameters

# Function definitions for root-finding:
# alpha_eq() is the equation of which the root maximizes alpha
def alpha_eq(alpha, C):
    return np.log(alpha) - special.digamma(alpha) - C

# 1st derivative of alpha_eq (for Halley's root-finding method)
def alpha_eq_prime(alpha, C):
    return 1/alpha - special.polygamma(1, alpha)

# 2nd derivative of alpha_eq (for Halley's method)
def alpha_eq_prime2(alpha, C):
    return -1/alpha**2 - special.polygamma(2, alpha)

# Guess initial parameters for gamma mixture fit
def init_params(X, ncomp, random_state=None):
    
    X = X.reshape(-1,1)
    n = X.shape[0]

    # Take cube root: if X is mixture of gammas, X_cbrt approximates mixture of normals
    X_cbrt = X**(1/3)
    
    # Fit gaussian mixture to X_cbrt
    gm = GaussianMixture(ncomp, random_state=random_state)
    fit = gm.fit(X_cbrt)
    
    # Compute posterior probabilities of each data point
    z = fit.predict_proba(X_cbrt)

    # Computed alpha and beta from weighted MOM estimators of X
    # Weights are the posterior probabilities z
    X_mean = X.T @ z / np.sum(z, axis=0)
    X_var = np.diag(z.T@(X - X_mean)**2)*n/((n-1)*np.sum(z, axis=0))
    w = np.mean(z, axis=0)
    alpha = (X_mean**2/X_var).reshape(ncomp,)
    beta = (X_var/X_mean).reshape(ncomp,)
    
    return w, alpha, beta

# Estimate parameters of gamma mixture fit
def gamma_mixture(X, ncomp=2, random_state=None, tol=1e-3, verbose=False):
    
    if ncomp == 1:
        # Simply fit a single gamma
        alpha, _, beta = stats.gamma.fit(X, floc=0)
        w = 1        
        return w, alpha, beta
    
    N = np.size(X)
    X = X.reshape(N, 1)
    
    # Initialize alphas and betas
    w, alpha, beta = init_params(X, ncomp, random_state)

    # Keep track of previous parameter estimates
    w_old = np.repeat(np.Inf, ncomp)
    alpha_old = np.repeat(np.Inf, ncomp)
    beta_old = np.repeat(np.Inf, ncomp)
    
    i = 0  # iteration counter
    rmsd = np.Inf  # root-square difference between old and new estimates
    start_time = time.time()
    
    while rmsd > tol:
        w_old = w
        alpha_old = alpha
        beta_old = beta

        # Expectation step: calculate likelihoods
        gamma_mat = np.zeros((N, ncomp))
        for j in range(ncomp):
            gamma_mat[:,j,None] = w[j]*stats.gamma.pdf(X, a=alpha[j], scale=beta[j])

        # Normalize s.t. rowsums equal 1
        gamma_mat /= np.sum(gamma_mat, axis=1, keepdims=True)
        
        # Maximization step: update w, alpha, beta
        # Compute some arrays that have to be used multiple times
        sumN_gamma = np.sum(gamma_mat, axis=0)
        sumN_gammaX = np.sum(gamma_mat*X, axis=0)

        # Update weights
        w = sumN_gamma/N
        
        # Compute constant C for alpha root finding
        C = (np.log(sumN_gammaX/sumN_gamma) -
            np.sum(gamma_mat*np.log(X), axis=0)/sumN_gamma)
        
        # Update alpha using Halley's method
        alpha = optimize.newton(
            alpha_eq, x0=alpha, 
            fprime=alpha_eq_prime, fprime2=alpha_eq_prime2, 
            args=(C, )
        )

        # Update beta based on alpha
        beta = sumN_gammaX/(alpha*sumN_gamma)

        new_params = np.vstack((w, alpha, beta))
        old_params = np.vstack((w_old, alpha_old, beta_old))
        
        # Compute root mean-squared difference between old and new parameters
        rmsd = np.sqrt(np.sum(((new_params - old_params)/old_params)**2))
        
        i += 1
        
        if verbose:
            print(i, rmsd, time.time()-start_time)

    # Sort parameters by beta (larger beta = fatter tail)
    order = np.argsort(beta)
    w = np.array([w[i] for i in order])
    alpha = np.array([alpha[i] for i in order])
    beta = np.array([beta[i] for i in order])
    
    return w, alpha, beta


