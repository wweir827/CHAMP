import louvain
from math import log


def iterative_resolution_parameter_estimation(G, gamma=1.0, tol=1e-2, max_iter=25, debug=False):
    """
    Monolayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." The nested functions here are just used to match the pseudocode in the paper.

    :param G: input graph
    :param gamma: starting gamma value
    :param tol: convergence tolerance
    :param max_iter: maximum number of iterations
    :param debug: whether or not to print debug output
    :return: gamma to which the iteration converged and the resulting partition
    """

    def maximize_modularity(resolution_param):
        # RBConfigurationVertexPartition implements sum (A_ij - gamma (k_ik_j)/(2m)) delta(sigma_i, sigma_j)
        # i.e. "standard" modularity with resolution parameter
        return louvain.find_partition(G, louvain.RBConfigurationVertexPartition, resolution_parameter=resolution_param)

    def estimate_SBM_parameters(partition):
        m = G.ecount()
        community = partition.membership
        m_in = sum(community[e.source] == community[e.target] for e in G.es)
        kappa_r_list = [0] * len(partition)
        for v, c in enumerate(community):
            kappa_r_list[c] += partition.graph.degree(v)
        sum_kappa_sqr = sum(x ** 2 for x in kappa_r_list)

        omega_in = (2 * m_in) / (sum_kappa_sqr / (2 * m))
        # guard for div by zero with single community partition
        omega_out = (2 * m - 2 * m_in) / (2 * m - sum_kappa_sqr / (2 * m)) if len(partition) > 1 else 0

        # return estimates for omega_in, omega_out (for multilayer, this would return theta_in, theta_out, p, K)
        return omega_in, omega_out

    def update_gamma(omega_in, omega_out):
        if omega_out == 0:
            return omega_in / log(omega_in)
        return (omega_in - omega_out) / (log(omega_in) - log(omega_out))

    part, last_gamma = None, None
    for iteration in range(max_iter):
        part = maximize_modularity(gamma)
        omega_in, omega_out = estimate_SBM_parameters(part)

        if omega_in == 0 or omega_in == 1:
            # not clear how to deal with this case -- perhaps return None
            raise ValueError("gamma={:0.3f} resulted in degenerate partition".format(gamma))

        last_gamma = gamma
        gamma = update_gamma(omega_in, omega_out)

        if debug:
            warn_degenerate = "(degenerate)" if gamma is None else ""
            print("Iter {:>2}: {} communities with Q={:0.3f} and "
                  "gamma={:0.3f}->{:0.3f} {}".format(iteration, len(part), part.q, last_gamma, gamma, warn_degenerate))

        if abs(gamma - last_gamma) < tol:
            break  # gamma converged
    else:
        if debug:
            print("Gamma failed to converge within {} iterations. "
                  "Final move of {:0.3f} was not within tolerance {}".format(max_iter, abs(gamma - last_gamma), tol))

    if debug:
        print("Returned {} communities with Q={:0.3f} and gamma={:0.3f}".format(len(part), part.q, gamma))

    return gamma, part
