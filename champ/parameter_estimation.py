import louvain
from math import log


def iterative_monolayer_resolution_parameter_estimation(G, gamma=1.0, tol=1e-2, max_iter=25, verbose=False):
    """
    Monolayer variant of ALG. 1 from "Relating modularity maximization and stochastic block models in multilayer
    networks." The nested functions here are just used to match the pseudocode in the paper.

    :param G: input graph
    :param gamma: starting gamma value
    :param tol: convergence tolerance
    :param max_iter: maximum number of iterations
    :param verbose: whether or not to print verbose output
    :return: gamma to which the iteration converged and the resulting partition
    """

    if 'weight' not in G.es:
        G.es['weight'] = [1.0] * G.ecount()
    m = sum(G.es['weight'])

    def maximize_modularity(resolution_param):
        # RBConfigurationVertexPartition implements sum (A_ij - gamma (k_ik_j)/(2m)) delta(sigma_i, sigma_j)
        # i.e. "standard" modularity with resolution parameter
        return louvain.find_partition(G, louvain.RBConfigurationVertexPartition, resolution_parameter=resolution_param,
                                      weights='weight')

    def estimate_SBM_parameters(partition):
        community = partition.membership
        m_in = sum(e['weight'] * (community[e.source] == community[e.target]) for e in G.es)
        kappa_r_list = [0] * len(partition)
        for e in G.es:
            kappa_r_list[community[e.source]] += e['weight']
            kappa_r_list[community[e.target]] += e['weight']
        sum_kappa_sqr = sum(x ** 2 for x in kappa_r_list)

        omega_in = (2 * m_in) / (sum_kappa_sqr / (2 * m))
        # guard for div by zero with single community partition
        omega_out = (2 * m - 2 * m_in) / (2 * m - sum_kappa_sqr / (2 * m)) if len(partition) > 1 else 0

        # return estimates for omega_in, omega_out (for multilayer, this would return theta_in, theta_out, p, K)
        return omega_in, omega_out

    def update_gamma(omega_in, omega_out):
        return (omega_in - omega_out) / (log(omega_in) - log(omega_out))

    part, last_gamma = None, None
    for iteration in range(max_iter):
        part = maximize_modularity(gamma)
        omega_in, omega_out = estimate_SBM_parameters(part)

        if omega_in == 0 or omega_in == 1 or omega_out == 0 or omega_in == 1:
            raise ValueError("gamma={:0.3f} resulted in degenerate partition".format(gamma))

        last_gamma = gamma
        gamma = update_gamma(omega_in, omega_out)

        if verbose:
            print("Iter {:>2}: {} communities with Q={:0.3f} and "
                  "gamma={:0.3f}->{:0.3f}".format(iteration, len(part), part.q, last_gamma, gamma))

        if abs(gamma - last_gamma) < tol:
            break  # gamma converged
    else:
        if verbose:
            print("Gamma failed to converge within {} iterations. "
                  "Final move of {:0.3f} was not within tolerance {}".format(max_iter, abs(gamma - last_gamma), tol))

    if verbose:
        print("Returned {} communities with Q={:0.3f} and gamma={:0.3f}".format(len(part), part.q, gamma))

    return gamma, part
