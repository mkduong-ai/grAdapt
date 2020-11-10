import numpy as np


def pairwise_distances(p, q):
    first_term = np.einsum('ij->i', p ** 2)  # sum along dimension
    second_term = np.einsum('ij->i', q ** 2)  # sum along dimension
    third_term = np.einsum('ik,jk->ij', p, q)  # pairwise product
    squared_distances = first_term[:, None] + second_term
    squared_distances -= 2 * third_term

    # Ignore numerical error
    squared_distances[squared_distances < 1e-10] = 0

    return np.sqrt(squared_distances)