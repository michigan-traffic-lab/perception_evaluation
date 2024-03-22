from .utils import nested_dict_to_matrix
from scipy.optimize import linear_sum_assignment

def hungarian_matching(matching_scores, minimize=False):
    # Convert the nested dictionary to a cost matrix
    matrix, col_keys, row_keys = nested_dict_to_matrix(matching_scores)

    # Convert scores to costs
    if not minimize:
        cost_matrix = matrix.max() - matrix
    else:
        cost_matrix = matrix

    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the total score for the optimal assignment
    total_score = matrix[row_ind, col_ind].sum()

    # Map the indices back to the original keys
    # col_keys = list(matching_scores.keys())  # Outer keys as column labels
    # row_keys = list(matching_scores[col_keys[0]].keys())  # Inner keys as row labels

    # Create the matching result using original keys
    result = {col: row for col, row in zip([col_keys[j] for j in col_ind], [row_keys[i] for i in row_ind])}

    return result, total_score
