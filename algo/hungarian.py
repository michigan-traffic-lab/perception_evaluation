from .utils import nested_dict_to_matrix
from scipy.optimize import linear_sum_assignment

def hungarian_matching(matching_scores):
    # Convert the nested dictionary to a cost matrix
    matrix = nested_dict_to_matrix(matching_scores)

    # Convert scores to costs
    cost_matrix = matrix.max() - matrix

    # Apply the Hungarian algorithm to find the optimal assignment
    row_ind, col_ind = linear_sum_assignment(cost_matrix)

    # Calculate the total score for the optimal assignment
    total_score = matrix[row_ind, col_ind].sum()

    # Map the indices back to the original keys
    row_keys = list(matching_scores.values())[0].keys()
    col_keys = matching_scores.keys()

    # Create the matching result using original keys
    result = {col: row for row, col in zip([list(row_keys)[i] for i in row_ind], [list(col_keys)[j] for j in col_ind])}

    return result, total_score
