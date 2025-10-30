import numpy as np


def check_threshold_exceed(data, threshold):
    for path, coordinates in data.items():
        
        y_values = [coord[1] for coord in coordinates]
        
        max_y = max(y_values)
        min_y = min(y_values)
        if max_y - min_y > threshold:
            return True  
    return False  


def extract_phase_from_paths(matrix, paths):
    results = {}  

    # new_phase_matrix = np.angle(matrix)
    for path_id, coordinates in paths.items():
        path_data = [matrix[row, col] for (row, col) in coordinates]

        first_row, first_col = coordinates[0]
        last_row, last_col = coordinates[-1]

        initial_adjacent_values = []
        if first_row > 0 and first_col > 0:
            initial_adjacent_values.append(matrix[first_row - 1, first_col - 1])
        if first_row > 0:
            initial_adjacent_values.append(matrix[first_row - 1, first_col])
        if first_row > 0 and first_col < matrix.shape[1] - 1:
            initial_adjacent_values.append(matrix[first_row - 1, first_col + 1])

        if initial_adjacent_values:
            initial_avg_value = sum(initial_adjacent_values) / len(initial_adjacent_values)
            path_data.insert(0, initial_avg_value)

        ending_adjacent_values = []
        if last_row < matrix.shape[0] - 1 and last_col > 0:
            ending_adjacent_values.append(matrix[last_row + 1, last_col - 1])
        if last_row < matrix.shape[0] - 1:
            ending_adjacent_values.append(matrix[last_row + 1, last_col])
        if last_row < matrix.shape[0] - 1 and last_col < matrix.shape[1] - 1:
            ending_adjacent_values.append(matrix[last_row + 1, last_col + 1])

        if ending_adjacent_values:
            ending_avg_value = sum(ending_adjacent_values) / len(ending_adjacent_values)
            path_data.append(ending_avg_value)

        results[path_id] = np.array(path_data)

    return results


def extract_data_from_paths(matrix, paths):
    results = {}  

    for path_id, coordinates in paths.items():
        path_data = [matrix[row, col] for (row, col) in coordinates]

        first_row, first_col = coordinates[0]
        last_row, last_col = coordinates[-1]

        initial_adjacent_values = []
        if first_row > 0 and first_col > 0:
            initial_adjacent_values.append(matrix[first_row - 1, first_col - 1])
        if first_row > 0:
            initial_adjacent_values.append(matrix[first_row - 1, first_col])
        if first_row > 0 and first_col < matrix.shape[1] - 1:
            initial_adjacent_values.append(matrix[first_row - 1, first_col + 1])

        if initial_adjacent_values:
            initial_avg_value = sum(initial_adjacent_values) / len(initial_adjacent_values)
            path_data.insert(0, initial_avg_value)

        ending_adjacent_values = []
        if last_row < matrix.shape[0] - 1 and last_col > 0:
            ending_adjacent_values.append(matrix[last_row + 1, last_col - 1])
        if last_row < matrix.shape[0] - 1:
            ending_adjacent_values.append(matrix[last_row + 1, last_col])
        if last_row < matrix.shape[0] - 1 and last_col < matrix.shape[1] - 1:
            ending_adjacent_values.append(matrix[last_row + 1, last_col + 1])

        if ending_adjacent_values:
            ending_avg_value = sum(ending_adjacent_values) / len(ending_adjacent_values)
            path_data.append(ending_avg_value)

        results[path_id] = np.array(path_data)

    return results
