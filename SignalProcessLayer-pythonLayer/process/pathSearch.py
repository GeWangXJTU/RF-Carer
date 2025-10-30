import numpy as np
from scipy.signal import savgol_filter


def dfs_search_window(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    paths = {}
    path_id = 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.2:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 1)
                row_end = min(rows, next_r + 2)
                col_start_left = max(0, c - 1)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 2)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.2:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 3)
                row_end = min(rows, next_r + 4)
                col_start_left = max(0, c - 3)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 1)
                col_end_mid = min(cols, c + 2)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 4)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.2:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 5)
                col_start_left = max(0, c - 6)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 1)
                col_end_mid = min(cols, c + 2)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 7)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.2:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 5)
                row_end = min(rows, next_r + 6)
                col_start_left = max(0, c - 5)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 2)
                col_end_mid = min(cols, c + 3)
                col_start_right = max(0, c)
                col_end_right = min(cols, c + 5)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.2:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 5)
                row_end = min(rows, next_r + 6)
                col_start_left = max(0, c - 5)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 2)
                col_end_mid = min(cols, c + 3)
                col_start_right = max(0, c)
                col_end_right = min(cols, c + 5)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    return paths


def dfs_search_window_1(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    paths = {}
    path_id = 1

    for start_col in range(0, cols):
        if np.sum(matrix[0, start_col]) > 0:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 1)
                col_start_left = max(0, c - 1)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 2)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0, start_col]) > 0:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 1)
                col_start_left = max(0, c - 2)
                col_end_left = min(cols, c - 1)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 2)
                col_end_right = min(cols, c + 3)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:7, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 1)
                col_start_left = max(0, c - 3)
                col_end_left = min(cols, c - 2)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 3)
                col_end_right = min(cols, c + 4)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0, start_col]) > 0:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 1)
                col_start_left = max(0, c - 4)
                col_end_left = min(cols, c - 3)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 4)
                col_end_right = min(cols, c + 5)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0, start_col]) > 0:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 1)
                col_start_left = max(0, c - 5)
                col_end_left = min(cols, c - 4)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 5)
                col_end_right = min(cols, c + 6)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    return paths


def dfs_search_window_2(matrix):
    rows, cols = matrix.shape[0], matrix.shape[1]
    paths = {}
    path_id = 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 1)
                row_end = min(rows, next_r + 2)
                col_start_left = max(0, c - 1)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c)
                col_end_mid = min(cols, c + 1)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 2)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 3)
                row_end = min(rows, next_r + 4)
                col_start_left = max(0, c - 3)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 1)
                col_end_mid = min(cols, c + 2)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 4)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r)
                row_end = min(rows, next_r + 5)
                col_start_left = max(0, c - 6)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 1)
                col_end_mid = min(cols, c + 2)
                col_start_right = max(0, c + 1)
                col_end_right = min(cols, c + 7)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1

    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 5)
                row_end = min(rows, next_r + 6)
                col_start_left = max(0, c - 5)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 2)
                col_end_mid = min(cols, c + 3)
                col_start_right = max(0, c)
                col_end_right = min(cols, c + 5)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    for start_col in range(0, cols):
        if np.sum(matrix[0:5, start_col]) >= 0.05:
            path = []
            r, c = 1, start_col

            while r < rows - 1:
                path.append((r, c))
                next_r = r + 1
                if next_r >= rows - 1:
                    break

                row_start = max(0, next_r - 5)
                row_end = min(rows, next_r + 6)
                col_start_left = max(0, c - 5)
                col_end_left = min(cols, c)
                col_start_mid = max(0, c - 2)
                col_end_mid = min(cols, c + 3)
                col_start_right = max(0, c)
                col_end_right = min(cols, c + 5)

                candidates = []
                if c > 0:
                    left_sum = np.mean(matrix[row_start:row_end, col_start_left:col_end_left])
                    candidates.append((left_sum, next_r, c - 1))
                mid_sum = np.mean(matrix[row_start:row_end, col_start_mid:col_end_mid])
                candidates.append((mid_sum, next_r, c))
                if c < cols - 1:
                    right_sum = np.mean(matrix[row_start:row_end, col_start_right:col_end_right])
                    candidates.append((right_sum, next_r, c + 1))

                max_sum, new_r, new_c = max(candidates, key=lambda x: x[0])

                if max_sum == 0:
                    break

                r, c = new_r, new_c

            if path:
                if len(path) == rows - 2:
                    paths[f'path_{path_id}'] = path
                    path_id += 1
    return paths


def segmentCreate(y_coords):
    segments = []
    current_segment = [y_coords[0]]

    for i in range(1, len(y_coords)):
        if abs(y_coords[i] - y_coords[i - 1]) < 5:
            current_segment.append(y_coords[i])
        else:
            segments.append(current_segment)
            current_segment = [y_coords[i]]

    segments.append(current_segment)
    return segments


def segmentMerge(segments):
    i = 1
    while i < len(segments):
        segment = segments[i]
        if len(segment) == 1:
            prev_segment = segments[i - 1]
            if abs(segment[0] - prev_segment[-1]) > 3:
                prev_segment.append(prev_segment[-1])
                segments.pop(i)
                continue
            else:
                prev_segment.append(segment[0])
                segments.pop(i)
                continue
        i += 1
    i = 1
    i = 1
    while i < len(segments):
        segment = segments[i]
        if len(segment) == 2:
            prev_segment = segments[i - 1]
            next_segment = segments[i + 1] if i + 1 < len(segments) else None

            if next_segment is not None:
                if abs(segment[0] - prev_segment[-1]) > 3:
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    segments.pop(i)
                    continue
                else:
                    prev_segment.extend(segment)
                    segments.pop(i)
                    continue
        i += 1
    i = 1
    while i < len(segments):
        segment = segments[i]
        if len(segment) == 3:
            prev_segment = segments[i - 1]
            next_segment = segments[i + 1] if i + 1 < len(segments) else None
            if next_segment is not None:
                if abs(segment[0] - prev_segment[-1]) > 3:
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    segments.pop(i)
                    continue
                else:
                    prev_segment.extend(segment)
                    segments.pop(i)
                    continue
        i += 1
    i = 1
    while i < len(segments):
        segment = segments[i]
        if len(segment) == 4:
            prev_segment = segments[i - 1]
            next_segment = segments[i + 1] if i + 1 < len(segments) else None
            if next_segment is not None:
                if abs(segment[0] - prev_segment[-1]) > 3:
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    prev_segment.append(prev_segment[-1])
                    segments.pop(i)
                    continue
                else:
                    prev_segment.extend(segment)
                    segments.pop(i)
                    continue
        i += 1
    return segments


def eraseMask(coordinates_to_zero, result_mask_5):
    for x, y in coordinates_to_zero:
        if 0 <= x < result_mask_5.shape[0] and 0 <= y < result_mask_5.shape[1]:
            x = int(x)
            y = int(y)
            y_range = range(y - 10, y + 10)

            for y_offset in y_range:
                if 0 <= y_offset < len(result_mask_5[0]):
                    if result_mask_5[x, y_offset] < 0.75:
                        result_mask_5[x, y_offset] = 0

    return result_mask_5


def fillCoords(result, x_coords, y_smoothed_sg):
    # x_coords=list(x_coords)
    i = 0
    while i < len(result):
        target = result[i]
        greater_than_target = [x for x in x_coords if x > target]
        less_than_target = [x for x in x_coords if x < target]

        if greater_than_target:
            closest_greater_value = min(greater_than_target, key=lambda x: x - target)
            closest_greater_index = x_coords.index(closest_greater_value)
        else:
            closest_greater_value = None
            closest_greater_index = None

        if less_than_target:
            closest_less_value = min(less_than_target, key=lambda x: target - x)
            closest_less_index = x_coords.index(closest_less_value)
        else:
            closest_less_value = None
            closest_less_index = None
        if closest_less_value is not None and closest_greater_value is not None:
            gap = np.arange(closest_less_value + 1, closest_greater_value)

            random_numbers = np.linspace(y_smoothed_sg[closest_less_index], y_smoothed_sg[closest_greater_index],
                                         num=len(gap) + 2)[1:-1]
            y_smoothed_sg = np.insert(y_smoothed_sg, closest_less_index + 1, random_numbers)
            x_coords[closest_less_index + 1:closest_less_index + 1] = gap
            i = i + len(gap)
        elif closest_less_value is not None and closest_greater_value is None:
            gap = np.arange(closest_less_value + 1, 2000)  # 1200
            random_numbers = np.full(len(gap), y_smoothed_sg[closest_less_index])
            y_smoothed_sg = np.append(y_smoothed_sg, random_numbers)
            x_coords.extend(gap)
            i = i + len(gap)
        elif closest_less_value is None and closest_greater_value is not None:
            gap = np.arange(0, closest_greater_value)
            random_numbers = np.full(len(gap), y_smoothed_sg[closest_greater_index])
            y_smoothed_sg = np.insert(y_smoothed_sg, 0, random_numbers)
            x_coords[0:0] = gap
            i = i + len(gap)
    y_smoothed_sg = np.round(y_smoothed_sg)
    return x_coords, y_smoothed_sg


def savFilter(y_coords):
    window_length = 83  # 取值为奇数且不能超过len(x)。它越大，则平滑效果越明显；越小，则更贴近原始曲线。
    polyorder = 5  # 多项式阶数,它越小，则平滑效果越明显；越大，则更贴近原始曲线。4,5,6
    y_smoothed_sg = savgol_filter(y_coords, window_length, polyorder)
    return y_smoothed_sg


def meanCoords(y_coords):
    indices = []

    # for i in range(len(y_coords)):
    #     start = max(0, i - 9)
    #     end = min(len(y_coords), i + 10)
    #
    #     previous_points = y_coords[start:i]
    #     next_points = y_coords[i + 1:end]
    #
    #     average_value = (sum(previous_points) + sum(next_points)) / (len(previous_points) + len(next_points))
    #
    #     if abs(y_coords[i] - average_value) > 5:
    #         indices.append(i)
    return indices


def dfs_search_window_3(matrix, threshold1, threshold2):
    print("search path start")
    rows, cols = matrix.shape[0], matrix.shape[1]
    paths = {}
    path_id = 1
    missflag = True
    print("while start")
    # while missflag is True:
    path = []
    miss = []
    print("for start")
    for i, row in enumerate(matrix):
        found = False
        for j, value in enumerate(row):
            if value > threshold1:
                path.append((i, j))
                found = True
                break
        if not found:
            miss.append(i)
    print("for end")
    print(len(miss))
    print(missflag)
    if len(miss) > threshold2 * rows:
        missflag = False
    print(missflag)
    if path and missflag is True:
        x_coords = np.array([coord[0] for coord in path])
        y_coords = np.array([coord[1] for coord in path])
        print("mean start")
        indices = meanCoords(y_coords)
        print("mean end")
        x_coords = [x_coords[i] for i in range(len(x_coords)) if i not in indices]
        y_coords = [y_coords[i] for i in range(len(y_coords)) if i not in indices]
        print("sav start")
        y_smoothed_sg = savFilter(y_coords)
        print("sav end")
        # y_smoothed_sg = y_coords
        x_man = list(range(2000))
        set_x_coords = set(x_coords)

        x_kong = []

        for index, value in enumerate(x_man):
            if value not in set_x_coords:
                x_kong.append(index)
        print("fill start")
        x_coords_1, y_smoothed_sg_1 = fillCoords(x_kong, x_coords, y_smoothed_sg)
        print("fill end")
        y_smoothed_sg_1 = y_smoothed_sg_1.astype(int).tolist()
        coordinates = list(zip(x_coords, y_smoothed_sg_1))
        paths[f'path_{path_id}'] = coordinates
        path_id = path_id + 1
        print("erase start")
        matrix = eraseMask(coordinates, matrix)
        print("erase end")
    print("search path end")
    return paths
