import numpy as np
from matplotlib import pyplot as plt


def plot_paths(matrix, paths):
    matrix = matrix.T
    rows, cols = matrix.shape

    fig, ax = plt.subplots(figsize=(cols, rows))

    heatmap = ax.imshow(matrix, cmap="coolwarm", origin="upper", aspect='auto')

    plt.colorbar(heatmap, ax=ax)

    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    for idx, (path_id, path) in enumerate(paths.items()):
        path_rows, path_cols = zip(*path)

        ax.plot(
            path_rows, path_cols, color=colors[idx % len(colors)], marker='o', label=path_id, linestyle='-', linewidth=2
        )

    ax.set_xlabel('Slow Time(frame)', fontsize=20)
    ax.set_ylabel('Fast Time(bin)', fontsize=20)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    plt.legend()

    plt.show()


def plot_paths_solo(matrix, paths):
    matrix = matrix.T
    rows, cols = matrix.shape

    fig, ax = plt.subplots(figsize=(cols, rows))

    ax.imshow(matrix, cmap="coolwarm", origin="upper", aspect='auto')

    specified_path_id = 'path_1'

    if specified_path_id in paths:
        path = paths[specified_path_id]

        path_rows, path_cols = zip(*path)

        ax.plot(
            path_rows, path_cols, marker='o', label=specified_path_id, linestyle='-', linewidth=2
        )

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    # ax.grid(which="minor", color="black", linestyle='-', linewidth=0.5)
    ax.tick_params(which="minor", size=0)

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)
    plt.legend()

    plt.show()
