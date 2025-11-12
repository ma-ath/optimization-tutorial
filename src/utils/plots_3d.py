import numpy as np
from typing import Optional
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

from src.functions._function import Function


def plot_surface(function: Function, resolution: int = 200, title: Optional[str] = None):
    """
    Plot a static 3D surface of a 2D function f([x, y]).
    """
    # Create meshgrid
    x = np.linspace(*function.function_domain, resolution)
    y = np.linspace(*function.function_domain, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda a, b: function(np.array([a, b])))(X, Y)

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.85, linewidth=0)

    # Labels and styling
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('f(x₁, x₂)')
    ax.set_title(title or getattr(function, "name", "Function Surface"))
    # fig.colorbar(surf, shrink=0.5, aspect=10, label='f(x₁, x₂)')
    plt.show()


def plot_surface_with_path(function: Function, x_history: np.ndarray, resolution: int = 200, title: Optional[str] = None):
    """
    Plot a 3D surface with an optimization path over it.
    x_history: np.ndarray of shape (n_steps, 2)
    """
    x_history = np.array(x_history)
    assert x_history.ndim == 2 and x_history.shape[1] == 2, \
        "x_history must have shape (n_steps, 2)"

    # Compute surface
    x = np.linspace(*function.function_domain, resolution)
    y = np.linspace(*function.function_domain, resolution)
    X, Y = np.meshgrid(x, y)
    Z = np.vectorize(lambda a, b: function(np.array([a, b])))(X, Y)

    # Compute path
    Z_path = np.array([function(x) for x in x_history])

    # Plot
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)
    ax.plot(x_history[:, 0], x_history[:, 1], Z_path, color='red', lw=2, label='Optimization path')

    # Start and end points
    ax.scatter(x_history[0, 0], x_history[0, 1], Z_path[0], color='blue', s=60, label='Start')
    ax.scatter(x_history[-1, 0], x_history[-1, 1], Z_path[-1], color='green', s=60, label='End')

    # Labels and styling
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('f(x₁, x₂)')
    ax.set_title(title or getattr(function, "name", "Function Surface with Path"))
    # fig.colorbar(surf, shrink=0.5, aspect=10, label='f(x₁, x₂)')
    ax.legend()
    plt.show()


def animate_path_on_surface(
    function: Function,
    *x_histories: np.ndarray,
    resolution: int = 100,
    interval: int = 100,
    title: Optional[str] = None,
):
    """
    Animate one or multiple optimization paths or populations on a 3D surface.

    Each individual (or single optimizer) leaves a red trajectory trail.
    """
    if len(x_histories) == 1 and isinstance(x_histories[0], (list, np.ndarray)):
        x_histories = (x_histories[0],)

    # Surface grid
    x_min, x_max = function.function_domain
    y_min, y_max = function.function_domain
    X, Y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                       np.linspace(y_min, y_max, resolution))
    Z = np.vectorize(lambda a, b: function(np.array([a, b])))(X, Y)

    # Figure setup
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, cmap="viridis", alpha=0.8, linewidth=0)

    # Precompute trajectories and function values
    histories_data = []
    for hist in x_histories:
        arr = np.array(hist)
        if arr.ndim == 2:  # single trajectory
            z_vals = np.array([function(x) for x in arr])
            histories_data.append({"type": "path", "x": arr, "z": z_vals})
        elif arr.ndim == 3:  # population
            n_frames, pop_size, _ = arr.shape
            z_vals = np.array([[function(arr[f, i]) for i in range(pop_size)] for f in range(n_frames)])
            histories_data.append({"type": "population", "x": arr, "z": z_vals})
        else:
            raise ValueError("Each x_history must have shape (n_frames, 2) or (n_frames, population_size, 2)")

    # Initialize artists
    scatters, all_lines = [], []
    for h in histories_data:
        if h["type"] == "path":
            # thick bright red line + black moving dot
            (line,) = ax.plot([], [], [], color="red", lw=3)
            scat = ax.scatter([], [], [], color="black", s=60, zorder=10)
            scatters.append(scat)
            all_lines.append([line])
        elif h["type"] == "population":
            n_frames, pop_size, _ = h["x"].shape
            # slightly transparent red trails for each individual
            lines = [
                ax.plot([], [], [], color="red", alpha=0.5, lw=1.5, zorder=5)[0]
                for _ in range(pop_size)
            ]
            scat = ax.scatter([], [], [], color="black", s=25, zorder=10)
            scatters.append(scat)
            all_lines.append(lines)

    # Configure axes
    ax.set_xlabel("x₁")
    ax.set_ylabel("x₂")
    ax.set_zlabel("f(x₁, x₂)")
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(np.min(Z), np.max(Z))
    ax.set_title(title or getattr(function, "name", "Optimization Surface"))

    n_frames = max(h["x"].shape[0] for h in histories_data)

    # Animation update
    def update(frame):
        for h, lines, scat in zip(histories_data, all_lines, scatters):
            arr, z_vals = h["x"], h["z"]
            if h["type"] == "path":
                if frame < len(arr):
                    lines[0].set_data(arr[:frame, 0], arr[:frame, 1])
                    lines[0].set_3d_properties(z_vals[:frame])
                    scat._offsets3d = (
                        [arr[frame, 0]],
                        [arr[frame, 1]],
                        [z_vals[frame]],
                    )
            elif h["type"] == "population":
                if frame < len(arr):
                    xs = arr[frame, :, 0]
                    ys = arr[frame, :, 1]
                    zs = z_vals[frame, :]
                    scat._offsets3d = (xs, ys, zs)
                    for i, line in enumerate(lines):
                        line.set_data(arr[:frame + 1, i, 0], arr[:frame + 1, i, 1])
                        line.set_3d_properties(z_vals[:frame + 1, i])
        return sum(all_lines, []) + scatters

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)
    return HTML(anim.to_jshtml())


def animate_population_on_surface(function: Function, population_history: np.ndarray, resolution=100, interval=200, title=None):
    """
    Animate a population of points moving on a 3D function surface.

    Parameters
    ----------
    function : callable
        A function f([x, y]) -> float, with attribute `function_domain` (tuple) and optionally `name`.
    population_history : np.ndarray, shape (n_frames, population_size, 2)
        Population positions at each frame.
    resolution : int
        Grid resolution for surface plot.
    interval : int
        Time in milliseconds between frames.
    title : str, optional
        Figure title.

    Returns
    -------
    HTML
        An IPython HTML object displaying the animation.
    """
    population_history = np.array(population_history)
    n_frames, population_size, dim = population_history.shape
    assert dim == 2, "population_history must have shape (n_frames, population_size, 2)"

    # Compute surface
    x_min, x_max = function.function_domain
    y_min, y_max = function.function_domain
    X, Y = np.meshgrid(np.linspace(x_min, x_max, resolution),
                       np.linspace(y_min, y_max, resolution))
    Z = np.vectorize(lambda a, b: function(np.array([a, b])))(X, Y)

    # Evaluate population Z-values at each frame
    # Shape: (n_frames, population_size)
    Z_population = np.array([[function(population_history[f, i]) for i in range(population_size)]
                             for f in range(n_frames)])

    # Create figure
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection='3d')

    # Surface
    ax.plot_surface(X, Y, Z, cmap='viridis', alpha=0.8, linewidth=0)

    # Initialize scatter for population
    scatter = ax.scatter([], [], [], color='red', s=40)

    # Labels
    ax.set_xlabel('x₁')
    ax.set_ylabel('x₂')
    ax.set_zlabel('f(x₁, x₂)')
    ax.set_title(title or getattr(function, 'name', 'Population on Surface'))

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_zlim(np.min(Z), np.max(Z))

    # Animation update function
    def update(frame):
        xs = np.asarray(population_history[frame, :, 0]).ravel()
        ys = np.asarray(population_history[frame, :, 1]).ravel()
        zs = np.asarray(Z_population[frame, :]).ravel()
        scatter._offsets3d = (xs, ys, zs)
        ax.set_title(f"{title or getattr(function, 'name', 'Population on Surface')}")
        return scatter,

    anim = FuncAnimation(fig, update, frames=n_frames, interval=interval, blit=False)
    plt.close(fig)  # Prevent static figure from displaying
    return HTML(anim.to_jshtml())
