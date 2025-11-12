import numpy as np
import plotly.graph_objects as go


from src.functions._function import Function


def plot_surface(function: Function, resolution: int = 200):
    """
    Plot a 3D surface of a given 2D function f([x, y]).
    """
    # Create meshgrid
    x = np.linspace(*function.search_domain, resolution)
    y = np.linspace(*function.search_domain, resolution)
    X, Y = np.meshgrid(x, y)

    # Evaluate function on grid
    Z = np.array([[function(np.array([xi, yi])) for xi, yi in zip(row_x, row_y)]
                  for row_x, row_y in zip(X, Y)])

    # Create Plotly surface
    fig = go.Figure(data=[go.Surface(
        x=X, y=Y, z=Z,
        colorscale='Viridis',
        opacity=0.85,
        name='Surface'
    )])

    # Layout styling
    fig.update_layout(
        title=f'{function.name} Surface',
        scene=dict(
            xaxis_title='x₁',
            yaxis_title='x₂',
            zaxis_title='f(x₁, x₂)',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2))
        ),
        width=900,
        height=700,
        template='plotly_white'
    )
    return fig


def plot_surface_and_path(function: Function, x_history, resolution: int = 200, title: str = "Function Surface with Optimization Path"):
    """
    Plot a 3D surface and overlay the optimization path.
    x_history: list or np.ndarray of shape (n_steps, 2)
    """
    x_history = np.array(x_history)
    Z_path = np.array([function(x) for x in x_history])

    # Create base surface
    fig = plot_surface(function, resolution=resolution)

    # Add the optimization trajectory
    fig.add_trace(go.Scatter3d(
        x=x_history[:, 0],
        y=x_history[:, 1],
        z=Z_path,
        mode='lines+markers',
        line=dict(color='red', width=5),
        marker=dict(size=4, color='red'),
        name='Optimization Path'
    ))

    # Add start and end points
    fig.add_trace(go.Scatter3d(
        x=[x_history[0, 0]], y=[x_history[0, 1]], z=[Z_path[0]],
        mode='markers',
        marker=dict(color='blue', size=8, symbol='diamond'),
        name='Start'
    ))
    fig.add_trace(go.Scatter3d(
        x=[x_history[-1, 0]], y=[x_history[-1, 1]], z=[Z_path[-1]],
        mode='markers',
        marker=dict(color='green', size=8, symbol='diamond'),
        name='Optimum'
    ))

    fig.update_layout(title=title)
    return fig
