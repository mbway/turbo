#!/usr/bin/env python3
'''
Utilities for plotting 3D graphs with plotly.js and matplotlib
'''

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D # for matplotlib 3D plots

# Plotly imports
try:
    import plotly.offline as ply
    import plotly.graph_objs as go
except ImportError:
    # plotly not installed
    ply = None

# local imports
from turbo.gui.utils import in_jupyter
from .config import Config


if in_jupyter() and ply is not None:
    print('Setting up plotly for Jupyter')
    # connected = whether to use CDN or load offline from the version stored in
    # the python module.
    ply.init_notebook_mode(connected=True)

    import ipywidgets as widgets


def surface_3D(x, y, z, tooltips=None, axes_names=['x','y','z'], log_axes=(False,False,False)):
    '''plot a 3D surface using plotly

    Parameters should be of the form:
    ```
    X = np.arange(...)
    Y = np.arange(...)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    ```

    Args:
        tooltips: an array with the same length as the number of points,
            containing a string to display beside them
        log_axes: whether the `x,y,z` axes should have a logarithmic scale
            (False => linear)
    '''
    data = [go.Surface(
        x=x, y=y, z=z,
        text=tooltips, colorscale='Viridis', opacity=1
    )]
    layout = go.Layout(
        title='3D surface',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=axes_names[0], type='log' if log_axes[0] else None),
            yaxis=dict(title=axes_names[1], type='log' if log_axes[1] else None),
            zaxis=dict(title=axes_names[2], type='log' if log_axes[2] else None),
        )
    )
    fig = go.Figure(data=data, layout=layout)
    # show_link is a link to export to the 'plotly cloud'
    ply.iplot(fig, show_link=False)


def scatter_3D(x,y,z, interactive=False, color_by='z', markersize=2,
               tooltips=None, axes_names=['x','y','z'],
               log_axes=(False,False,False)):
    '''
    Args:
        interactive: whether to display an interactive slider to choose how many
            points to display
        color_by: can be one of: 'z', 'age'.
            'age' colors by the index (so the age of the sample)
        tooltips: an array with the same length as the number of points,
            containing a string to display beside them
        log_axes: whether the `x,y,z` axes should have a logarithmic scale
            (False => linear)
    '''
    # from https://plot.ly/python/sliders/
    x,y,z = x.flatten(), y.flatten(), z.flatten()
    num_samples = len(x)

    if color_by == 'z':
        color_by = z
        scale = 'Viridis'
    elif color_by == 'age':
        color_by = list(reversed(range(num_samples)))
        scale = 'Blues'

    data = [go.Scatter3d(
        x=x, y=y, z=z,
        text=tooltips,
        mode='markers',
        opacity=0.9,
        marker=dict(
            size=markersize,
            color=color_by,
            colorscale=scale,
            opacity=0.8
        )
    )]
    layout = go.Layout(
        title='3D Scatter Plot',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0),
        scene=dict(
            xaxis=dict(title=axes_names[0], type='log' if log_axes[0] else None),
            yaxis=dict(title=axes_names[1], type='log' if log_axes[1] else None),
            zaxis=dict(title=axes_names[2], type='log' if log_axes[2] else None),
        )
    )
    fig = go.Figure(data=data, layout=layout)

    if interactive:
        def plot(val):
            fig['data'][0].update(x=x[:val], y=y[:val], z=z[:val])
            ply.iplot(fig, show_link=False)

        # continuous_update = if the slider moves from 0 to 100, then call the
        # update function with every value from 0 to 100
        slider = widgets.IntSlider(value=num_samples, min=0, max=num_samples,
                                continuous_update=False, width='100%')
        slider.description = 'first n points: '
        widgets.interact(plot, val=slider)
    else:
        ply.iplot(fig, show_link=False)

def surface_3D_MPL(X, Y, Z):
    '''plot a 3D surface using matplotlib

    Parameters should be of the form:
    ```
    X = np.arange(...)
    Y = np.arange(...)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    ```

    note: use `%matplotlib tk` to open 3D plots interactively
    '''
    fig = plt.figure(figsize=Config.fig_sizes['3D'])
    ax = fig.gca(projection='3d')
    surf = ax.plot_surface(X, Y, Z, cmap='plasma', linewidth=0, antialiased=True)
    fig.colorbar(surf)
    plt.show()
