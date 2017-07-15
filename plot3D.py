import plotly.offline as py
py.init_notebook_mode(connected=False) # connected = whether to use CDN
import plotly.graph_objs as go
import matplotlib.pyplot as plt
import IPython.html.widgets as widgets

def surface3D(x, y, z):
    '''
    parameters should be of the form:
    X = np.arange(...)
    Y = np.arange(...)
    X, Y = np.meshgrid(X, Y)
    Z = f(X,Y)
    '''
    data = [go.Surface(x=x, y=y, z=z, colorscale='Viridis')]
    layout = go.Layout(
        title='3D surface',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = go.Figure(data=data, layout=layout)
    py.iplot(fig)

def scatter3D(x,y,z, interactive=False, color='z'):
    '''
        color can be one of: 'z', 'age'
            'age' colors by the index (so the age of the sample)
    '''
    # from https://plot.ly/python/sliders/
    x,y,z = x.flatten(), y.flatten(), z.flatten()
    num_samples = len(x)

    if color == 'z':
        color = z
        scale = 'Viridis'
    elif color == 'age':
        color = list(reversed(range(num_samples)))
        scale = 'Blues'

    data = [go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=3,
            color=color,
            colorscale=scale,
            opacity=0.8
        )
    )]
    layout = go.Layout(
        title='3D Scatter Plot',
        autosize=False,
        width=900,
        height=600,
        margin=dict(l=0, r=0, b=0, t=0)
    )
    fig = go.Figure(data=data, layout=layout)

    if interactive:
        def plot(val):
            fig['data'][0].update(x=x[:val], y=y[:val], z=z[:val])
            py.iplot(fig)

        # continuous_update = if the slider moves from 0 to 100, then call the
        # update function with every value from 0 to 100
        slider = widgets.IntSlider(value=num_samples, min=0, max=num_samples,
                                continuous_update=False, width='100%')
        slider.description = 'Samples'
        widgets.interact(plot, val=slider)
    else:
        py.iplot(fig)

def plotFromMPL():
    '''
        take the current matplotlib plot and instead render it with plotly
    '''
    fig = plt.gcf() # Get current figure
    py.iplot_mpl(fig)
