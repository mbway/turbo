#!/usr/bin/env python3
''' GUI utilities for use with Jupyter/IPython '''

try:
    from IPython import get_ipython
    from IPython.core.debugger import Pdb
    from IPython.display import clear_output, display, Image, HTML, Javascript
    import ipywidgets as widgets
except ImportError:
    pass # Jupyter not installed

import io
import matplotlib.pyplot as plt
import threading
import time

# local imports
import turbo.modules as tm
from turbo.gui.utils import in_jupyter


if in_jupyter():
    print('Setting up GUI for Jupyter')
    # hide scroll bars that sometimes appear (apparently by chance) because the
    # image fills the entire sub-area.
    display(HTML('<style>div.output_subarea.output_png{'
                 'overflow:hidden;width:100%;max-width:100%}</style>'))



def jupyter_set_width(width):
    '''set the width of the central element of the notebook

    Args:
        width (str): a css string to set the width to (eg "123px" or "90%" etc)
    '''
    display(Javascript('document.getElementById("notebook-container").style.width = "{}"'.format(width)))

def jupyter_set_trace():
    '''
    Note: don't use this for code running directly in a cell, wrap the contents
        of the cell in a function first

    Note: once out of this function put a breakpoint on the line after it then
        continue to reach it, otherwise stepping with next leads deep into some
        system code which you don't care about.
    '''
    Pdb.set_trace()

def using_svg_backend():
    '''
    using the following magic, Jupyter can be instructed to render matplotlib to
    svg for displaying. This has the advantage of more crisp images, however
    when rendering complex plots (such as a 2D heatmap) the resulting plot can
    be fatally large (eg >20MB) so make sure the svg backend isn't used with
    complex plots.

    `%config InlineBackend.figure_format = 'svg'`

    Note:
        this detection method may not detect a global setting of this
        configuration option instead of through a magic. I haven't tested it.
    '''
    return get_ipython().config['InlineBackend']['figure_format'] == 'svg'

class LabelledProgressBar(widgets.IntProgress):
    def __init__(self, min, max, initial_value, label_prefix):
        super().__init__(min=min, max=max, value=initial_value, description='',
                         layout=widgets.Layout(width='100%'), bar_style='info')
        self.label = widgets.HTML()
        self.label_prefix = label_prefix
        self.box = widgets.VBox(children=[self.label, self])
        display(self.box)

    def set_value(self, new_value):
        self.value = new_value
        self.update_label()
        if self.value == self.max:
            self.bar_style = 'success'
    def increment(self):
        self.set_value(self.value+1)
    def update_label(self):
        self.label.value = self.label_prefix + '{}/{}'.format(
            self.value-self.min, self.max-self.min)
    def close(self):
        self.box.close()


class OptimiserProgressBar(tm.Listener):
    def __init__(self, optimiser, close_when_complete=False, single_run=True):
        '''
        display a progress bar in Jupyter as the optimiser runs, once the maximum
        number of iterations has been reached, stop watching.

        Args:
            optimiser (turbo.Optimiser): the optimiser to listen to
            close_when_complete: whether to leave the progress bar in place or
                delete it once the optimiser finishes.
            single_run (bool): whether to unregister the listener after a single
                run, or keep listening.
        '''
        assert in_jupyter(), 'not running in Jupyter'
        self.close_when_complete = close_when_complete
        self.single_run = single_run
        self.opt = optimiser
        self.opt.register_listener(self)
    def run_started(self, finished_trials, max_trials):
        this_run = max_trials - finished_trials
        self.bar = LabelledProgressBar(0, this_run, initial_value=0,
                                       label_prefix='Finished Trials: ')
    def eval_finished(self, trial_num, y, eval_info):
        self.bar.increment()
    def run_finished(self):
        if self.close_when_complete:
            self.bar.close()
        if self.single_run:
            self.opt.unregister_listener(self)



def figure_to_Image(fig):
    ''' save a matplotlib `Figure` as a Jupyter `Image` '''
    img = io.BytesIO()
    # default dpi is 72
    fig.savefig(img, format='png', bbox_inches='tight', dpi=150)
    return Image(data=img.getvalue(), format='png', width='100%')

class PlotMemoization:
    '''A utility for memoising the results of a plotting function.

    This is useful when generating graphs interactively by changing some
    parameter, this class allows a more responsive UI if the particular
    parameter values have been encountered before.

    '''

    def __init__(self, plot):
        '''
        Args:
            plot (args -> matplotlib.figure.Figure): a function which takes some
                parameters and optionally returns the plotted figure for
                memoisation. If the function does not return a figure then the
                current figure will be memoised instead.
        '''
        # key : Image
        self.saved = {}
        self.plot = plot

    def show(self, key, args):
        '''Show `plot(**args)` and store with the given key, displaying from memory if already computed

        Args:
            key: a value to store the plot against
            args (dict): arguments to `self.plot`
        '''
        if key not in self.saved:
            # if function returns None then use the current figure
            fig = self.plot(**args) or plt.gcf()
            self.saved[key] = figure_to_Image(fig) # memoise
            plt.close(fig) # free resources
        display(self.saved[key])

    def pre_compute(keys, args_list):
        '''compute the plots for the given keys in one go so that they are all
        stored in memory from then on.

        Args:
            keys: a list of keys to pre-compute
            args_list: a list of dictionaries corresponding to `keys`
        '''
        bar = LabelledProgressBar(0, len(keys), initial_value=0)
        # plot each step (displaying the output) and save each figure
        for key, args in zip(keys, args_list):
            clear_output(wait=True)
            display(bar)
            self.show(key, args)
            bar.increment()
        bar.close()
        clear_output()

    def clear(self):
        ''' delete the memoised plots '''
        self.saved.clear()


def slider(values, function=None, description=None, initial=0):
    '''A utility for easily setting up a Jupyter slider for items of a list

    Args:
        values (int or list): either a list of items for the slider value to
            correspond to, or an integer, in which case the values will be the
            range from 0 to `values`
        function: (optional) a function which takes an item of `values` as an argument
        description (str): the description/label to apply to the slider
        initial (int): the index of the initial value
    '''
    if isinstance(values, int):
        values = range(0, values)
    # allow negative indices e.g. values[-1]
    if initial < 0:
        initial = len(values)-initial
    # 100% width causes a scrollbar to appear since it is too wide for the output container
    slider = widgets.IntSlider(value=initial, min=0, max=len(values)-1,
                continuous_update=False, layout=widgets.Layout(width='98%'))
    slider.description = description
    if function is not None:
        widgets.interact(lambda i: function(values[i]), i=slider)
    return slider

def dropdown(values, function=None, description=None, initial=0):
    '''A utility for easily setting up a Jupyer dropdown with the given values

    Args:
        values: a list of the possible dropdown values
        function: (optional) a function which takes the current selected value as an argument
        description (str): the label to give to the dropdown
        initial (Int): an index into the values list to give initially
    '''
    dropdown = widgets.Dropdown(
        options=values,
        value=values[initial],
        description=description
    )
    if function is not None:
        widgets.interact(lambda val: function(val), val=dropdown)
    return dropdown

