#!/usr/bin/env python3
''' GUI utilities for use with Jupyter/IPython '''

try:
    from IPython.display import clear_output, display, Image, HTML
    import ipywidgets as widgets
except ImportError:
    pass # Jupyter not installed

import io
import matplotlib.pyplot as plt
import threading
import time

# local imports
import turbo.modules as tm


def in_jupyter():
    ''' whether the current script is running in IPython/Jupyter '''
    try:
        __IPYTHON__
    except NameError:
        return False
    return True

if in_jupyter():
    print('Setting up GUI for Jupyter')
    # hide scroll bars that sometimes appear (apparently by chance) because the
    # image fills the entire sub-area.
    display(HTML('<style>div.output_subarea.output_png{'
                 'overflow:hidden;width:100%;max-width:100%}</style>'))

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
    def eval_finished(self, trial_num, y):
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
    fig.savefig(img, format='png', bbox_inches='tight')
    return Image(data=img.getvalue(), format='png', width='100%')

def range_plot_slider(range_, plot, pre_compute=False, slider_name=''):
    '''
    Display an integer slider which takes the given values, with `plot()` being
    run every time the value changes.

    If plot returns a figure then the result can be memoised so that the plot
    does not have to be recalculated if the slider falls on the same value twice.

    Args:
        range\_: either a `range` or an `integer` (in which case the range will
            be from 0 to `range_` exclusive)
        plot (int -> matplotlib.figure.Figure): a function which takes the
            current slider value and optionally returns the plotted figure for
            memoisation. If the function does not return a figure then the
            current figure will be memoised instead.
    '''
    if isinstance(range_, int):
        range_ = range(0, range_)

    saved = {} # iteration : Image

    def slider_changed(n):
        if n not in saved:
            # if function returns None then use the current figure
            fig = plot(n) or plt.gcf()
            saved[n] = figure_to_Image(fig) # memoise
            plt.close(fig) # free resources
        display(saved[n])

    if pre_compute:
        bar = LabelledProgressBar(0, range_[-1], initial_value=0)
        # plot each step (displaying the output) and save each figure
        for n in range_:
            clear_output(wait=True)
            display(bar)
            slider_changed(n)
            bar.increment()
        bar.close()
        clear_output()

    return list_slider(range_, slider_changed, slider_name=slider_name)

def list_slider(list_, function, slider_name='Item N: '):
    '''A utility for easily setting up a Jupyter slider for items of a list

    Args:
        list\_: a list of items for the slider value to correspond to
        function: a function which takes an item of `list_` as an argument
        slider_name: the description/label to apply to the slider
    '''
    slider = widgets.IntSlider(value=len(list_), min=1, max=len(list_),
                continuous_update=False, layout=widgets.Layout(width='100%'))
    slider.description = slider_name
    widgets.interact(lambda val: function(list_[val-1]), val=slider)
    return slider

