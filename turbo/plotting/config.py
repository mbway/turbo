
class Config:

    def fig_sizes_by_width(width):
        return {
            '1D_trial' : (width, width*0.6), # for the 1D trial plot
            '2D_trial' : (width, width), # for the 4 heatmap 2D trial plot
            'overview' : (width, width/2),  # for plots that show something about each trial over the course of the run, with trial number as the x axis
            '2D_surrogate' : (width, width*0.8), # for the surrogate hyperparameter 2D scatter plot
            '3D' : (width, width), # for 3D plots
        }

    # sizes are in inches
    # since using the svg backend results in different sizes, 12 inches is a
    # nice balance where the svg fills the entire Jupyter width, while the
    # bitmap versions are slightly smaller, but neither are bad.
    fig_sizes = fig_sizes_by_width(12)


    trial_marker_colors = {
        'pre_phase' : 'violet',
        'bayes'     : '#4c72b0',
        'fallback'  : 'red',

        'incumbent' : 'deepskyblue'
    }
    trial_edge_colors = {
        'pre_phase' : 'black',
        'bayes'     : '#333355',
        'fallback'  : '#111111',

        'incumbent' : 'black'
    }


def scatter_trials(ax, rec, xs, ys, trials, zorder=10):
    '''draw a scatter plot using the trials colored by type

    xs, ys are the points to scatter, they correspond to trials, which is a list of (trial_num, Recorder.Trial)
    '''
    i_n, _ = rec.get_incumbent()
    # find the first index into trials where the trial number matches that of the incumbent, or None if not found
    i_index = next((i for i, (n, t) in enumerate(trials) if n == i_n), None)
    colors = [Config.trial_marker_colors[t.selection_info['type']] for n, t in trials]
    edge_colors = [Config.trial_edge_colors[t.selection_info['type']] for n, t in trials]
    if i_index is not None:
        colors[i_index] = 'none'
        edge_colors[i_index] = 'none'

    # s = size in points**2
    ax.scatter(xs, ys, s=16, marker='o', color=colors, zorder=zorder, linewidth=0.5, edgecolors=edge_colors)

    if i_index is not None:
        ax.scatter(xs[i_index], ys[i_index], s=80, marker='*',
                   color=Config.trial_marker_colors['incumbent'], zorder=zorder,
                   linewidth=0.5, edgecolors=Config.trial_edge_colors['incumbent'])
