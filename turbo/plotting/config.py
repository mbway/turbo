
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
    colors = [trial_marker_colors[t.selection_info['type']] for n, t in trials]
    edge_colors = [trial_edge_colors[t.selection_info['type']] for n, t in trials]
    if i_index is not None:
        colors[i_index] = 'none'
        edge_colors[i_index] = 'none'

    # s = size in points**2
    ax.scatter(xs, ys, s=16, marker='o', color=colors, zorder=zorder, linewidth=0.5, edgecolors=edge_colors)

    if i_index is not None:
        ax.scatter(xs[i_index], ys[i_index], s=80, marker='*',
                   color=trial_marker_colors['incumbent'], zorder=zorder,
                   linewidth=0.5, edgecolors=trial_edge_colors['incumbent'])
