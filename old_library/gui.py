#!/usr/bin/env python3
'''
GUI utilities to be used with the optimisation module.
Qt GUIs as well as Jupyter/IPython and console interaction utilities
'''
# python 2 compatibility
from __future__ import (absolute_import, division, print_function, unicode_literals)
from .py2 import *

try:
    if PY_VERSION == 3:
        import PyQt5.QtCore as qtc
        import PyQt5.QtGui as qtg
        import PyQt5.QtWidgets as qt
        QT_VERSION = 5
    elif PY_VERSION == 2:
        # Qt5 is not available in python 2
        import PyQt4.QtCore as qtc
        import PyQt4.QtGui as qtg
        qt = qtg # bit of a hack but seems to work
        QT_VERSION = 4
except ImportError:
    print('failed to import Qt. The other GUI features will still work')
    # provide dummy placeholders for the Qt stuff used at the global scope (eg
    # for inheritance) Obviously attempting to _use_ those classes will fail
    class qt:
        QWidget   = object
        QTextEdit = object
    class qtc:
        QObject = object
        QThread = object
        @staticmethod
        def pyqtSignal(types, name=None):
            return None

import sys
import time
import signal
import threading
import io

from IPython.display import clear_output, display, Image, HTML
import ipywidgets as widgets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # prettify matplotlib

#TODO: confirm dangerous actions with a popup

# Local imports
from .core import DEFAULT_HOST, DEFAULT_PORT, Evaluator, Optimiser

def in_jupyter():
    ''' whether the current script is running in IPython/Jupyter '''
    try:
        __IPYTHON__
    except NameError:
        return False
    return True


if in_jupyter():
    print('Setting up GUI for Jupyter')
    # hide scroll bars that sometimes appear (by chance) because the image fills
    # the entire sub-area
    display(HTML('<style>div.output_subarea.output_png{'
                 'overflow:hidden;width:100%;max-width:100%}</style>'))

def pyqt_pdb():
    qtc.pyqtRemoveInputHook()
    import pdb
    pdb.set_trace()

class BetterQThread(qtc.QThread):
    '''
    construct a QThread using a similar API to the threading module
    '''
    def __init__(self, target, args=None):
        super().__init__()
        self.args = [] if args is None else args
        self.target = target
    def run(self):
        self.target(*self.args)

class ScrollTextEdit(qt.QTextEdit):
    '''
    a QTextEdit with helper functions to provide nice scrolling, ie only keep
    scrolling to the bottom as new text is added if the end of the text box was
    visible before the text was added
    '''
    def end_is_visible(self):
        bar = self.verticalScrollBar()
        return bar.maximum() - bar.value() < bar.singleStep()
    def scroll_to_end(self):
        bar = self.verticalScrollBar()
        bar.setValue(bar.maximum())

def format_matching(doc, search, fmt, whole_line=True):
    '''
    apply the given format to lines matching the search query

    doc: QTextDocument to search through
    search: string or QRegExp to search for in the document
    fmt: QTextCharFormat to apply to the matching ranges
    whole_line: whether to apply the formatting to the whole line or just
                the matching text
    '''
    hl = qtg.QTextCursor(doc)
    while not hl.isNull() and not hl.atEnd():
        hl = doc.find(search, hl)
        if not hl.isNull():
            # the text between the anchor and the position will be selected
            if whole_line:
                hl.movePosition(qtg.QTextCursor.StartOfLine)
                hl.movePosition(qtg.QTextCursor.EndOfLine, qtg.QTextCursor.KeepAnchor)
            # merge the char format with the current format
            hl.mergeCharFormat(fmt)

def format_regex_color(doc, regex, color, whole_line):
    '''
    a helper function for the most common case of format_matching where lines
    matching a case-insensitive regex string should be colored a certain color.
    regex: a string representing a regular expression
    color: a QColor object
    '''
    char_format = qtg.QTextCursor(doc).charFormat()
    char_format.setForeground(color)

    regex = qtc.QRegExp(regex)
    regex.setCaseSensitivity(qtc.Qt.CaseInsensitive)

    format_matching(doc, regex, char_format, whole_line=whole_line)

def start_GUI(e_or_o):
    '''
    start a GUI for the given evaluator or optimiser and wait for it to close
    '''
    app = qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    gui = None
    if isinstance(e_or_o, Evaluator):
        gui = EvaluatorGUI(e_or_o)
    elif isinstance(e_or_o, Optimiser):
        gui = OptimiserGUI(e_or_o)
    else:
        raise TypeError

    def handle_ctrl_c(*args):
        gui.close()
        app.quit()
    signal.signal(signal.SIGINT, handle_ctrl_c)

    sys.exit(app.exec_())


def indent(text, indentation=1):
    return '\n'.join('\t' * indentation + line for line in  text.splitlines())

class LoggableGUI(qt.QWidget):
    def __init__(self):
        super().__init__()

        self.label_font = qtg.QFont(qtg.QFont().defaultFamily(), 10)

        self.grid = qt.QGridLayout()
        self.grid.setSpacing(5)

        self.last_log_length = 0 # length of log string as of last update
        self.log = ScrollTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(qt.QTextEdit.NoWrap)
        if QT_VERSION == 5:
            # introduced in 5.2
            self.log.setFont(qtg.QFontDatabase.systemFont(qtg.QFontDatabase.FixedFont))
        else:
            f = qtg.QFont('Monospace')
            f.setStyleHint(qtg.QFont.TypeWriter)
            self.log.setFont(f)
        self.grid.addWidget(self.log, 0, 0)

        self.watcher = qtc.QTimer()
        self.watcher.setInterval(500) # ms
        self.watcher.timeout.connect(self.update_UI)
        self.watcher.start() # set the timer going

        # fit the sidebar layout inside a widget to control the width
        sidebarw = qt.QWidget()
        sidebarw.setFixedWidth(200)
        self.grid.addWidget(sidebarw, 0, 1)

        self.sidebar = qt.QVBoxLayout()
        sidebarw.setLayout(self.sidebar)

        self.setLayout(self.grid)
        self.resize(1024, 768)
        self._center()

        self.raw = qt.QTextEdit(self)
        self.raw.setWindowFlags(self.raw.windowFlags() | qtc.Qt.Window)
        self.raw.setReadOnly(True)
        self.raw.resize(1600, 900)
        self.raw.move(100, 100)

    def format_text(self):
        '''
        change the formatting on the text to highlight important words
        '''
        doc = self.log.document()

        format_regex_color(doc, r'(problem|error|exception|traceback)', qtc.Qt.red, whole_line=True)

        format_regex_color(doc, r'(warning|not allowing)', qtc.Qt.darkYellow, whole_line=True)

        format_regex_color(doc, r'(current best)', qtc.Qt.darkGreen, whole_line=True)

    def update_log(self, text):
        '''
        make the log edit area reflect the current text.
        Note: it is assumed that the log can only be appended to and not
              otherwise modified
        '''
        new_len = len(text)
        was_at_end = self.log.end_is_visible()
        if new_len > self.last_log_length:
            new_text = text[self.last_log_length:]

            # just using self.log.append() will add an implicit newline
            cur = qtg.QTextCursor(self.log.document())
            cur.movePosition(qtg.QTextCursor.End)
            cur.insertText(new_text)

            self.last_log_length = new_len

        if was_at_end:
            self.log.scroll_to_end()

        self.format_text()

    def _raw_string(self, dict_, exclude, expand):
        s = ''
        for k, v in sorted(dict_.items(), key=lambda x: x[0]):
            if k in exclude:
                s += '{}: skipped\n\n\n'.format(k)
            elif v is not None and k in expand and hasattr(k, '__dict__'):
                s += '{}:\n{}\n\n'.format(k, indent(self._raw_string(v.__dict__, [], [])))
            else:
                event_class = threading.Event if PY_VERSION == 3 else threading._Event
                if isinstance(v, event_class):
                    v = '{}: set={}'.format(type(v), v.is_set())
                elif isinstance(v, list):
                    list_str = '[\n' + ',\n'.join(['\t' + str(x) for x in v]) + '\n]'
                    v = 'len = {}\n{}'.format(len(v), list_str)
                s += '{}:\n{}\n\n'.format(k, v)
        return s

    def _show_raw(self, dict_, exclude, expand):
        s = self._raw_string(dict_, exclude, expand)
        self.raw.setText(s)
        self.raw.show()

    def _center(self):
        frame = self.frameGeometry()
        frame.moveCenter(qt.QDesktopWidget().availableGeometry().center())
        self.move(frame.topLeft())
    def _add_named_field(self, name, default_value, font=None, widget=None):
        ''' add a label and a line edit to the sidebar '''
        font = self.label_font if font is None else font
        label = qt.QLabel(name)
        label.setFont(font)
        self.sidebar.addWidget(label)
        if widget is None:
            widget = qt.QLineEdit(default_value)
        self.sidebar.addWidget(widget)
        return widget
    def _add_button(self, name, onclick):
        ''' add a button to the sidebar '''
        button = qt.QPushButton(name)
        button.clicked.connect(onclick)
        self.sidebar.addWidget(button)
        return button


class EvaluatorGUI(LoggableGUI):
    '''
    note: if the evaluator has an floating point attribute 'progress' which
    varies from 0.0 to 1.0 to signify the progress of the current operation,
    then the GUI will sync this value with a progress bar.
    '''
    def __init__(self, evaluator, name=None):
        super().__init__()
        assert isinstance(evaluator, Evaluator)

        self.evaluator = evaluator
        self.evaluator_thread = BetterQThread(target=self._run_evaluator)
        self.evaluator_thread.setTerminationEnabled(True)

        if name is None:
            self.setWindowTitle('Evaluator GUI')
        else:
            self.setWindowTitle('Evaluator GUI: {}'.format(name))
        self.name = name


        self.host = self._add_named_field('host:', DEFAULT_HOST)
        self.port = self._add_named_field('port:', str(DEFAULT_PORT))

        self._add_button('Start Client', self.start)
        self._add_button('Stop Client', self.stop)
        self._add_button('Force Stop', self.force_stop)

        self.info = qt.QLabel('')

        self.sidebar.addSpacing(20)

        self._add_button('Show Raw', self.show_raw)

        self.sidebar.addStretch() # fill remaining space

        # add a progress bar if the evaluator supports it
        if hasattr(evaluator, 'progress'):
            self.progress = qt.QProgressBar()
            self.progress.setRange(0, 100)
            self.grid.addWidget(self.progress, 1, 0)
        else:
            self.progress = None

        self.update_UI()
        self.log.scroll_to_end()

        self.show()

    def _run_evaluator(self):
        host = self.host.text()
        port = int(self.port.text())
        self.evaluator.run_client(host, port)

    def set_info(self, text):
        self.info.setText(text)
        print('{} info: {}'.format(self.name, text), flush=True)

    def update_UI(self):
        self.update_log(self.evaluator.log_record)
        if self.progress is not None:
            # progress bar only takes ints
            self.progress.setValue(self.evaluator.progress * 100)

    def start(self):
        if self.evaluator_thread.isRunning():
            self.set_info('evaluator already running')
        else:
            self.evaluator_thread.start()
    def stop(self):
        self.evaluator.stop()
    def force_stop(self):
        self.evaluator.stop()
        self.evaluator_thread.terminate()

    def show_raw(self):
        self._show_raw(self.evaluator.__dict__,
                       exclude=['log_record'],
                       expand=[])

    def closeEvent(self, event):
        if self.evaluator_thread.isRunning():
            self.set_info('shutting down')
            self.evaluator.stop()
            self.evaluator_thread.wait(5000) # ms
        self.raw.close()
        event.accept()


class OptimiserGUI(LoggableGUI):
    def __init__(self, optimiser, name=None):
        super().__init__()
        assert isinstance(optimiser, Optimiser)

        self.optimiser = optimiser
        self.optimiser_thread = BetterQThread(target=self._run_optimiser)
        self.optimiser_thread.setTerminationEnabled(True)

        if name is None:
            self.setWindowTitle('Optimiser GUI')
        else:
            self.setWindowTitle('Optimiser GUI: {}'.format(name))
        self.name = name

        self.host = self._add_named_field('host:', DEFAULT_HOST)
        self.port = self._add_named_field('port:', str(DEFAULT_PORT))
        self._add_button('Start Server', self.start)
        self._add_button('Stop Server', self.stop)
        self._add_button('Force Stop', self.force_stop)

        self.sidebar.addSpacing(20)

        self.checkpoint_filename = self._add_named_field(
            'filename:', optimiser.checkpoint_filename)

        self._add_button('Save Checkpoint', self.save_checkpoint)
        self._add_button('Cancel Checkpoint', self.cancel_checkpoint)
        self._add_button('Load Checkpoint', self.load_checkpoint)


        # auto-checkpoint
        self.auto_checkpoint_interval = qt.QSpinBox()
        self.auto_checkpoint_interval.setRange(1, 100)
        self.auto_checkpoint_interval.setValue(5)
        self.auto_checkpoint_interval.setToolTip(
            'remember, new jobs are prevented while taking a checkpoint, '
            'so best to balance being careful with being wasteful')
        self._add_named_field('Auto-Checkpoint Interval: ', None, widget=self.auto_checkpoint_interval)
        self.auto_checkpoint = qt.QCheckBox('Auto-Checkpoint')
        self.auto_checkpoint.stateChanged.connect(self.auto_checkpoint_changed)
        self.sidebar.addWidget(self.auto_checkpoint)
        self.last_auto_checkpoint = 0 # number of jobs
        self.auto_checkpoint_filename = './auto_checkpoint_{}.json'
        self.auto_checkpoint.setToolTip(
            'when checked, whenever <interval> new jobs are finished,\n'
            'the optimiser takes a checkpoint to "{}"\n'
            ''.format(self.auto_checkpoint_filename))

        self.sidebar.addSpacing(20)

        self._add_button('Clean Log', self.clean_log)

        self.info = qt.QLabel('')

        self.sidebar.addSpacing(20)

        self._add_button('Show Raw', self.show_raw)

        self.sidebar.addStretch() # fill remaining space

        self.update_UI()
        self.log.scroll_to_end()

        self.show()

    def _run_optimiser(self):
        host = self.host.text()
        port = int(self.port.text())
        self.optimiser.run_server(host, port)

    def set_info(self, text):
        self.info.setText(text)
        print('{} info: {}'.format(self.name, text), flush=True)

    def update_UI(self):
        self.update_log(self.optimiser.log_record)
        num_finished = self.optimiser.num_finished_jobs
        interval = self.auto_checkpoint_interval.value()
        if self.auto_checkpoint.isChecked() and num_finished > self.last_auto_checkpoint + interval:
            self.last_auto_checkpoint = num_finished
            self.optimiser.save_when_ready(self.auto_checkpoint_filename.format(num_finished))


    def start(self):
        if self.optimiser_thread.isRunning():
            self.set_info('optimiser already running')
        else:
            self.optimiser_thread.start()
    def stop(self):
        self.optimiser.stop()
    def force_stop(self):
        self.optimiser.stop()
        self.optimiser_thread.terminate()

    def show_raw(self):
        self._show_raw(self.optimiser.__dict__,
                       exclude=['log_record', 'samples', 'step_log'],
                       expand=[])

    def save_checkpoint(self):
        filename = self.checkpoint_filename.text()
        self.optimiser.save_when_ready(filename)
    def cancel_checkpoint(self):
        self.optimiser.cancel_save()
    def load_checkpoint(self):
        filename = self.checkpoint_filename.text()
        self.optimiser.stop()
        self.optimiser.load_checkpoint(filename)

    def auto_checkpoint_changed(self):
        if self.auto_checkpoint.isChecked():
            self.last_auto_checkpoint = self.optimiser.num_finished_jobs

    def clean_log(self):
        self.optimiser.clean_log()
        self.log.setText(self.optimiser.log_record)
        self.last_log_length = len(self.optimiser.log_record)

    def closeEvent(self, event):
        if self.optimiser_thread.isRunning():
            self.set_info('shutting down')
            self.optimiser.stop()
            self.optimiser_thread.wait(5000) # ms
        self.raw.close()
        event.accept()




class LogMonitor:
    '''
    asynchronously monitor the log of the given loggable object and output it to
    the given file.
    return a flag which will stop the monitor when set
    '''
    counter = 0
    def __init__(self, loggable, f):
        '''
        f: a filename or file object (eg open(..., 'w') or sys.stdout)
        '''
        self.stop_flag = threading.Event()
        self.loggable = loggable
        if LogMonitor.counter > 0:
            f += str(LogMonitor.counter)
        self.f = open(f, 'w') if is_string(f) else f
        LogMonitor.counter += 1
    def listen_async(self):
        t = threading.Thread(target=self.listen, name='LogMonitor')
        # don't wait the threads to finish, just kill it when the program exits
        t.setDaemon(True)
        t.start()
    def listen(self):
        self.stop_flag.clear()
        amount_written = 0
        while not self.stop_flag.is_set():
            length = len(self.loggable.log_record)
            if length > amount_written:
                more = self.loggable.log_record[amount_written:length]
                self.f.write(more)
                self.f.flush()
                amount_written = length
        self.f.close()

def interactive(loggable, run_task, log_filename=None, poll_interval=0.5):
    '''
    run a task related to a loggable object (eg Optimiser server or Evaluator
    client) in the background while printing its output, catching any
    KeyboardInterrupts and shutting down the loggable object gracefully when one
    occurs.

    Evaluator example:
    >>> evaluator = MyEvaluator()
    >>> task = lambda: evaluator.run_client(host, port)
    >>> op.gui.interactive(evaluator, task, '/tmp/evaluator.log')

    Optimiser example
    >>> optimiser = op.GridSearchOptimiser(ranges)
    >>> task = lambda: optimiser.run_server(host, port, max_jobs=20)
    >>> op.gui.interactive(optimiser, task, '/tmp/optimiser.log')

    loggable: an object with a log_record attribute and stop() method
    run_task: a function to run (related to the loggable object),
        eg lambda: optimiser.run_sequential(my_evaluator)
    log_filename: filename to write the log to (recommend somewhere in /tmp/) or
        None to not write.
    poll_interval: time to sleep between checking if the log has been updated
    '''
    thread = threading.Thread(target=run_task, name='interactive')
    thread.start()
    f = None
    amount_printed = 0

    def print_more():
        ''' print everything that has been added to the log since amount_printed '''
        length = len(loggable.log_record)
        if length > amount_printed:
            more = loggable.log_record[amount_printed:length]
            print(more, end='')
            if f is not None:
                f.write(more)
                f.flush()
        return length

    try:
        f = open(log_filename, 'w') if log_filename is not None else None
        try:
            while thread.is_alive():
                amount_printed = print_more()
                time.sleep(poll_interval)
        except KeyboardInterrupt:
            print('-- KeyboardInterrupt caught, stopping gracefully --')
            loggable.stop()
            thread.join()

        # finish off anything left not printed
        print_more()
        print('-- interactive task finished -- ')
    finally:
        if f is not None:
            f.close()


def optimiser_progress_bar(optimiser, close_when_complete=False):
    '''
    display a progress bar in Jupyter as the optimiser runs, once the maximum
    number of jobs has been reached, stop watching.
    close_when_complete: whether to leave the progress bar in place or delete
        it once the optimiser finishes.
    '''
    def watch():
        while optimiser.run_state is None:
            time.sleep(0.2)

        label = widgets.HTML()
        bar = widgets.IntProgress(min=0, max=optimiser.run_state.max_jobs,
                                  value=0, description='', layout=widgets.Layout(width='100%'),
                                  bar_style='info')
        box = widgets.VBox(children=[label, bar])
        display(box)

        while optimiser.run_state is not None:
            bar.value = optimiser.run_state.finished_this_run
            label.value = 'Finished Jobs: {}/{}'.format(bar.value, bar.max)
            time.sleep(0.2)

        if close_when_complete:
            box.close()
        else:
            bar.value = bar.max
            label.value = 'Finished Jobs: {}/{}'.format(bar.value, bar.max)
            bar.bar_style = 'success'

    t = threading.Thread(target=watch)
    t.setDaemon(True)
    t.start()



class DebugGUIs(qtc.QObject):
    '''
    a debug/helper class for quickly spawning GUIs for the given optimisers and
    evaluators which run in a separate thread.
    Example usage:
    >>> guis = op_gui.DebugGUIs([optimiser1, optimiser2], evaluator)
    >>> # do some stuff
    >>> guis.stop()

    '''

    # signal must be defined at the class for subclass of QObject, but still
    # access through self.add_signal
    add_signal = qtc.pyqtSignal(object, name='add_signal')

    def __init__(self, optimisers, evaluators):
        '''
        optimisers: may be either a list or a single optimiser
        evaluators: may be either a list or a single evaluator
        '''
        super().__init__()
        self.optimisers = self._ensure_list(optimisers)
        self.evaluators = self._ensure_list(evaluators)
        self.guis = []
        # counters for naming
        self.op_counter = 1
        self.ev_counter = 1
        self.ready = threading.Event()
        self.app_thread = threading.Thread(target=self._start, name='DebugGUIs')
        self.app_thread.setDaemon(True)
        self.app_thread.start()
        self.ready.wait()

    def _ensure_list(self, x):
        ''' return x if x is iterable, otherwise return [x] '''
        try:
            iter(x)
            return x
        except TypeError:
            return [x]

    def _start(self):
        self.app = qt.QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(True)

        # must be queued to be used from non-Qt threads
        self.add_signal.connect(self._add, qtc.Qt.QueuedConnection)

        for o in self.optimisers:
            assert isinstance(o, Optimiser)
            self._add(o)

        for e in self.evaluators:
            assert isinstance(e, Evaluator)
            self._add(e)

        self.ready.set()
        self.app.exec_()

    def _add(self, e_or_o):
        '''
        e_or_o: an Evaluator or Optimiser to spawn a GUI for
        '''
        if isinstance(e_or_o, Optimiser):
            self.guis.append(OptimiserGUI(e_or_o, str(self.op_counter)))
            print('spawning optimiser GUI {}'.format(self.op_counter))
            self.op_counter += 1
        elif isinstance(e_or_o, Evaluator):
            self.guis.append(EvaluatorGUI(e_or_o, str(self.ev_counter)))
            print('spawning evaluator GUI {}'.format(self.ev_counter))
            self.ev_counter += 1
        else:
            raise TypeError

    def add(self, e_or_o):
        # NOTE: this doesn't actually work. The window opens but does not update
        print('signalling to add')
        self.add_signal.emit(e_or_o)
        # forces Qt to process the event
        qtc.QEventLoop().processEvents()

    def stop(self):
        for g in self.guis:
            g.close()
        self.t.exit()
        self.app.quit()
        self.app_thread.join()
    def wait(self):
        print('waiting for GUIs to manually close', flush=True)
        self.app_thread.join()


def step_log_slider(optimiser, function, pre_compute=False):
    '''
    A utility function for easily using a slider to select a step from the optimiser's step log
    function: a function which takes the step number and step as arguments and returns a figure
        (if None is returned and then the last matplotlib figure is displayed)
    pre_compute: whether to plot each figure once at the start to make scrubbing
        faster. If pre_compute is False then this function will still memoise (store
        images when they are requested for the first time)
    '''
    # step numbers, may not be contiguous  or in order
    step_nums = sorted(optimiser.step_log.keys())

    # dictionary of step number to image of the figure for that step
    saved = {}
    def save_fig(s, fig):
        img = io.BytesIO()
        # default dpi is 72
        fig.savefig(img, format='png', bbox_inches='tight')
        saved[s] = Image(data=img.getvalue(), format='png', width='100%')

    def show_step(s):
        if s not in saved:
            # if function returns None then use the current figure
            fig = function(s, optimiser.step_log[s]) or plt.gcf()
            save_fig(s, fig) # memoise
            plt.close(fig) # free resources
        display(saved[s])

    if pre_compute:
        # plot each step (displaying the output) and save each figure
        for s in step_nums:
            clear_output(wait=True)
            show_step(s)
        clear_output()

    return list_slider(step_nums, show_step, slider_name='Step N: ')

def list_slider(list_, function, slider_name='Item N: '):
    '''
    A utility for easily setting up a Jupyter slider for items of a list
    list_: a list of items for the slider value to correspond to
    function: a function which takes an item of list_ as an argument
    slider_name: the description/label to apply to the slider
    '''
    slider = widgets.IntSlider(value=len(list_), min=1, max=len(list_),
                               continuous_update=False, layout=widgets.Layout(width='100%'))
    slider.description = slider_name
    widgets.interact(lambda val: function(list_[val-1]), val=slider)
    return slider

