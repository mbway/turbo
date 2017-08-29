#!/usr/bin/env python3
'''
GUI utilities to be used with the optimisation module.
Qt GUIs as well as Jupyter/IPython and console interaction utilities
'''
from __future__ import print_function

import sys
if sys.version_info[0] == 3: # python 3
    import PyQt5.QtCore as qtc
    import PyQt5.QtGui as qtg
    import PyQt5.QtWidgets as qt
elif sys.version_info[0] == 2: # python 2
    # Qt5 is not available in python 2
    import PyQt4.QtCore as qtc
    import PyQt4.QtGui as qt
    import PyQt4.QtGui as qtg
else:
    print('unsupported python version')

import signal
import threading
import io

from IPython.display import clear_output, display, Image, HTML
import ipywidgets as widgets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns # prettify matplotlib


# Local imports
import optimisation as op

def is_ipython():
    ''' whether the current script is running in IPython/Jupyter '''
    try:
        __IPYTHON__
    except NameError:
        return False
    return True

class BetterQThread(qtc.QThread):
    '''
    construct a QThread using a similar API to the threading module
    '''
    def __init__(self, target, args=None):
        super(BetterQThread, self).__init__()
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

def start_GUI(e_or_o):
    '''
    start a GUI for the given evaluator or optimiser and wait for it to close
    '''
    app = qt.QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(True)
    gui = None
    if isinstance(e_or_o, op.Evaluator):
        gui = EvaluatorGUI(e_or_o)
    elif isinstance(e_or_o, op.Optimiser):
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
        super(LoggableGUI, self).__init__()

        self.label_font = qtg.QFont(qtg.QFont().defaultFamily(), 10)

        grid = qt.QGridLayout()
        grid.setSpacing(5)

        self.last_log_length = 0 # length of log string as of last update
        self.log = ScrollTextEdit()
        self.log.setReadOnly(True)
        self.log.setLineWrapMode(qt.QTextEdit.NoWrap)
        self.log.setFont(qtg.QFontDatabase.systemFont(qtg.QFontDatabase.FixedFont))
        grid.addWidget(self.log, 0, 0)

        self.watcher = qtc.QTimer()
        self.watcher.setInterval(500) # ms
        self.watcher.timeout.connect(self.update_UI)
        self.watcher.start() # set the timer going

        # fit the sidebar layout inside a widget to control the width
        sidebarw = qt.QWidget()
        sidebarw.setFixedWidth(200)
        grid.addWidget(sidebarw, 0, 1)

        self.sidebar = qt.QVBoxLayout()
        sidebarw.setLayout(self.sidebar)

        self.setLayout(grid)
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
        color = qtg.QTextCursor(doc).charFormat()
        color.setForeground(qtc.Qt.red)

        problems = qtc.QRegExp(r'(problem|error|exception|traceback)')
        problems.setCaseSensitivity(qtc.Qt.CaseInsensitive)

        format_matching(doc, problems, color, whole_line=True)

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
            elif v is not None and k in expand:
                s += '{}:\n{}\n\n'.format(k, indent(self._raw_string(v.__dict__, [], [])))
            else:
                if isinstance(v, threading.Event):
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
    def _add_named_field(self, name, default_value, font=None):
        ''' add a label and a line edit to the sidebar '''
        font = self.label_font if font is None else font
        label = qt.QLabel(name)
        label.setFont(font)
        self.sidebar.addWidget(label)
        edit = qt.QLineEdit(default_value)
        self.sidebar.addWidget(edit)
        return edit
    def _add_button(self, name, onclick):
        ''' add a button to the sidebar '''
        button = qt.QPushButton(name)
        button.clicked.connect(onclick)
        self.sidebar.addWidget(button)
        return button


class EvaluatorGUI(LoggableGUI):
    def __init__(self, evaluator, name=None):
        super(EvaluatorGUI, self).__init__()
        assert isinstance(evaluator, op.Evaluator)

        self.evaluator = evaluator
        self.evaluator_thread = BetterQThread(target=self._run_evaluator)
        self.evaluator_thread.setTerminationEnabled(True)

        if name is None:
            self.setWindowTitle('Evaluator GUI')
        else:
            self.setWindowTitle('Evaluator GUI: {}'.format(name))
        self.name = name


        self.host = self._add_named_field('host:', op.DEFAULT_HOST)
        self.port = self._add_named_field('port:', str(op.DEFAULT_PORT))

        self._add_button('Start Client', self.start)
        self._add_button('Stop Client', self.stop)
        self._add_button('Force Stop', self.force_stop)

        self.info = qt.QLabel('')

        self.sidebar.addSpacing(20)

        self._add_button('Show Raw', self.show_raw)

        self.sidebar.addStretch() # fill remaining space

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
                       expand=['run_state'])

    def closeEvent(self, event):
        if self.evaluator_thread.isRunning():
            self.set_info('shutting down')
            self.evaluator.stop()
            self.evaluator_thread.wait(time=5000) # ms
        self.raw.close()
        event.accept()


class OptimiserGUI(LoggableGUI):
    def __init__(self, optimiser, name=None):
        super(OptimiserGUI, self).__init__()
        assert isinstance(optimiser, op.Optimiser)

        self.optimiser = optimiser
        self.optimiser_thread = BetterQThread(target=self._run_optimiser)
        self.optimiser_thread.setTerminationEnabled(True)

        if name is None:
            self.setWindowTitle('Optimiser GUI')
        else:
            self.setWindowTitle('Optimiser GUI: {}'.format(name))
        self.name = name

        self.host = self._add_named_field('host:', op.DEFAULT_HOST)
        self.port = self._add_named_field('port:', str(op.DEFAULT_PORT))
        self._add_button('Start Server', self.start)
        self._add_button('Stop Server', self.stop)
        self._add_button('Force Stop', self.force_stop)

        self.sidebar.addSpacing(20)

        self.checkpoint_filename = self._add_named_field(
            'filename:', optimiser.checkpoint_filename)

        self._add_button('Save Checkpoint', self.save_checkpoint)
        self._add_button('Load Checkpoint', self.load_checkpoint)

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
                       expand=['run_state'])

    def save_checkpoint(self):
        filename = self.checkpoint_filename.text()
        self.optimiser.save_when_ready(filename)
    def load_checkpoint(self):
        filename = self.checkpoint_filename.text()
        self.optimiser.stop()
        self.optimiser.load_checkpoint(filename)

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
        self.f = open(f, 'w') if isinstance(f, str) else f
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
    >>> op.interactive(evaluator, task, '/tmp/evaluator.log')

    Optimiser example
    >>> optimiser = op.GridSearchOptimiser(ranges)
    >>> task = lambda: optimiser.run_server(host, port, max_jobs=20)
    >>> op.interactive(optimiser, task, '/tmp/optimiser.log')

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
        super(DebugGUIs, self).__init__()
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
            assert isinstance(o, op.Optimiser)
            self._add(o)

        for e in self.evaluators:
            assert isinstance(e, op.Evaluator)
            self._add(e)

        self.ready.set()
        self.app.exec_()

    def _add(self, e_or_o):
        '''
        e_or_o: an Evaluator or Optimiser to spawn a GUI for
        '''
        if isinstance(e_or_o, op.Optimiser):
            self.guis.append(OptimiserGUI(e_or_o, str(self.op_counter)))
            print('spawning optimiser GUI {}'.format(self.op_counter))
            self.op_counter += 1
        elif isinstance(e_or_o, op.Evaluator):
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
        if s not in saved.keys():
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
                               continuous_update=False, width='100%')
    slider.description = slider_name
    widgets.interact(lambda val: function(list_[val-1]), val=slider)
    return slider


def Jupyter_setup():
    print('setting up GUI for Jupyter')
    # hide scroll bars that sometimes appear (by chance) because the image fills
    # the entire sub-area
    display(HTML('''<style>div.output_subarea.output_png{overflow:hidden;}</style>'''))

if is_ipython():
    Jupyter_setup()


def main():
    '''
    example of the module usage
    '''

    ranges = {'a':[1,2], 'b':[3,4]}
    class TestEvaluator(op.Evaluator):
        def test_config(self, config):
            return config.a # placeholder cost function
    optimiser = op.GridSearchOptimiser(ranges, order=['a','b'])
    evaluator = TestEvaluator()

    app = qt.QApplication(sys.argv)

    op_gui = OptimiserGUI(optimiser)
    ev_gui = EvaluatorGUI(evaluator)

    def handle_ctrl_c(*args):
        op_gui.close()
        ev_gui.close()
        app.quit()
    signal.signal(signal.SIGINT, handle_ctrl_c)

    sys.exit(app.exec_())

if __name__ == '__main__':
    main()


