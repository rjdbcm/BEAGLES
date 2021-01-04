#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
sys.path.append(os.getcwd())
import cmd
import glob
import time
import select
import shutil
import signal
import warnings
try:
    import readline
except ImportError:
    warnings.warn('Could not load readline module.')
    readline = None
from subprocess import Popen, DEVNULL
from beagles.backend.net import TRAINERS
from beagles.base import *
from beagles.io.flags import SharedFlagIO


class CatchSignals:
    kill = False
    def __init__(self):
        signal.signal(signal.SIGINT, self.clean_exit)
        signal.signal(signal.SIGTERM, self.clean_exit)

    def clean_exit(self, signum, frame):
        self.kill = True

SHELL_HLINE = "{}".format('-'* shutil.get_terminal_size()[0])

class BeaglesShell(cmd.Cmd):
    intro = f'Welcome to the {APP_NAME} shell.\nType help or ? to list commands.\n'
    file = None
    processes = {}
    _current = None
    flags = Flags()
    io = SharedFlagIO(subprogram=False, flags=flags)
    prompt = f'{APP_NAME} >>> '
    completekey = 'tab'
    data_folders = glob.glob('./data/*', recursive=True) + glob.glob('./tests/*', recursive=True)
    # ----- helper methods -----

    def list_dir(self, path):
        return glob.glob(os.path.join(path, "*"))

    def draw_line(self):
        self.print(SHELL_HLINE, prefix='')

    def print(self, line, prefix='\t', suffix='\n'):
        self.stdout.write(prefix + line + suffix)

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
        self.do_term('all')
        self.io.cleanup_flags()
        exit(0)

    def progress(self, total=100.0):
        bar_form = '        Progress: {} {}%        \r'
        bar_len = shutil.get_terminal_size()[0] - len(bar_form) - 4
        self.reset_progress()
        filled_len = int(round(bar_len * self.flags.progress / float(total)))
        percents = round(100.0 * self.flags.progress / float(total), 1)
        bar = u"\u2593" * filled_len + u"\u2591" * (bar_len - filled_len)
        self.stdout.write(bar_form.format(bar, percents))
        self.reset_progress(0.0)
        sys.stdout.flush()
        self.stdout.flush()

    def reset_progress(self, default=100.0):
        self.io.io_flags()
        if self.flags.progress >= 100.0:
            self.flags.progress = default

    def poll_processes(self):
        _processes = self.processes.copy()
        [self.remove_process(k, p) for k, p in _processes.items()]

    def remove_process(self, name: str, process: Popen):
        if process.poll() is not None:
            proc = self.processes.pop(name)
            self.print(f'Removing finished process. Name: {name} PID: {proc.pid}', prefix='\n\t')

    def create_process(self, name: str, process: Popen):
        self.flags.pid = process.pid
        self.flags.project_name = name
        self.do_send()
        self.processes.update({name: process})

    # ----- overrides -----
    def preloop(self):
        self.flags.cli = True
        if readline and os.path.exists(SHELL_HISTORY_PATH):
            readline.read_history_file(SHELL_HISTORY_PATH)
        self._do_flag_methods()

    def postloop(self):
        if readline:
            readline.set_history_length(SHELL_HISTORY_SIZE)
            readline.write_history_file(SHELL_HISTORY_PATH)

    def emptyline(self):
        if self.lastcmd == 'watch':
            return
        if self.lastcmd:
            return self.onecmd(self.lastcmd)

    def do_help(self, arg):
        'List available commands with "help" or detailed help with "help cmd".'
        if arg:
            # XXX check arg syntax
            try:
                func = getattr(self, 'help_' + arg)
            except AttributeError:
                try:
                    doc=getattr(self, 'do_' + arg).__doc__
                    if doc:
                        self.print(f'{str(doc)}')
                        return
                except AttributeError:
                    pass
                self.print(f'{str(self.nohelp % (arg,))}')
                return
            func()
        else:
            names = self.get_names()
            cmds_doc = []
            cmds_undoc = []
            help = {}
            for name in names:
                if name[:5] == 'help_':
                    help[name[5:]]=1
            names.sort()
            # There can be duplicates if routines overridden
            prevname = ''
            for name in names:
                if name[:3] == 'do_':
                    if name == prevname:
                        continue
                    prevname = name
                    cmd=name[3:]
                    if cmd in help:
                        cmds_doc.append(cmd)
                        del help[cmd]
                    elif getattr(self, name).__doc__:
                        cmds_doc.append(cmd)
                    else:
                        cmds_undoc.append(cmd)
            self.print(f'{str(self.doc_leader)}', prefix='')
            wid, len = shutil.get_terminal_size()
            self.print_topics(self.doc_header, cmds_doc, len, wid)
            self.print_topics(self.misc_header, list(help.keys()), len, wid)
            self.print_topics(self.undoc_header, cmds_undoc, len, wid)

    def print_topics(self, header, cmds, cmdlen, maxcol):
        if cmds:
            self.stdout.write(" %s\n"%str(header))
            if self.ruler:
                self.stdout.write(" %s\n"%str(self.ruler * len(header)))
            self.columnize(cmds, maxcol-1)
            self.stdout.write("\n")

    def columnize(self, list, displaywidth=80):
        """Display a list of strings as a compact set of columns.

        Each column is only as wide as necessary.
        Columns are separated by two spaces (one was not legible enough).
        """
        if not list:
            self.stdout.write(" <empty>\n")
            return

        nonstrings = [i for i in range(len(list))
                        if not isinstance(list[i], str)]
        if nonstrings:
            raise TypeError("list[i] not a string for i in %s"
                            % ", ".join(map(str, nonstrings)))
        size = len(list)
        if size == 1:
            self.print(f'{str(list[0])}')
            return
        # Try every row count from 1 upwards
        for nrows in range(1, len(list)):
            ncols = (size+nrows-1) // nrows
            colwidths = []
            totwidth = -2
            for col in range(ncols):
                colwidth = 0
                for row in range(nrows):
                    i = row + nrows*col
                    if i >= size:
                        break
                    x = list[i]
                    colwidth = max(colwidth, len(x))
                colwidths.append(colwidth)
                totwidth += colwidth + 2
                if totwidth > displaywidth:
                    break
            if totwidth <= displaywidth:
                break
        else:
            nrows = len(list)
            ncols = 1
            colwidths = [0]
        for row in range(nrows):
            texts = []
            for col in range(ncols):
                i = row + nrows*col
                if i >= size:
                    x = ""
                else:
                    x = list[i]
                texts.append(x)
            while texts and not texts[-1]:
                del texts[-1]
            for col in range(len(texts)):
                texts[col] = texts[col].ljust(colwidths[col])
            self.print(f'{str("  ".join(texts))}', prefix='')

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.

        If this method is not overridden, it prints an error message and
        returns.

        """
        self.print(f'*** Unknown syntax: {line}')
        self.do_help('')

    def precmd(self, line):
        self.draw_line()
        if self.file and 'playback' not in line:
            print(line, file=self.file)
        return line

    def postcmd(self, stop, line):
        self.poll_processes()
        self.draw_line()
        self.do_send()
        self.do_read()

    # ----- flag commands -----
    def do_send(self, _=None):
        """No operation just sends flags to SharedFlagIO"""
        self.io.send_flags()

    def do_read(self, _=None):
        """Reads flags from SharedFlagIO if available otherwise defaults"""
        self.flags = self.io.read_flags() if self.io.read_flags() else Flags()

    @classmethod
    def _do_flag_methods(cls):
        form = """def do_{0}(self, arg):
                    if arg:
                        try:
                            self.flags.{0} = arg
                        except ValueError:
                            print(' ERROR: Invalid setting for {0}: ', arg)
                    self.do_help('{0}')
                    print('\tCurrent Setting: ', self.flags.{0})
                """
        _locals = {}
        [exec(form.format(i), globals(), _locals) for i in cls.flags]
        for name, func in _locals.items():
            func.__doc__ = DEFAULTS['DESCS'].get(name[len('do_'):])
            setattr(cls, name, func)

    def do_paths(self, arg):
        """Show all current path settings."""
        valid = ['annotation', 'backup', 'config', 'built_graph', 'img_out', 'labels',
                 'dataset', 'summary', 'binary', 'imgdir', 'video_out', 'model']
        self.do_read()
        width = shutil.get_terminal_size()[0]
        fl = dict(zip([i for i in self.flags], [i for i in self.flags.values()]))
        self.columnize([f' {name}: {val}' for name, val in fl.items() if name in valid],
                       displaywidth=width)

    def do_flags(self, arg):
        """Show all current flag settings"""
        self.do_read()
        width = shutil.get_terminal_size()[0]
        fl = dict(zip([i for i in self.flags], [i for i in self.flags.values()]))
        self.columnize([f' {name}: {val}' for name, val in fl.items()], displaywidth=width)

    # ----- process commands -----
    def do_term(self, arg):
        _processes = self.processes.copy()
        if arg == "all":
            for name in _processes.keys():
                self.print(f'Terminating {name} PID: {self.processes.get(name).pid}')
                self.processes.pop(name).terminate()
        else:
            try:
                self.print(f'Terminating {arg} PID: {self.processes.get(arg).pid}')
                self.processes.pop(arg).terminate()
            except AttributeError:
                if not arg == "all":
                    self.print(f'No process named {arg}', prefix='')

    def do_close(self, arg):
        """Stop recording and exit"""
        self.close()
        return True

    def do_fetch(self, arg):
        """Start the BEAGLES backend"""
        self.io.io_flags()
        forbidden = ['all', '']
        if arg in self.processes.keys():
            self.print(f'The process name {arg} is already taken.')
            return
        if arg in forbidden:
            self.print(f'The process name "{arg}" is invalid.')
            return
        self.do_send()
        self.do_read()
        command = [sys.executable, BACKEND_ENTRYPOINT]
        process = Popen(command, stdout=sys.stdout, stderr=sys.stderr, shell=False)
        process.poll()
        time.sleep(1)
        self.create_process(arg, process)

    def do_watch(self, arg):
        self.io.io_flags()
        if not self.processes:
            self.print('No processes to watch.')
        else:
            while True:
                # read flags and try to catch pid
                self.flags = self.io.read_flags()
                self.progress()
                key = select.select([sys.stdin], [], [], 0.100)[0]
                if key:
                    break
                self.poll_processes()
                if not self.processes:
                    break

    # ----- command completion -----
    def complete_model(self, text, *_):
        model_dir =  os.path.dirname(self.flags.model)
        if text:
            return [os.path.join(model_dir, text) for p in os.listdir(model_dir) if p.startswith(text) and p.endswith('.cfg')]
        else:
            return [p for p in os.listdir(model_dir) if p.endswith('.cfg')]

    def complete_playback(self, text, *_):
        if text:
            return [os.path.join(text, i) for i in os.listdir(text) if i.endswith('.bgl')]
        else:
            return ['tests/setup.bgl']

    def complete_load(self, text, *_):
        model, _ = os.path.basename(self.flags.model).split('.')
        files = glob.glob(self.flags.backup + model + '*.index')
        checkpoints = ['0', '-1']
        # a dash followed by a number or numbers followed by a dot
        _regex = re.compile('-[0-9]+\.')
        for f in files:
            _ckpt = re.search(_regex, f)
            start, end = _ckpt.span()
            n = f[start + 1:end - 1]
            checkpoints.append(n)
        checkpoints = [c for c in checkpoints if c.startswith(text)]
        checkpoints[0] = ' ' + checkpoints[0]
        return checkpoints

    def complete_trainer(self, text, *_):
        if text:
            names = [name for name in TRAINERS.keys() if name.startswith(text)]
            names[0] = ' ' + names[0]
            return names
        else:
            return TRAINERS.keys()

    def complete_clr_mode(self, text, *_):
        modes = ["triangular", "triangular2", "exp_range"]
        if text:
            names = [name for name in modes if name.startswith(text)]
            names[0] = ' ' + names[0]
            return names
        else:
            return modes

    # ----- record and playback -----
    def do_record(self, arg):
        """Save future commands to filename: RECORD rose.cmd"""
        self.file = open(arg, 'w')

    def do_playback(self, arg):
        """Playback commands from a file: PLAYBACK rose.cmd"""
        if self.file:
            self.file.close()
            self.file = None

        if os.path.isfile(arg):
            self.print(f'Running commands from {arg}')
            with open(arg) as f:
                self.cmdqueue.extend(f.read().splitlines())
        else:
            self.print(f'File {arg} not found')


if __name__ == '__main__':
    caught_sig = CatchSignals
    shell = BeaglesShell()
    def ask():
        affirmative = ['y', 'Y']
        answer = input(
            'Are you sure you want exit and close all running processes? y/[n]')
        if answer in affirmative:
            shell.close()
        else:
            shell.cmdloop()
    try:
        while not caught_sig.kill:
            intro = "BEAGLES (BEhavioral Annotation and Gesture LEarning Suite) Backend Shell\n"\
                    + SHELL_HLINE
            shell.cmdloop(intro)
        else:
            print(f'A fatal signal was caught. {APP_NAME} has exited gracefully.')
            ask()
    except KeyboardInterrupt:
        print(f' keyboard interrupt was caught. {APP_NAME} has exited the shell loop.')
        ask()
