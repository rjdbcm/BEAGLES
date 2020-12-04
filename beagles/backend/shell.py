#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
EXEC_PATH = os.path.abspath("../../")
os.chdir(EXEC_PATH)
import sys
sys.path.append(os.getcwd())
import cmd
import time
import select
import shutil
import readline
from subprocess import Popen, DEVNULL
from beagles.backend.net import TRAINERS
from beagles.base import *
from beagles.io.flags import SharedFlagIO

form = """
def do_{0}(self, arg):
    if arg:
        self.flags.{0} = arg
    self.do_help('{0}')
    print('Current Setting: ', self.flags.{0})
"""

class BeaglesShell(cmd.Cmd):
    intro = f'Welcome to the {APP_NAME} shell.\nType help or ? to list commands.\n'
    file = None
    processes = {}
    _current = None
    flags = Flags()
    io = SharedFlagIO(subprogram=False, flags=flags)
    prompt = f'{APP_NAME} >>> '
    completekey = 'tab'
    use_rawinput = 0

    @classmethod
    def _preloop(cls):
        _locals = {}
        [exec(form.format(i), globals(), _locals) for i in cls.flags]
        for name, func in _locals.items():
            func.__doc__ = DEFAULTS['DESCS'].get(name[len('do_'):])
            setattr(cls, name, func)

    def emptyline(self):
        """Called when an empty line is entered in response to the prompt.

        If this method is not overridden, it repeats the last nonempty
        command entered.

        """
        if self.lastcmd == 'watch':
            return
        if self.lastcmd:
            return self.onecmd(self.lastcmd)

    def default(self, line):
        """Called on an input line when the command prefix is not recognized.

        If this method is not overridden, it prints an error message and
        returns.

        """
        self.stdout.write(f'*** Unknown syntax: {line}\n')
        self.do_help('')

    def progress(self, total=100.0):
        form = ' {}{}%\r'
        def announce(bar, percents):
            sys.stdout.write(form.format(bar, percents))
            sys.stdout.flush()
        bar_len = shutil.get_terminal_size()[0] - len(form) - 4
        filled_len = int(round(bar_len * self.flags.progress / float(total)))
        percents = round(100.0 * self.flags.progress / float(total), 1)
        bar = u"\u2593" * filled_len + u"\u2591" * (bar_len - filled_len)
        announce(bar, percents)
        result = select.select([sys.stdin], [], [], 0.5)[0]
        return result

    def do_term(self, arg):
        _processes = self.processes.copy()
        if arg == "all":
            for name in _processes.keys():
                print(f'Terminating {name} PID: {self.processes.get(name).pid}')
                self.processes.pop(name).terminate()
        else:
            try:
                print(f'Terminating {arg} PID: {self.processes.get(arg).pid}')
                self.processes.pop(arg).terminate()
            except AttributeError:
                if not arg == "all":
                    print(f'No process named {arg}')

    def preloop(self):
        self.flags.cli = True
        self._preloop()

    def precmd(self, line):
        avail_flags = self.io.read_flags()
        self.flags = avail_flags if avail_flags else Flags()
        if self.file and 'playback' not in line:
            print(line, file=self.file)
        return line

    def postcmd(self, stop, line):
        self.io.send_flags()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
        self.do_term('all')
        exit(0)

    def do_exit(self, arg):
        """Stop recording and exit: EXIT"""
        self.close()
        return True

    def do_fetch(self, arg):
        """Start the BEAGLES backend"""
        if arg in self.processes.keys():
            print(f'The process name {arg} is already taken.')
            return
        if arg == 'all' or '':
            print(f'The process name {arg} is invalid.')
            return
        self.io.send_flags()
        self.flags = self.io.read_flags()
        current = Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=DEVNULL, stderr=DEVNULL, shell=False)
        self.processes.update({arg: current})

    def do_watch(self, arg):
        if not self.processes:
            print("No processes to watch.")
        else:
            while True:
                self.flags = self.io.read_flags()
                ret = self.progress()
                if len(ret) > 0:
                    break
                time.sleep(.25)

    def do_flags(self, arg):
        """Show all current flag settings"""
        self.flags = self.io.read_flags()
        width = shutil.get_terminal_size()[0]
        fl = dict(zip([i for i in self.flags], [i for i in self.flags.values()]))
        self.columnize([f'{desc}: {val}' for desc, val in fl.items()], displaywidth=width)

    # ----- command completion -----
    def complete_load(self, *_):
        pass

    def complete_trainer(self, text, *_):
        if text:
            return [name for name in TRAINERS.keys() if name.startswith(text)]
        else:
            return TRAINERS.keys()

    def complete_clr_mode(self, text, *_):
        modes = ["triangular", "triangular2", "exp_range"]
        if text:
            return [name for name in modes if name.startswith(text)]
        else:
             return modes

    # ----- record and playback -----
    def do_record(self, arg):
        """Save future commands to filename: RECORD rose.cmd"""
        self.file = open(arg, 'w')

    def do_playback(self, arg):
        """Playback commands from a file: PLAYBACK rose.cmd"""
        self.close()
        with open(arg) as f:
            self.cmdqueue.extend(f.read().splitlines())

if __name__ == '__main__':
    BeaglesShell().cmdloop(intro=r"""
         ________  _______   ________  ________  ___       _______   ________      
        |\   __  \|\  ___ \ |\   __  \|\   ____\|\  \     |\  ___ \ |\   ____\     
        \ \  \|\ /\ \   __/|\ \  \|\  \ \  \___|\ \  \    \ \   __/|\ \  \___|_    
         \ \   __  \ \  \_|/_\ \   __  \ \  \  __\ \  \    \ \  \_|/_\ \_____  \   
          \ \  \|\  \ \  \_|\ \ \  \ \  \ \  \|\  \ \  \____\ \  \_|\ \|____|\  \  
           \ \_______\ \_______\ \__\ \__\ \_______\ \_______\ \_______\____\_\  \ 
            \|_______|\|_______|\|__|\|__|\|_______|\|_______|\|_______|\_________\
                                                                       \|_________|
             (BEhavioral Annotation and Gesture LEarning Suite) Backend Wrapper""")
