#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
EXEC_PATH = os.path.abspath("../../")
os.chdir(EXEC_PATH)
import sys
sys.path.append(os.getcwd())
import cmd
import shutil
from subprocess import Popen, DEVNULL
from beagles.backend.net import TRAINERS
from beagles.base import *
from beagles.io.flags import SharedFlagIO

form = """
def do_{0}(self, arg):
    if arg:
        self.flags.{0} = arg
    print('Current Setting: ', self.flags.{0})
"""

class BeaglesShell(cmd.Cmd):
    intro = f'Welcome to the {APP_NAME} shell.\nType help or ? to list commands.\n'
    file = None
    flags = Flags()
    io = SharedFlagIO(subprogram=False, flags=flags)
    prompt = f'{APP_NAME} >>> '
    completekey = 'tab'

    @classmethod
    def _preloop(cls):
        _locals = {}
        [exec(form.format(i), globals(), _locals) for i in cls.flags]
        for name, func in _locals.items():
            func.__doc__ = DEFAULTS['DESCS'].get(name[len('do_'):])
            setattr(cls, name, func)

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
        self.io.io_flags()

    def close(self):
        if self.file:
            self.file.close()
            self.file = None
        exit(0)

    def do_exit(self, arg):
        """Stop recording and exit: EXIT"""
        self.close()
        return True

    def do_fetch(self, arg):
        """Start the BEAGLES backend"""
        self.io.send_flags()
        self.flags = self.io.read_flags()
        Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=DEVNULL, shell=False)

    def do_flags(self, arg):
        """Show all current flag settings"""
        width = shutil.get_terminal_size()[0]
        fl = dict(zip([i for i in self.flags], [i for i in self.flags.values()]))
        self.columnize([f'{desc}: {val}' for desc, val in fl.items()], displaywidth=width)

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
