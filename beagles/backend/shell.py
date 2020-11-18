import cmd
import shutil
from subprocess import Popen, PIPE
from beagles.base import *
from beagles.io.flags import SharedFlagIO
EXEC_PATH = os.path.abspath("../../")
os.chdir(EXEC_PATH)


class BeaglesShell(cmd.Cmd):
    intro = f'Welcome to the {APP_NAME} shell.\nType help or ? to list commands.\n'
    file = None
    flags = Flags()
    io = SharedFlagIO(subprogram=False, flags=flags)
    prompt = f'{APP_NAME} >>> '

    def do_exit(self, arg):
        """Stop recording and exit: EXIT"""
        self.close()
        return True

    def do_fetch(self, arg):
        self.io.send_flags()
        self.flags = self.io.read_flags()
        Popen([sys.executable, BACKEND_ENTRYPOINT], stdout=PIPE, shell=False)

    def do_env(self, arg):
        width = shutil.get_terminal_size()[0]
        fl = dict(zip([i for i in self.flags], [i for i in self.flags.values()]))
        self.columnize([f'{desc}: {val}' for desc, val in fl.items()], displaywidth=width)

    # ----- record and playback -----
    def do_record(self, arg):
        """Save future commands to filename: RECORD rose.cmd"""
        self.file = open(arg, 'w')

    def do_playback(self, arg):
        """Playback commands from a file: PLAYBACK rose.cmd"""
        self.close()
        with open(arg) as f:
            self.cmdqueue.extend(f.read().splitlines())

    def precmd(self, line):
        self.io.io_flags()
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

form = """
def do_{0}(self, arg):
    if arg:
        self.flags.{0} = arg
    print('Current Setting: ', self.flags.{0})
"""

if __name__ == '__main__':
    _locals = {}
    for i in BeaglesShell.flags:
        exec(form.format(i), globals(), _locals)
        for name, func in _locals.items():
            func.__doc__ = DEFAULTS['DESCS'].get(name[len('do_'):])
            setattr(BeaglesShell, name, func)

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
