#!/usr/bin/env python3
# TODO: refactor needed, this is fragile and difficult to properly test
import os
import sys
import argparse
try:
    argv = sys.argv[1]
except IndexError:
    argv = []
EXEC_PATH = os.path.abspath("../../") if argv else os.getcwd()
sys.path.append(EXEC_PATH)
from libs.backend.trainer import Trainer
from libs.backend.predictor import Predictor
from libs.backend.annotator import Annotator
from libs.constants import BACKEND_ENTRYPOINT
from libs.io.flags import  FlagIO
from libs.utils.flags import Flags
os.chdir(EXEC_PATH)

def parse_arguments():
    parser = argparse.ArgumentParser(
        formatter_class=lambda prog: argparse.HelpFormatter(prog, max_help_position=32))
    parser.add_argument('--train', default=Flags().train,
                        action='store_true',
                        help='train a model on annotated data')
    parser.add_argument('--video', default=Flags().video,
                        help='generate frame-by-frame annotation')
    parser.add_argument('--project_name', default=Flags().project_name, type=str,
                        help='project name for tensorboard summaries')
    parser.add_argument('--save_video', default=Flags().save_video,
                        help='filename of video output')
    parser.add_argument('--imgdir', default=Flags().imgdir,
                        metavar='',
                        help='path to testing directory with images')
    parser.add_argument('--img_out', default=Flags().img_out)
    parser.add_argument('--output_type', default=Flags().output_type)
    parser.add_argument('--binary', default=Flags().binary,
                        metavar='',
                        help='path to .weights directory')
    parser.add_argument('--config', default=Flags().config,
                        metavar='',
                        help='path to .cfg directory')
    parser.add_argument('--dataset', default=Flags().dataset,
                        metavar='',
                        help='path to dataset directory')
    parser.add_argument('--backup', default=Flags().backup,
                        metavar='',
                        help='path to checkpoint directory')
    parser.add_argument('--labels', default=Flags().labels,
                        metavar='',
                        help='path to textfile containing labels')
    parser.add_argument('--annotation', default=Flags().annotation,
                        metavar='',
                        help='path to the annotation directory')
    parser.add_argument('--summary', default=Flags().summary,
                        help='path to Tensorboard summaries directory')
    parser.add_argument('--log', default=Flags().log,
                        help='path to log directory')
    parser.add_argument('--trainer', default=Flags().trainer,
                        metavar='', help='training algorithm')
    parser.add_argument('--momentum', default=Flags().momentum,
                        metavar='',
                        help='applicable for rmsprop and momentum optimizers')
    parser.add_argument('--keep', default=Flags().keep,
                        metavar='N',
                        help='number of most recent training results to save')
    parser.add_argument('--batch', default=Flags().batch,
                        metavar='N', help='batch size')
    parser.add_argument('--epoch', default=Flags().epoch,
                        type=int, metavar='N', help='number of epochs')
    parser.add_argument('--save', default=Flags().save,
                        metavar='N',
                        help='save a checkpoint ever N training examples')
    parser.add_argument('--gpu', default=Flags().gpu,
                        metavar='[0 .. 1.0]',
                        help='amount of GPU to use')
    parser.add_argument('--gpu_name', default=Flags().gpu_name,
                        metavar='/gpu:N', help='GPU device name')
    parser.add_argument('-l', '--load', default=Flags().load,
                        metavar='',
                        help='filename of weights or checkpoint to load')
    parser.add_argument('-m', '--model', default=Flags().model,
                        metavar='', help='filename of model to use')
    parser.add_argument('--threshold', default=Flags().threshold,
                        metavar='[0.01 .. 0.99]',
                        help='threshold of confidence to record an annotation')
    parser.add_argument('--clip', default=Flags().clip,
                        help="clip if gradient explodes")
    parser.add_argument('--lr', default=Flags().lr,
                        metavar='N', help='learning rate')
    parser.add_argument('--max_lr', default=Flags().max_lr,
                        metavar='N', help='max learning rate')
    parser.add_argument('-v', '--verbalise', default=Flags().verbalise,
                        action='store_true',
                        help='show graph structure while building')
    parser.add_argument('--kill', default=Flags().kill,
                        help=argparse.SUPPRESS)
    parser.add_argument('--done', default=Flags().done,
                        help=argparse.SUPPRESS)
    parser.add_argument('--started', default=Flags().started,
                        help=argparse.SUPPRESS)
    return parser.parse_args()

if __name__ == '__main__':
    print(
        r"""
         ________  _______   ________  ________  ___       _______   ________      
        |\   __  \|\  ___ \ |\   __  \|\   ____\|\  \     |\  ___ \ |\   ____\     
        \ \  \|\ /\ \   __/|\ \  \|\  \ \  \___|\ \  \    \ \   __/|\ \  \___|_    
         \ \   __  \ \  \_|/_\ \   __  \ \  \  __\ \  \    \ \  \_|/_\ \_____  \   
          \ \  \|\  \ \  \_|\ \ \  \ \  \ \  \|\  \ \  \____\ \  \_|\ \|____|\  \  
           \ \_______\ \_______\ \__\ \__\ \_______\ \_______\ \_______\____\_\  \ 
            \|_______|\|_______|\|__|\|__|\|_______|\|_______|\|_______|\_________\
                                                                       \|_________|
             (BEhavioral Annotation and Gesture LEarning Suite) Backend Wrapper""", '\n'
    )
    io = FlagIO(subprogram=True)
    flags = parse_arguments() if argv else io.read_flags()
    if not flags and not argv:
        raise RuntimeError(
            f"Wrapper started in flag mode without a prior call to io_flags()")
    flags.started = True
    if isinstance(flags, Flags):
        io.io_flags()
    if flags.train:
        wrapper = Trainer(flags)
    elif flags.video != []:
        wrapper = Annotator(flags)
    else:
        wrapper = Predictor(flags)
    wrapper()
    io.read_flags()
    flags.progress = 100
    flags.done = True
    io.io_flags()
    exit(0)
