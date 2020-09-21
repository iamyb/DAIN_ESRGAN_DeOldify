import os
import sys
import argparse
import importlib

def call_handler(proc, target):
    module = importlib.import_module('utils.'+proc)
    getattr(module, proc)(target)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python run.py')
    parser.add_argument('-i', '--input', required=True)
    parser.add_argument('-p', '--proc', type=lambda arg: arg.split(',') )
    args = parser.parse_args()

    target = args.input
    all_proc = args.proc if args.proc else ['dain','esrgan','deoldify','build']
    sys.argv = [sys.argv[0]]  # to avoid the argparse error in DAIN
    
    for proc in all_proc:
        call_handler(proc, target)
