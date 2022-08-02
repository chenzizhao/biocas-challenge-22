"""
python3 main.py --task 11 --wav testcase/task1_wav/ --out testcase/my_output/task11_output.json
"""

# Update me before release
resume_paths = {
    11:'saved_v/model11/model_best11.pth',
    12:'saved_v/model12/model_best12.pth',
    21:'saved_v/model21/model_best21.pth',
    22:'saved_v/model22/model_best22.pth'
}

import argparse
import pathlib
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='release main')
    parser.add_argument('--task', type=int, choices=[11,12,21,22], required=True)
    parser.add_argument('--wav', type=pathlib.Path, required=True)
    parser.add_argument('--out', type=pathlib.Path, required=True)
    args = parser.parse_args()
    tasklevel = args.task
    resume_path = resume_paths[tasklevel]
    config_file = f'config_task{tasklevel}.json'
    command = f'python3 test.py -c {config_file} -r {resume_path} --wav {args.wav} --out {args.out}'
    os.system(command)
    