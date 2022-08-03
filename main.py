"""
python3 main.py --task 11 --wav testcase/task1_wav/ --out testcase/my_output/task11_output.json
"""

# Update me before release
resume_paths = {
    11:'saved_v/model11',
    12:'saved_v/model12',
    21:'saved_v/model21',
    22:'saved_v/model22'
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
    base_path = resume_paths[tasklevel]
    resume_path = os.path.join(base_path, f'model_best{tasklevel}.pth')
    config_file = os.path.join(base_path, 'config.json')
    command = f'python3 test.py -c {config_file} -r {resume_path} --wav {args.wav} --out {args.out} --checkpoint_dir saved/'
    os.system(command)
    