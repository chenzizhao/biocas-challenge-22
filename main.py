"""
python3 main.py --task 11 --wav testcase/task1_wav/ --out testcase/my_output/task11_output.json
"""

# Update me before release
resume_paths = {
    11:'saved/models/Audio_Resp_11/0608_020201/model_best.pth',
    12:'saved/models/Audio_Resp_12/0608_212233/model_best.pth',
    21:'saved/models/Audio_Resp_21/0608_214241/model_best.pth',
    22:'saved/models/Audio_Resp_22/0608_221550/model_best.pth'
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
    command = f'python test.py -c {config_file} -r {resume_path} --wav {args.wav} --out {args.out}'
    os.system(command)
    