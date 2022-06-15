"""
python3 main.py --task 11 --wav testcase/task1_wav/ --out testcase/my_output/task11_output.json
"""

import argparse
import json
from os import listdir
from tqdm import tqdm
from pathlib import Path

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt', encoding='utf8') as handle:
        json.dump(content, handle, sort_keys=False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='release main')
    parser.add_argument('--task', type=int, choices=[11,12,21,22], required=True)
    parser.add_argument('--wav', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    
    output = {}
    for wav_name in tqdm(listdir(args.wav)):
        output[wav_name] = 'Normal'
    write_json(output, args.out)
