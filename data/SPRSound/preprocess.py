"""
TODO
This file extracts patients' info from the (name of) json files.
"""

from os import listdir
from os.path import join
import json
import pandas as pd
# from torchaudio import load
from pydub import AudioSegment

JSON_DIR = "json"
WAV_DIR = "wav"
CLIP_DIR = "clip"
PROC_DIR = "processed"
TASK1_CSV = "task1.csv"
TASK2_CSV = "task2.csv"

def preprocess(wav):
    """
    This function will be exported to main.py
    """
    processed = wav
    # TODO
    # could be a image
    return processed

def parse_json_and_clip(jname):
    name = jname[:-5]
    wav_name = name + '.wav'
    entry = name.split('_')
    patient_id = int(entry[0])
    age = float(entry[1])
    gender = int(entry[2])
    loc = int(entry[3][-1])
    rec_id = int(entry[4])

    with open(join(JSON_DIR, jname)) as f:
        rec_json = json.load(f)
    
    label_22 = rec_json["record_annotation"]
    label_21 = 'Adventitious' if label_22 not in ('Normal', 'Poor Quality') else label_22
    new_entry2 = (wav_name, patient_id, age, gender, loc, rec_id, 
        label_21, label_22)

    new_entry1s = []
    audio = AudioSegment.from_wav(join(WAV_DIR, wav_name))
    events = rec_json['event_annotation']
    for (i, event) in enumerate(events):
        start = int(event['start'])
        end = int(event['end'])
        clip_name = name + '_' + str(i) + '.wav'
        clip = audio[start:end]
        clip.export(join(CLIP_DIR, clip_name), format='wav')

        label_12 = event['type']
        label_11 = 'Adventitious' if label_12 != 'Normal' else label_12
        new_entry1 = (clip_name, patient_id, age, gender, loc, rec_id, i,
            label_11, label_12)
        new_entry1s.append(new_entry1)
    
    return new_entry1s, new_entry2

if __name__ == '__main__':

    ls1 = []
    ls2 = []
    for jname in listdir(JSON_DIR):
        entry1s, entry2 = parse_json_and_clip(jname)
        ls1.extend(entry1s)
        ls2.append(entry2)

    print(f'audio clipped. stored in {CLIP_DIR}/.')

    df1 = pd.DataFrame(ls1, columns=[
        'wav_name', 'patient_id', 'age', 'gender', 'loc', 'rec_id', 'event_id', 
        'label_11', 'label_12'])
    df2 = pd.DataFrame(ls2, columns=[
        'wav_name', 'patient_id', 'age', 'gender', 'loc', 'rec_id',
        'label_21', 'label_22'])

    df1.to_csv(TASK1_CSV)
    df2.to_csv(TASK2_CSV)
    print(f'json parsed. {TASK1_CSV} and {TASK2_CSV} files stored.')
    print('done.')
