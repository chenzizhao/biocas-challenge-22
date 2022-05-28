"""
TODO
"""


"""
This file extracts patients' info from the (name of) json files.
"""

from os import listdir
from os.path import join
import json
import pandas as pd
from torchaudio import load

def parse_json2(json_dir):
    ls = []
    for fname in listdir('json'):
        entry = fname[:-5].split('_')
        
        patient_id = int(entry[0])
        age = float(entry[1])
        gender = int(entry[2])
        loc = int(entry[3][-1])
        rec_id = int(entry[4])

        with open(join(json_dir, fname)) as f:
            rec_json = json.load(f)
            label_22 = rec_json["record_annotation"]

        if label_22 in ("CAS", "DAS", "CAS & DAS"):
            label_21 = "Adventitious"
        else:
            label_21 = label_22

        new_entry = (fname, patient_id, age, gender, loc, rec_id, label_21, label_22)
        ls.append(new_entry)
    
    df = pd.DataFrame(ls, columns=[
        'fname', 'patient_id', 'age', 'gender', 'loc', 'rec_id', 
        'label_21', 'label_22'])
    return df

if __name__ == '__main__':

    processed_dir = 'processed'
    json_dir = 'json'

    df2 = parse_json2(json_dir)
    df2.to_csv(join(processed_dir, 'rec_info.csv'))
    
    print(f"done. All files stored in {processed_dir} folder.")
