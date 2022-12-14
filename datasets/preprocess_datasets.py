import pandas as pd
import numpy as np
from helper import save_as_json
import warnings

FIELDS = ['text', 'context']

warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess(folder, readFile, saveFile):
    for field in FIELDS:
        df = pd.read_json(folder+'/'+readFile, lines=True)
        # Remove @...
        df[field] = df[field].replace(regex=['(@\w+)'], value='')
        # Remove links
        df[field] = df[field].replace(regex=['(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)'], value='')
        # Remove multiple spaces
        df[field] = df[field].replace(regex=['  +'], value=' ')
        # Remove emojis
        df[field] = df[field].apply(lambda s: ''.join(filter(lambda c: ord(c) < 256, s)))
        # Remove leading and trailing spaces
        df[field] = df[field].str.strip()
        # Drop empty columns
        df[field].replace('', np.nan, inplace=True)
        df.dropna(subset=[field], inplace=True)
    # Save
    save_as_json(df, folder+'/'+saveFile)


preprocess('ETS-Sarcasm', 'ETS_1.jsonl', 'ETS_1_preprocessed.jsonl')
preprocess('ETS-Sarcasm', 'ETS_2.jsonl', 'ETS_2_preprocessed.jsonl')
preprocess('SPIRS', 'SPIRS.jsonl', 'SPIRS_preprocessed.jsonl')
