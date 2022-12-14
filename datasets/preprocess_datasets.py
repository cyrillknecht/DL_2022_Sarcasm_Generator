import pandas as pd
import numpy as np
from helper import save_as_json
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess(folder, readFile, saveFile, field):
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


preprocess('ETS-Sarcasm', 'ETS.jsonl', 'ETS_preprocessed.jsonl', 'text')
preprocess('SPIRS', 'SPIRS.jsonl', 'SPIRS_preprocessed.jsonl', 'context')
preprocess('SPIRS', 'SPIRS.jsonl', 'SPIRS_preprocessed.jsonl', 'text')
