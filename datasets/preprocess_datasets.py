import pandas as pd
import numpy as np
from helper import save_as_json
import warnings


warnings.simplefilter(action='ignore', category=FutureWarning)


def preprocess(folder, readFile, saveFile):
    df = pd.read_json(folder+'/'+readFile, lines=True)
    # Remove @...
    df['text'] = df['text'].str.replace('(@\w+)', '')
    # Remove links
    df['text'] = df['text'].replace(regex=['(https?:\/\/)?([\da-z\.-]+)\.([a-z\.]{2,6})([\/\w\.-]*)'], value='')
    # Remove multiple spaces
    df['text'] = df['text'].replace(regex=['  +'], value=' ')
    # Remove emojis
    df['text'] = df['text'].apply(lambda s: ''.join(filter(lambda c: ord(c) < 256, s)))
    # Remove leading and trailing spaces
    df['text'] = df['text'].str.strip()
    # Drop empty columns
    df['text'].replace('', np.nan, inplace=True)
    df.dropna(subset=['text'], inplace=True)
    # Save
    save_as_json(df, folder+'/'+saveFile)


preprocess('ETS-Sarcasm', 'ETS.jsonl', 'ETS_preprocessed.jsonl')
preprocess('iSarcasm', 'iSarcasm.jsonl', 'iSarcasm_preprocessed.jsonl')
preprocess('SPIRS', 'SPIRS.jsonl', 'SPIRS_preprocessed.jsonl')
