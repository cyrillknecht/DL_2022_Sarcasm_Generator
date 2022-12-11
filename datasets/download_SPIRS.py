from auth import *
import csv
from tqdm import tqdm


def get_dataset(file, label, id_idx):
    print("Downloading SPIRS/"+file+".csv")
    with open("SPIRS/"+file+".csv") as csv_file:
        lines = len(csv_file.readlines())

    with open("SPIRS/"+file+".jsonl", "w") as jsonFile:
        with open("SPIRS/"+file+".csv") as csvFile:
            reader = csv.reader(csvFile)
            for row in tqdm(reader, total=lines):
                tweet_id = row[id_idx]
                try:
                    tweet = api.get_status(id=tweet_id)
                    # Remove line breaks
                    text = tweet.text.replace('\n', ' ').replace('\r', '')
                    # Escape double quotes
                    text = text.replace('"', '\\"')
                    # Escape backslash
                    text = text.replace('\\', '\\\\')

                    jsonFile.write('{"label":"' + label + '","text":"' + text + '"}\n')

                except:
                    pass


get_dataset('SPIRS-sarcastic-ids', 'SARCASM', 4)
get_dataset('SPIRS-non-sarcastic-ids', 'NOT_SARCASM', 2)
