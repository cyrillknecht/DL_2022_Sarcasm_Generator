from auth import *
import csv
from tqdm import tqdm
import json


def get_tweet(tweet_id):
    try:
        tweet = api.get_status(id=tweet_id)
        # Remove line breaks
        return tweet.text.replace('\n', ' ').replace('\r', '')
    except:
        pass

    return ''


def get_dataset(file, label, eli_idx, sar_idx):
    print("")
    print("Downloading SPIRS/"+file+".csv")
    with open("SPIRS/"+file+".csv") as csv_file:
        lines = len(csv_file.readlines())

    with open("SPIRS/"+file+".jsonl", "w") as jsonFile:
        with open("SPIRS/"+file+".csv") as csvFile:
            reader = csv.reader(csvFile)
            for row in tqdm(reader, total=lines):
                eli_text = get_tweet(row[eli_idx])
                sar_text = get_tweet(row[sar_idx])

                if sar_text and eli_text:

                    json.dump({"label": label, "context": eli_text, "text": sar_text}, jsonFile)
                    jsonFile.write('\n')


get_dataset('SPIRS-sarcastic-ids', 'SARCASM', 6, 4)
get_dataset('SPIRS-non-sarcastic-ids', 'NOT_SARCASM', 3, 1)
