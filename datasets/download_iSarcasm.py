from auth import *
import csv
from tqdm import tqdm


def get_dataset(file):
    print("Downloading iSarcasm/"+file+".csv")
    with open("iSarcasm/"+file+".csv") as csv_file:
        lines = len(csv_file.readlines())

    with open("iSarcasm/"+file+".jsonl", "w") as jsonFile:
        with open("iSarcasm/"+file+".csv") as csvFile:
            reader = csv.reader(csvFile)
            for row in tqdm(reader, total=lines):
                tweet_id = row[0]
                classification = row[1]
                try:
                    tweet = api.get_status(id=tweet_id)
                    # Remove line breaks
                    text = tweet.text.replace('\n', ' ').replace('\r', '')
                    # Escape double quotes
                    text = text.replace('"', '\\"')
                    # Escape backslash
                    text = text.replace('\\', '\\\\')
                    label = 'NOT_SARCASM'
                    if classification == "sarcastic":
                        label = 'SARCASM'

                    jsonFile.write('{"label":"' + label + '","text":"' + text + '"}\n')

                except:
                    pass


get_dataset('iSarcasm_train')
get_dataset('iSarcasm_test')
