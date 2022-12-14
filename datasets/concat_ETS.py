import pandas as pd
import os
from helper import save_as_json, shuffle_file


def addFile(path):
    print("Read " + path)
    with open("ETS-Sarcasm/ETS_1.jsonl", "a") as jsonFile_1:
        with open("ETS-Sarcasm/ETS_2.jsonl", "a") as jsonFile_2:
            rawData = pd.read_json(path, lines=True)
            data = rawData[['label', 'response', 'context']].copy().rename(columns={"response": "text"}, errors="raise")
            for i, row in rawData.iterrows():
                contextStr = ''
                for r in row['context']:
                    contextStr += ' '+r
                data['context'][i] = contextStr

            mid = int(len(data.index) / 2)

            data_1 = data.iloc[:mid, :]
            out = data_1.to_json(orient="records").replace('},{', '}\n{').replace('[', '').replace('}]', '}\n')
            jsonFile_1.write(out)

            data_2 = data.iloc[mid:, :]
            out = data_2.to_json(orient="records").replace('},{', '}\n{').replace('[', '').replace('}]', '}\n')
            jsonFile_2.write(out)


if os.path.exists("ETS-Sarcasm/ETS_1.jsonl"):
  os.remove("ETS-Sarcasm/ETS_1.jsonl")

if os.path.exists("ETS-Sarcasm/ETS_2.jsonl"):
  os.remove("ETS-Sarcasm/ETS_2.jsonl")

addFile('ETS-Sarcasm/reddit/sarcasm_detection_shared_task_reddit_testing.jsonl')
addFile('ETS-Sarcasm/reddit/sarcasm_detection_shared_task_reddit_training.jsonl')
addFile('ETS-Sarcasm/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl')
addFile('ETS-Sarcasm/twitter/sarcasm_detection_shared_task_twitter_training.jsonl')

# shuffle json file
shuffle_file("ETS-Sarcasm/ETS_1.jsonl")
shuffle_file("ETS-Sarcasm/ETS_2.jsonl")


