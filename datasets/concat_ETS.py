import pandas as pd
import os


def addFile(path):
    print("Read " + path)
    with open("ETS-Sarcasm/ETS.jsonl", "a") as jsonFile:
        rawData = pd.read_json(path, lines=True)
        data = rawData[['label', 'response']].copy().rename(columns={"response": "text"}, errors="raise")
        out = data.to_json(orient="records").replace('},{', '}\n{').replace('[', '').replace(']', '\n')
        jsonFile.write(out)


if os.path.exists("ETS-Sarcasm/ETS.jsonl"):
  os.remove("ETS-Sarcasm/ETS.jsonl")

addFile('ETS-Sarcasm/reddit/sarcasm_detection_shared_task_reddit_testing.jsonl')
addFile('ETS-Sarcasm/reddit/sarcasm_detection_shared_task_reddit_training.jsonl')
addFile('ETS-Sarcasm/twitter/sarcasm_detection_shared_task_twitter_testing.jsonl')
addFile('ETS-Sarcasm/twitter/sarcasm_detection_shared_task_twitter_training.jsonl')


