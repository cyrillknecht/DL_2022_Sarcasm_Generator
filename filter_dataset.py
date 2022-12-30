import pandas as pd

train_data = pd.read_json('dataset/ets_twitter_train_data_generator.jsonl', lines=True)

train_data = train_data[train_data["label"]!="SARCASM"]

with open("dataset/ets_twitter_train_data_generator_only_non_sarcasm.jsonl", "w") as f:
    f.write(train_data.to_json(orient="records", lines=True))