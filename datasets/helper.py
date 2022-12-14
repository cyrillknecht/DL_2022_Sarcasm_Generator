import pandas as pd


def save_as_json(df, path):
    out = df.to_json(orient="records").replace('},{', '}\n{').replace('[', '').replace(']', '')

    with open(path, "w") as jsonFile:
        jsonFile.write(out)


def shuffle_file(path):
    data = pd.read_json(path, lines=True)
    data = data.sample(frac=1)
    save_as_json(data, path)