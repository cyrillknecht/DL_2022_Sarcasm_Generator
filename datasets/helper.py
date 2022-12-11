
def save_as_json(df, path):
    out = df.to_json(orient="records").replace('},{', '}\n{').replace('[', '').replace(']', '')

    with open(path, "w") as jsonFile:
        jsonFile.write(out)
