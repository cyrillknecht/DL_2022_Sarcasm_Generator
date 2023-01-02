"""
    Judges the quality of generated answers by an independent classifier
"""
from classifier import Classifier
import csv

TRAINED_CLASSIFIER_PATH = "models/judge"
FOLDER = "generated"

FILES = [
    "eda_augmented_results_30_10.csv",
    "eda_augmented_results_20_20.csv",
    "eda_augmented_results_10_30.csv",
    "eda_augmented_results_0_40.csv",
]

judge = Classifier()
judge.load_classifier(TRAINED_CLASSIFIER_PATH)

for file in FILES:
    samples = 0
    sarcasm = 0
    if file[-4:] == ".txt":
        with open(FOLDER + "/" + file) as f:
            for line in f:
                samples += 1
                if judge.classify_tweet(line):
                    sarcasm += 1
    elif file[-4:] == ".csv":  # adapted from gpt2simple's load_dataset
        with open(FOLDER + "/" + file, "r", encoding="utf8", errors="ignore") as f:
            f.readline()  # skip header
            reader = csv.reader(f)
            for line in reader:
                samples += 1
                if judge.classify_tweet(line[0]):
                    sarcasm += 1

    print("Results for " + file)
    print("Total Samples: " + str(samples))
    print("Classified as sarcasm: " + str(sarcasm))
    if sarcasm == 0:
        percentage = 0
    else:
        percentage = sarcasm / samples * 100

    print("Sarcasm percentage: " + str(percentage))
    print()
