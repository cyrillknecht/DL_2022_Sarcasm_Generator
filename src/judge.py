"""
    Judges the quality of generated answers by an independent classifier
"""
from classifier import Classifier
import csv
import sys

TRAINED_CLASSIFIER_PATH = "models/judge"
FOLDER = "generated"

# NOTE: Uncomment multiple files to be evaluated or pass name as argument!
FILES = [
    # Small Dataset
    "100_classic_results.csv",  # 0%
    # "100_self_augmented_results_30_10.csv",  # 25%
    # "100_self_augmented_results_20_20.csv",  # 50%
    # "100_self_augmented_results_16_24.csv",  # 60%
    # "100_self_augmented_results_14_26.csv",  # 65%
    # "100_self_augmented_results_12_28.csv",  # 70%
    "100_self_augmented_results_10_30.csv",  # 75%
    # "100_self_augmented_results_0_40.csv",  # 100%
    # "100_eda_augmented_results_30_10.csv",  # eda: 25%
    # "100_eda_augmented_results_20_20.csv",  # eda: 50%
    "100_eda_augmented_results_10_30.csv",  # eda: 75%
    # "100_eda_augmented_results_0_40.csv",  # eda: 100%

    # Medium Dataset
    "500_classic_results.csv",  # 0%
    # "500_self_augmented_results_30_10.csv",  # 25%
    # "500_self_augmented_results_20_20.csv",  # 50%
    # "500_self_augmented_results_16_24.csv",  # 60%
    # "500_self_augmented_results_14_26.csv",  # 65%
    # "500_self_augmented_results_12_28.csv",  # 70%
    "500_self_augmented_results_10_30.csv",  # 75%
    # "500_self_augmented_results_0_40.csv",  # 100%
    # "500_eda_augmented_results_30_10.csv",  # eda: 25%
    # "500_eda_augmented_results_20_20.csv",  # eda: 50%
    "500_eda_augmented_results_10_30.csv",  # eda: 75%
    # "500_eda_augmented_results_0_40.csv",  # eda: 100%
]

if len(FILES) == 0:
    FILES = [sys.argv[1]]

judge = Classifier()
judge.load_classifier(TRAINED_CLASSIFIER_PATH)

for file in FILES:
    samples = 0
    sarcasm = 0
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
