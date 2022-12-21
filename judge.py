"""
    Judges the quality of generated answers by an independent classifier
"""
from classifier import Classifier

TRAINED_CLASSIFIER_PATH = 'models/judge'
FOLDER = 'generated'
FILES = ['classic_results.txt', 'self_augmented_results.txt']

judge = Classifier()
judge.load_classifier(TRAINED_CLASSIFIER_PATH)

for file in FILES:
    samples = 0
    sarcasm = 0
    with open(FOLDER+"/"+file) as f:
        for line in f:
            samples += 1
            if judge.classify_tweet(line):
                sarcasm += 1

    print("Results for " + file)
    print("Total Samples: " + str(samples))
    print("Classified as sarcasm: " + str(sarcasm))
    print("Sarcasm percentage: " + str(samples / sarcasm))
    print()
