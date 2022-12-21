"""
    Finetune standard gpt-2 model with a mix of
    This model will be compared to classical finetuned model by the judge.
"""
from helper import preprocessing, concat_last_context_rows
from classifier import Classifier
import gpt_2_simple as gpt2
import pandas as pd

# Training options
MODEL = '124M'
EPOCHS = 2
STEPS_PER_EPOCH = 10
GENERATE = 100

# Settings for dataset / generated samples mix
DATASET_SAMPLES_INITIAL = 100
DATASET_SAMPLES_PER_EPOCH = 20
GENERATED_SAMPLES_PER_EPOCH = 20

# Path variables
DATASET_PATH = 'dataset/ets_twitter_train_data_generator.jsonl'
TRAIN_SET_PATH = 'dataset/ets_twitter_train_data_generator.txt'
RESULT_PATH = 'generated/self_augmented_results.txt'
CLASSIFIER_PATH = 'models/self_augmented'
FILES = ['generated/classic_finetuned_results.txt']

# Get all sarcastic tweets from dataset
train_data = pd.read_json(DATASET_PATH, lines=True)
all_sarcastic_tweets = train_data[train_data["label"] == "SARCASM"][['response', 'context']]

# Concatenate the last two context entries
all_sarcastic_tweets = concat_last_context_rows(all_sarcastic_tweets, 2)

# Preprocess data
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'context')
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'response')

# Save as txt file
with open(TRAIN_SET_PATH, 'w') as f:
    for row in all_sarcastic_tweets.head(DATASET_SAMPLES_INITIAL).iterrows():
        f.write(row['context'] + row['response'] + "\n")

# Get base model
gpt2.download_gpt2(model_name="124M")
sess = gpt2.start_tf_sess()

# Init classifier for judging generated tweets
judge = Classifier()
judge.load_classifier(CLASSIFIER_PATH)

# Get prompts for training
prompts = train_data[train_data["label"] == "NOT_SARCASM"][['context']]
# Concatenate the last two context entries
prompts = concat_last_context_rows(prompts, 2)
prompts = preprocessing(prompts, 'context')
promptIndex = 0

# Use fresh model for first epoch
restore_from = 'fresh'

# Finetuning epochs
promptIndex = 0
for epoch in range(EPOCHS):
    nGenerated = 0
    gpt2.finetune(sess,
                  dataset=TRAIN_SET_PATH,
                  model_name=MODEL,
                  steps=STEPS_PER_EPOCH,
                  restore_from=restore_from,
                  run_name='run',
                  print_every=10,
                  sample_every=200,
                  save_every=STEPS_PER_EPOCH,
                  reuse=False
                  )

    # Always use same model for subsequent epochs
    restore_from = 'latest'

    # Create new dataset for next epoch
    with open(TRAIN_SET_PATH, 'w') as f:
        # Take samples from dataset
        startIndex = DATASET_SAMPLES_INITIAL + epoch * DATASET_SAMPLES_PER_EPOCH
        for sample in range(DATASET_SAMPLES_PER_EPOCH):
            row = all_sarcastic_tweets.iloc[startIndex + sample]
            f.write(row['context'] + row['response'] + "\n")

        # Generate new training samples
        while nGenerated < GENERATED_SAMPLES_PER_EPOCH:
            generated_tweet = gpt2.generate(sess, run_name='run', return_as_list=True, length=128, prefix=row['context'])[0]
            generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ")

            if judge.classify_tweet(generated_tweet):
                # Only take it if answers are classified as sarcasm
                nGenerated += 1
                f.write(generated_tweet+"\n")

                promptIndex += 1

                if promptIndex >= len(prompts):
                    promptIndex = 0

# Generate tweets and write to output file
with open(RESULT_PATH, 'w') as f:
    for i, row in prompts.head(GENERATE).iterrows():
        generated_tweet = gpt2.generate(sess, run_name='run', return_as_list=True, length=128, prefix=row['context'])[0]
        generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ")
        f.write(generated_tweet + "\n")
