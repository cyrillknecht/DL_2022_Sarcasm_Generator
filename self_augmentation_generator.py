"""
    Finetune standard gpt-2 model with a mix of regular samples and self-generated samples
    This model will be compared to classical finetuned model by the judge.
"""
from helper import preprocessing, concat_last_context_rows
from classifier import Classifier
import gpt_2_simple as gpt2
import pandas as pd

import tensorflow as tf
import os
tf.get_logger().setLevel('ERROR')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Training options
MODEL = '124M'
EPOCHS = 10
#STEPS_PER_EPOCH = 50
GENERATE = 100

# Settings for dataset / generated samples mix
# (NOTE: Per "epoch", a total of DATSET_SAMPLES_PER_EPOCH + GENERATED_SAMPLES_PER_EPOCH will be generated)
DATASET_SAMPLES_INITIAL = 100
DATASET_SAMPLES_PER_EPOCH = 20
GENERATED_SAMPLES_PER_EPOCH = 20
MAX_GENERATOR_TRIES = 5*GENERATED_SAMPLES_PER_EPOCH
GENERATION_BATCH_SIZE = 4

# Path variables
DATASET_PATH = 'dataset/ets_twitter_train_data_generator_subset_100.jsonl'
TRAIN_SET_PATH = 'dataset/ets_twitter_train_data_generator.csv'
RESULT_PATH = 'generated/self_augmented_results.csv'
CLASSIFIER_PATH = 'models/classifier'
FILES = ['generated/classic_finetuned_results.txt']

# Get all sarcastic tweets from dataset
train_data = pd.read_json(DATASET_PATH, lines=True)
all_sarcastic_tweets = train_data[train_data["label"] == "SARCASM"][['response', 'context']]


# Concatenate the last two context entries
all_sarcastic_tweets = concat_last_context_rows(all_sarcastic_tweets, 2)

# Preprocess data
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'context')
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'response')

NUM_SARCASTIC_TWEETS = all_sarcastic_tweets.shape[0]

# Save as csv file
with open(TRAIN_SET_PATH, 'w') as f:

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    for i, row in all_sarcastic_tweets.head(DATASET_SAMPLES_INITIAL).iterrows():
        f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")

current_number_of_samples = DATASET_SAMPLES_INITIAL

# Init classifier for judging generated tweets
classifier = Classifier()
classifier.load_classifier(CLASSIFIER_PATH)

# Get prompts for training
prompts = train_data[train_data["label"] == "NOT_SARCASM"][['context']]
# Concatenate the last two context entries
prompts = concat_last_context_rows(prompts, 2)
prompts = preprocessing(prompts, 'context')

NUM_NON_SARCASTIC_TWEETS = prompts.shape[0]

promptIndex = 0

# Get base model
gpt2.download_gpt2(model_name="124M")
sess = gpt2.start_tf_sess()

# Use fresh model for first epoch
restore_from = 'fresh'

# Finetuning epochs
promptIndex = 0
for epoch in range(EPOCHS):
    # Reset session
    gpt2.reset_session(sess)
    sess = gpt2.start_tf_sess()

    print(f"Running {current_number_of_samples} iterations on {current_number_of_samples} samples...")

    
    gpt2.finetune(sess,
                  dataset=TRAIN_SET_PATH,
                  model_name=MODEL,
                  steps=current_number_of_samples,
                  restore_from=restore_from,
                  run_name='run_self_augmentation',
                  print_every=5,
                  batch_size=1,
                  sample_every=100,
                  save_every=current_number_of_samples,
                  reuse=False,
                  use_memory_saving_gradients=False #unfortunately doesn't work in Tensorflow 2
                  )

    # Always use same model for subsequent epochs
    restore_from = 'latest'

    current_number_of_samples = 0

    # Create new dataset for next epoch
    with open(TRAIN_SET_PATH, 'w') as f:

        # Make header because the first line is skipped when using csv
        f.write("data\n")

        # Take samples from dataset
        startIndex = DATASET_SAMPLES_INITIAL + epoch * DATASET_SAMPLES_PER_EPOCH
        for sample in range(DATASET_SAMPLES_PER_EPOCH):
            row = all_sarcastic_tweets.iloc[(startIndex + sample) % NUM_SARCASTIC_TWEETS]
            f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")
            current_number_of_samples += 1
            
        nGenerated = 0
        generatorTries = 0

        print(f"Generating new samples...")

        # Generate new training samples
        while nGenerated < GENERATED_SAMPLES_PER_EPOCH and generatorTries < MAX_GENERATOR_TRIES:
        
            # Kinda stupid: we can only supply a single prefix even when generating multiple samples...
            prefix = "<sc> "+ prompts.iloc[promptIndex % NUM_NON_SARCASTIC_TWEETS]['context'] + " <ec> <sr> "
            generated_tweets = gpt2.generate(sess, run_name='run_self_augmentation', return_as_list=True, length=128, prefix=prefix, nsamples=GENERATION_BATCH_SIZE, batch_size=GENERATION_BATCH_SIZE, truncate="<|endoftext|>")

            for generated_tweet in generated_tweets:

                #print(f"Context: {prefix}")
                #print(f"Generated:\n{generated_tweet}\n\n")
            
                #generated_tweet = re.sub("\r|\n|,|;|  +", " ", generated_tweet)
                generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ").replace(",", "").replace(";", "").strip()

                if classifier.classify_tweet(generated_tweet):
                    # Only take it if answers are classified as sarcasm
                    f.write(generated_tweet+"\n")
                    nGenerated += 1
                    current_number_of_samples += 1
                
                generatorTries += 1
                
                if nGenerated >= GENERATED_SAMPLES_PER_EPOCH: # Prevent generating more than GENERATED_SAMPLES_PER_EPOCH samples
                    break
                    
            # Since GENERATION_BATCH_SIZE samples are generated with the same prefix, 
            # we only increase the promptIndex _after_ going over all of the generated tweets
            promptIndex += 1

            if promptIndex >= len(prompts):
                promptIndex = 0

        print(f"Finished sample generation. {nGenerated} generated samples were classified as 'sarcastic' (out of {generatorTries} generated samples)")

print(f"Finished training. ")

# Generate tweets and write to output file
with open(RESULT_PATH, 'w') as f:

    print(f"Generating final outputs...")

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    counter = 0
    for i, row in prompts.head(GENERATE).iterrows(): # Note: i is the _original_ row number, which means i is not just a "counter variable"
        prefix = "<sc> " + row['context'] + " <ec> <sr> "
        generated_tweet = gpt2.generate(sess, run_name='run_self_augmentation', return_as_list=True, length=128, prefix=prefix, nsamples=1, batch_size=1, truncate="<|endoftext|>")[0]
        generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ").replace(",", "").replace(";", "").strip()
        f.write(generated_tweet + "\n")
        if counter % 5 == 0:
            print(f"Generated {counter} outputs ({counter/GENERATE*100:2f}% done)...")
        counter += 1

print(f"Finished generating outputs. ")
