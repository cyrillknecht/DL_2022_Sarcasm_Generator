"""
    Finetune standard gpt-2 model with a mix of regular samples and EDA-augmented samples
    This model will be compared to classical finetuned model by the judge.
"""
from helper import preprocessing, concat_last_context_rows
from classifier import Classifier
import gpt_2_simple as gpt2
import pandas as pd
from eda import eda

# Training options
MODEL = '124M'
EPOCHS = 10
STEPS_PER_EPOCH = 100
GENERATE = 100

# Settings for dataset / generated samples mix 
# (NOTE: Per "epoch", a total of DATSET_SAMPLES_PER_EPOCH + AUGMENTED_SAMPLES_PER_EPOCH will be generated)
DATASET_SAMPLES_INITIAL = 100
DATASET_SAMPLES_PER_EPOCH = 20
AUGMENTED_SAMPLES_PER_EPOCH = 20

# Path variables
DATASET_PATH = 'dataset/ets_twitter_train_data_generator.jsonl'
TRAIN_SET_PATH = 'dataset/ets_twitter_train_data_generator.csv'
RESULT_PATH = 'generated/self_augmented_results.txt'
FILES = ['generated/classic_finetuned_results.txt']

# Get all sarcastic tweets from dataset
train_data = pd.read_json(DATASET_PATH, lines=True)
all_sarcastic_tweets = train_data[train_data["label"] == "SARCASM"][['response', 'context']]

# Concatenate the last two context entries
all_sarcastic_tweets = concat_last_context_rows(all_sarcastic_tweets, 2)

# Preprocess data
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'context')
all_sarcastic_tweets = preprocessing(all_sarcastic_tweets, 'response')

# Save as csv file
with open(TRAIN_SET_PATH, 'w') as f:

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    for i, row in all_sarcastic_tweets.head(DATASET_SAMPLES_INITIAL).iterrows():
        f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")

current_number_of_samples = DATASET_SAMPLES_INITIAL

# Get prompts for training
prompts = train_data[train_data["label"] == "NOT_SARCASM"][['context']]

# Concatenate the last two context entries
prompts = concat_last_context_rows(prompts, 2)
prompts = preprocessing(prompts, 'context')

# Get base model
gpt2.download_gpt2(model_name="124M")
sess = gpt2.start_tf_sess()

# Use fresh model for first epoch
restore_from = 'fresh'

# Finetuning epochs
for epoch in range(EPOCHS):
    # Reset session
    gpt2.reset_session(sess)
    sess = gpt2.start_tf_sess()

    nGenerated = 0
    gpt2.finetune(sess,
                  dataset=TRAIN_SET_PATH,
                  model_name=MODEL,
                  steps=current_number_of_samples,
                  restore_from=restore_from,
                  run_name='run_eda_augmentation',
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
        startIndex = DATASET_SAMPLES_INITIAL + epoch * (DATASET_SAMPLES_PER_EPOCH + AUGMENTED_SAMPLES_PER_EPOCH)
        
        # Non-augmented samples
        for sample in range(DATASET_SAMPLES_PER_EPOCH): 
            row = all_sarcastic_tweets.iloc[startIndex + sample]
            f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")
            current_number_of_samples += 1
        
        print(f"Augmenting {AUGMENTED_SAMPLES_PER_EPOCH} samples...")
        
        # Augmented samples
        for sample in range(AUGMENTED_SAMPLES_PER_EPOCH): 
            row = all_sarcastic_tweets.iloc[startIndex + DATASET_SAMPLES_PER_EPOCH + sample]
            #print(f"### Sample prior to augmentation: {row['response']}\n")
            row['response'] = eda(row['response'], num_aug=1)[0]
            #print(f"### Sample after augmentation: {row['response']}\n")
            f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")

            current_number_of_samples += 1
                
        print(f"Sample augmentation done!")

print(f"Finished training. ")

# Generate tweets and write to output file
with open(RESULT_PATH, 'w') as f:

    print(f"Generating final outputs...")

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    for i, row in prompts.head(GENERATE).iterrows():
        prefix = "<sc> " + row['context'] + " <ec> <sr> "
        generated_tweet = gpt2.generate(sess, run_name='run_self_augmentation', return_as_list=True, length=128, prefix=prefix, truncate="<|endoftext|>")[0]
        generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ")
        f.write(generated_tweet + "\n")
        if i % 10 == 0:
            print(f"Generated {i} outputs ({i/GENERATE*100:2f}% done)...")

print(f"Finished generating outputs.")
