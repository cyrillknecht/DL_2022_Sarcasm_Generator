"""
    Finetune standard gpt-2 model with all sarcasm samples from the test set of
    ets-twitter-dataset and generate samples for evaluation.
    This model will be compared to the self augmented model by the judge.
"""
from helper import preprocessing, concat_last_context_rows
import gpt_2_simple as gpt2
import pandas as pd

# Training Parameters
MODEL = '124M'
STEPS = 480
GENERATE = 100

SARCASM_DATASET_PATH = 'dataset/ets_twitter_train_data_generator_only_sarcasm_subset_100.jsonl'
NON_SARCASM_DATASET_PATH = 'dataset/ets_twitter_train_data_generator_only_non_sarcasm.jsonl'
TRAIN_SET_PATH = 'dataset/temporary_train_data_generator.csv'
RESULT_PATH = 'generated/classic_results.csv'

# Get all sarcastic tweets from dataset for fine-tuning
train_data = pd.read_json(SARCASM_DATASET_PATH, lines=True)
sarcastic_tweets = train_data[train_data["label"] == "SARCASM"][['response', 'context']]

# Concatenate the last two context entries
sarcastic_tweets = concat_last_context_rows(sarcastic_tweets, 2)

# Preprocess data
sarcastic_tweets = preprocessing(sarcastic_tweets, 'context')
sarcastic_tweets = preprocessing(sarcastic_tweets, 'response')

# Save as txt file
with open(TRAIN_SET_PATH, 'w') as f:

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    for i, row in sarcastic_tweets.iterrows():
        f.write("<sc> " + row['context'] + " <ec> <sr> " + row['response'] + " <er>" + "\n")

# Get base model
gpt2.download_gpt2(model_name="124M")
sess = gpt2.start_tf_sess()

print(f"Start training for {STEPS} steps. ")

# Finetune the model
gpt2.finetune(sess,
              dataset=TRAIN_SET_PATH,
              model_name=MODEL,
              steps=STEPS,
              restore_from='fresh',
              run_name='run_classic',
              print_every=5,
              sample_every=100,
              save_every=STEPS,
              reuse=False
              )

print(f"Finished training. ")

# Generate tweets
# We use non-sarcastic samples as prompts
prompts = pd.read_json(NON_SARCASM_DATASET_PATH, lines=True) 
prompts = prompts[prompts["label"] == "NOT_SARCASM"][['context']]
# Concatenate the last two context entries
prompts = concat_last_context_rows(prompts, 2)
prompts = preprocessing(prompts, 'context')

# Generate tweets and write to output file
with open(RESULT_PATH, 'w') as f:

    print(f"Generating final outputs...")

    # Make header because the first line is skipped when using csv
    f.write("data\n")

    counter = 0
    for i, row in prompts.head(GENERATE).iterrows():
        prefix = "<sc> " + row['context'] + " <ec> <sr> "
        generated_tweet = gpt2.generate(sess, run_name='run_classic', return_as_list=True, length=128, prefix=prefix, nsamples=1, batch_size=1, truncate="<|endoftext|>")[0]
        generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ").replace(",", "").replace(";", "").strip()
        print(generated_tweet+"\n\n")
        f.write(generated_tweet + "\n")
        if counter % 5 == 0:
            print(f"Generated {counter} outputs ({counter/GENERATE*100:.2f}% done)...")
        counter += 1

print(f"Finished generating outputs. ")
