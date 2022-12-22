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
STEPS = 100
GENERATE = 100
DATASET_PATH = 'dataset/ets_twitter_train_data_generator.jsonl'
TRAIN_SET_PATH = 'dataset/ets_twitter_train_data_generator.txt'
RESULT_PATH = 'generated/classic_results.txt'

# Get all sarcastic tweets from dataset for fine-tuning
train_data = pd.read_json(DATASET_PATH, lines=True)
sarcastic_tweets = train_data[train_data["label"] == "SARCASM"][['response', 'context']]

# Concatenate the last two context entries
sarcastic_tweets = concat_last_context_rows(sarcastic_tweets, 2)

# Preprocess data
sarcastic_tweets = preprocessing(sarcastic_tweets, 'context')
sarcastic_tweets = preprocessing(sarcastic_tweets, 'response')

# Save as txt file
with open(TRAIN_SET_PATH, 'w') as f:
    for i, row in sarcastic_tweets.iterrows():
        f.write(row['context'] + row['response'] + "\n")

# Get base model
gpt2.download_gpt2(model_name="124M")
sess = gpt2.start_tf_sess()

# Finetune the model
gpt2.finetune(sess,
              dataset=TRAIN_SET_PATH,
              model_name=MODEL,
              steps=STEPS,
              restore_from='fresh',
              run_name='run_classic',
              print_every=10,
              sample_every=200,
              save_every=STEPS,
              reuse=False
              )

# Generate tweets
# We use non-sarcastic samples as prompts
prompts = train_data[train_data["label"] == "NOT_SARCASM"][['context']]

# Concatenate the last two context entries
prompts = concat_last_context_rows(prompts, 2)
prompts = preprocessing(prompts, 'context')

# Generate tweets and write to output file
with open(RESULT_PATH, 'w') as f:
    for i, row in prompts.head(GENERATE).iterrows():
        generated_tweet = gpt2.generate(sess, run_name='run_classic', return_as_list=True, length=128, prefix=row['context'])[0]
        generated_tweet = generated_tweet.replace("\r", " ").replace("\n", " ")
        f.write(generated_tweet + "\n")
