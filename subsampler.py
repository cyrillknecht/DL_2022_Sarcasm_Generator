"""
    This script takes NUM_NEW_ROWS random rows from the dataset and writes them to a new file.
"""

from numpy.random import default_rng

NUM_NEW_ROWS = 500
INFILE_PATH = "dataset/ets_twitter_train_data_generator_only_sarcasm.jsonl"
OUTFILE_PATH = f"dataset/ets_twitter_train_data_generator_only_sarcasm_subset_{NUM_NEW_ROWS}.jsonl"

row_count = 0
with open(INFILE_PATH, 'r') as f:
    row_count = sum(1 for row in f)

assert NUM_NEW_ROWS < row_count, f"Cannot take {NUM_NEW_ROWS} rows from a dataset with only {row_count} rows."

rng = default_rng()
numbers = rng.choice(row_count, size=NUM_NEW_ROWS, replace=False)

print(f"Turning {row_count} rows into {NUM_NEW_ROWS} rows...")
print(f"Selected rows:\n{numbers}")

counter = 0
with open(OUTFILE_PATH, 'w') as outfile:
    with open(INFILE_PATH, 'r') as infile:
        for line in infile:
            if counter in numbers:
                outfile.write(line)
            counter += 1
        



