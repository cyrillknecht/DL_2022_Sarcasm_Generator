"""
Train two classifiers for use in the Self-Augmentation Sarcasm Generation Pipeline.
Classifiers are built on T5 model.

Inspired by: https://www.kaggle.com/code/evilmage93/t5-finetuning-on-sentiment-classification

Example Scripts using Classifier:
TRAIN_DATASET_PATH = 'YOUR_TRAIN_SET'
TEST_DATASET_PATH = 'YOUR_TEST_SET'
TRAINED_CLASSIFIER_PATH = 'YOUR_CLASSIFIER'

# Train judge and classifier
train_classifiers(TRAIN_DATASET_PATH)

# Evaluate a classifier
evaluation(TRAINED_CLASSIFIER_PATH, TEST_DATASET_PATH)

# Classify a conversation
judge = Classifier()
judge.load_classifier(TRAINED_CLASSIFIER_PATH)
print(judge.classify_tweet("Hello World"))

"""
import pandas as pd
from simplet5 import SimpleT5
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split


# Helper functions:

def split_data(df, percentage_second):
    """
    Split a dataframe in two parts.
    """
    df_1_text, df_2_text, df_1_labels, df_2_labels = train_test_split(df['source_text'].tolist(),
                                                                      df['target_text'].tolist(),
                                                                      shuffle=True,
                                                                      test_size=percentage_second,
                                                                      random_state=42,
                                                                      stratify=df['target_text'])
    df_1 = pd.DataFrame({'source_text': df_1_text, 'target_text': df_1_labels})
    df_2 = pd.DataFrame({'source_text': df_2_text, 'target_text': df_2_labels})

    return df_1, df_2


def prepare_train_datasets(train_dataset_path, num_context_tweets=2, additional_entries=True):
    """
    Split a dataframe into two equal dataframes. Augment the data if necessary.
    """

    df = pd.read_json(train_dataset_path, lines=True)

    # Augment dataset with additional entries using only context without the corresponding sarcastic response
    if additional_entries:
        sarcastic_df = df[df['label'] == 'SARCASM'].copy()
        sarcastic_df = sarcastic_df.assign(response='')
        sarcastic_df = sarcastic_df.assign(label="NOT_SARCASM")
        df = df.append(sarcastic_df)

    # Add the context-tweets to the training data
    df['source_text'] = df['context'].apply(lambda x: ' '.join(map(str, x[-num_context_tweets:])))
    df['source_text'] = df['source_text'] + df['response']
    df['target_text'] = df['label']

    # Split the dataset into two  equal sets
    train_df_1, train_df_2 = split_data(df, 0.5)

    return train_df_1, train_df_2


def prepare_test_dataset(path, num_context_tweets=2):
    """
    Prepare the test set.
    """
    df = pd.read_json(path, lines=True)

    # Add the context-tweets to the training data
    df['source_text'] = df['context'].apply(lambda x: ' '.join(map(str, x[-num_context_tweets:])))
    df['source_text'] = df['source_text'] + df['response']
    df['target_text'] = df['label']

    test_df = df.drop(labels=['context', 'response', 'id', 'label'], axis=1)

    return test_df


class Classifier:
    """
    A classifier based on T5 that can be fine-tuned using different datasets.
    """

    def __init__(self):
        self.classifier = SimpleT5()
        self.trained_classifier_path = None

    def finetune(self,
                 train_df,
                 max_epochs=5,
                 early_stopping_patience_epochs=2,
                 output_dir="outputs",
                 save_only_last_epoch=False,
                 use_gpu=True):
        """
        Finetune a T5 model. Resulting models can be found in output_dir.
        """
        train_data, eval_data = split_data(train_df, 0.05)
        self.classifier.from_pretrained(model_type="t5", model_name="t5-base")
        self.classifier.train(train_df=train_data,
                              eval_df=eval_data,
                              source_max_token_len=300,
                              target_max_token_len=200,
                              batch_size=8,
                              max_epochs=max_epochs,
                              outputdir=output_dir,
                              use_gpu=use_gpu,
                              save_only_last_epoch=save_only_last_epoch,
                              early_stopping_patience_epochs=early_stopping_patience_epochs
                              )

        return

    def load_classifier(self, trained_classifier_path, use_gpu=True):
        """
        Load a previously trained classifier.
        """
        self.trained_classifier_path = trained_classifier_path
        self.classifier.load_model("t5", trained_classifier_path, use_gpu=use_gpu)

    def evaluate_classifier(self, test_df):
        """
        Evaluate a classifier returning a report stating f1-score, precision and accuracy for the given classifier.
        Make sure model is loaded before invoking.
        """
        if self.trained_classifier_path is None:
            print("Please load a model first.")
            return None, None, None

        predictions = []
        for index, row in test_df.iterrows():
            prediction = self.classifier.predict(row['source_text'])[0]
            predictions.append(prediction)

        result_df = test_df.copy()
        result_df['predicted'] = predictions
        result_df['original'] = result_df['target_text']

        f1 = f"F1: {f1_score(result_df['original'], result_df['predicted'], average='macro')}"
        precision = f"Precision: {precision_score(result_df['original'], result_df['predicted'], average='macro')}"
        accuracy = f"Accuracy: {accuracy_score(result_df['original'], result_df['predicted'])}"

        return f1, precision, accuracy

    def classify_tweet(self, conversation):
        """
        Classifies a tweet/conversation. Returns true for detected sarcasm.
        Make sure model is loaded before invoking.
        """
        if self.trained_classifier_path is None:
            print("Please load a model first.")
            return None

        prediction = self.classifier.predict(conversation)[0]
        return True if prediction == "SARCASM" else False


def train_classifiers(train_dataset_path, use_gpu=True):
    """
    Fine-tune two classifiers using given dataset.
    """
    print("Preparing Datasets..")
    dataset_judge, dataset_classifier = prepare_train_datasets(train_dataset_path)

    print("Fine-tuning Classifier..")
    judge_model = Classifier()
    judge_model.finetune(dataset_judge,
                         max_epochs=8,
                         early_stopping_patience_epochs=0,
                         save_only_last_epoch=True,
                         output_dir="Judges",
                         use_gpu=use_gpu)

    print("Fine-tuning Judge..")
    classifier_model = Classifier()
    classifier_model.finetune(dataset_classifier,
                              max_epochs=9,
                              early_stopping_patience_epochs=0,
                              save_only_last_epoch=True,
                              output_dir="Classifiers",
                              use_gpu=use_gpu)

    return


def evaluation(trained_classifier_path, test_dataset_path, use_gpu=True):
    """
    Print evaluation for a fine-tuned model.
    """
    print("Preparing test-dataset..")
    test_dataset = prepare_test_dataset(test_dataset_path, 2)

    print("Loading classifier..")
    classifier = Classifier()
    classifier.load_classifier(trained_classifier_path, use_gpu=use_gpu)

    print("Evaluating Classifier..")
    print(classifier.evaluate_classifier(test_dataset))

    return
