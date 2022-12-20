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
def split_data(df: pd.DataFrame, percentage_second: int) -> (pd.DataFrame, pd.DataFrame):
    """
    Split a dataframe into two equal dataframes.

    Args:
        df (pd.DataFrame): Dataframe to split.
        percentage_second (int): Percentage of the second dataframe.

    Returns:
        (pd.DataFrame, pd.DataFrame): Two dataframes with equal size balanced on the target_text column.
    """
    df_1_text, df_2_text, df_1_labels, df_2_labels = train_test_split(df['source_text'].tolist(),
                                                                      df['target_text'].tolist(),
                                                                      shuffle=True,
                                                                      test_size=percentage_second / 100,
                                                                      random_state=42,
                                                                      stratify=df['target_text'])
    df_1 = pd.DataFrame({'source_text': df_1_text, 'target_text': df_1_labels})
    df_2 = pd.DataFrame({'source_text': df_2_text, 'target_text': df_2_labels})

    return df_1, df_2


def prepare_train_datasets(train_dataset_path: str, num_context_tweets: int = 2, additional_entries: bool = True) \
        -> (pd.DataFrame, pd.DataFrame):
    """
    Prepare the training sets.

    Split data into two datasets.
    Augment the training data with additional entries.
    Add context tweets to the training data.

    Args:
        train_dataset_path (str): Path to the training dataset.
        num_context_tweets (int): Number of context tweets to use for each entry.
        additional_entries (bool): Whether to augment the training data with additional entries.

    Returns: (pd.DataFrame, pd.DataFrame): Two augmented dataframes from the training set with equal size balanced on
    the target_text column.

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
    train_df_1, train_df_2 = split_data(df, 50)

    return train_df_1, train_df_2


def prepare_test_dataset(path: str, num_context_tweets: int = 2) -> pd.DataFrame:
    """
    Prepare the test set.

    Add context tweets to the test data.

    Args:
        path (str): Path to the test dataset.
        num_context_tweets (int): Number of context tweets to use for each entry.

    Returns:
        pd.DataFrame: Augmented dataframe from the test set.

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
        """
        Initialize a classifier.
        """
        self.classifier = SimpleT5()
        self.trained_classifier_path = None

    def finetune(self,
                 train_df: pd.DataFrame,
                 max_epochs: int = 10,
                 early_stopping_patience_epochs: int = 0,
                 output_dir: str = "outputs",
                 save_only_last_epoch: bool = False,
                 use_gpu: bool = True):
        """
        Finetune a T5 model. Resulting models can be found in output_dir.

        Args:
            train_df (pd.DataFrame): Training dataset.
            max_epochs (int): Maximum number of epochs to train.
            early_stopping_patience_epochs (int): Number of epochs to wait for early stopping.
            output_dir (str): Path to the output directory.
            save_only_last_epoch (bool): Whether to save only the last epoch.
            use_gpu (bool): Whether to use the GPU.

        """
        train_data, eval_data = split_data(train_df, 5)
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

    def load_classifier(self, trained_classifier_path: str, use_gpu: bool = True):
        """
        Load a previously trained classifier.

        Args:
            trained_classifier_path (str): Path to the trained classifier.
            use_gpu (bool): Whether to use the GPU.

        """
        self.trained_classifier_path = trained_classifier_path
        self.classifier.load_model("t5", trained_classifier_path, use_gpu=use_gpu)

    def evaluate_classifier(self, test_df: pd.DataFrame) -> (str, str, str) or (None, None, None):
        """
        Evaluate a classifier returning a report stating f1-score, precision and accuracy for the given classifier.
        Make sure model is loaded before invoking.

        Args:
            test_df (pd.DataFrame): Test dataset.

        Returns: (str, str, str) or (None, None, None): Report stating f1-score, precision and accuracy for the given
        classifier. Returns Nones if no classifier is loaded.

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

    def classify_tweet(self, conversation: str) -> bool or None:
        """
        Classifies a tweet/conversation. Returns true for detected sarcasm.
        Make sure model is loaded before invoking.

        Args:
            conversation (str): Conversation to classify.

        Returns:
            bool or None: True for detected sarcasm. None if no classifier is loaded.

        """
        if self.trained_classifier_path is None:
            print("Please load a model first.")
            return None

        prediction = self.classifier.predict(conversation)[0]
        return True if prediction == "SARCASM" else False


def train_classifiers(train_dataset_path: str, use_gpu: bool = True):
    """
    Fine-tune two classifiers using given dataset.

    Args:
        train_dataset_path (str): Path to the training dataset.
        use_gpu (bool): Whether to use the GPU.

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


def evaluation(trained_classifier_path: str, test_dataset_path: str, use_gpu: bool = True):
    """
    Print evaluation for a fine-tuned model.

    Args:
        trained_classifier_path (str): Path to the trained classifier.
        test_dataset_path (str): Path to the test dataset.
        use_gpu (bool): Whether to use the GPU.

    """
    print("Preparing test-dataset..")
    test_dataset = prepare_test_dataset(test_dataset_path, 2)

    print("Loading classifier..")
    classifier = Classifier()
    classifier.load_classifier(trained_classifier_path, use_gpu=use_gpu)

    print("Evaluating Classifier..")
    print(classifier.evaluate_classifier(test_dataset))
