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
from simplet5 import SimpleT5
from sklearn.metrics import precision_score, f1_score, accuracy_score
from helper import split_data, prepare_train_datasets, prepare_test_dataset


class Classifier:
    """
        A classifier based on T5 that can be fine-tuned using different datasets.
    """

    def __init__(self, batch_size=4):
        self.classifier = SimpleT5()
        self.trained_classifier_path = None
        self.batch_size=batch_size

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
                              batch_size=self.batch_size,
                              max_epochs=max_epochs,
                              outputdir=output_dir,
                              use_gpu=use_gpu,
                              save_only_last_epoch=save_only_last_epoch,
                              early_stopping_patience_epochs=early_stopping_patience_epochs
                              )

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

            if index % 100 == 0:
                print(f"Processed {index/test_df.shape[0]*100:.2f}% of all samples.")

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


def train_classifiers(train_dataset_path, use_gpu=True, batch_size=4):
    """
        Fine-tune two classifiers using given dataset.
    """
    print("Preparing Datasets..")
    dataset_judge, dataset_classifier = prepare_train_datasets(train_dataset_path)

    print(f"Judge dataset size: {dataset_judge.shape[0]}; Classifier dataset size: {dataset_classifier.shape[0]}")

    print("Fine-tuning Judge..")
    judge_model = Classifier(batch_size=batch_size)
    judge_model.finetune(dataset_judge,
                         max_epochs=10,
                         early_stopping_patience_epochs=0,
                         save_only_last_epoch=False,
                         output_dir="models/judge",
                         use_gpu=use_gpu)

    print("Fine-tuning Classifier..")
    classifier_model = Classifier(batch_size=batch_size)
    classifier_model.finetune(dataset_classifier,
                              max_epochs=10,
                              early_stopping_patience_epochs=0,
                              save_only_last_epoch=False,
                              output_dir="models/classifier",
                              use_gpu=use_gpu)


def evaluation(trained_classifier_path, test_dataset_path, use_gpu=True, batch_size=4):
    """
        Print evaluation for a fine-tuned model.
    """
    print("Preparing test-dataset..")
    test_dataset = prepare_test_dataset(test_dataset_path, 2)

    print("Loading classifier..")
    classifier = Classifier(batch_size=batch_size)
    classifier.load_classifier(trained_classifier_path, use_gpu=use_gpu)

    print("Evaluating Classifier..")
    print(classifier.evaluate_classifier(test_dataset))

# When running this file, the two classifiers will be retrained
if __name__ == "__main__":
    #train_classifiers("dataset/sarcasm_detection_shared_task_twitter_training.jsonl", batch_size=4)

    # Comment out the line above and uncomment these two lines to run only the evaluation part
    #evaluation("models/judge", "dataset/sarcasm_detection_shared_task_twitter_testing.jsonl")
    #evaluation("models/self_augmented", "dataset/sarcasm_detection_shared_task_twitter_testing.jsonl")
    #for i in range(5, 10):
    #    print(f"####### Starting evaluation for judge model {i}... #######")
    #    evaluation("models/judge/simplet5-" + str(i), "dataset/sarcasm_detection_shared_task_twitter_testing.jsonl", batch_size=16)
    #    print(f"####### Finished evaluation for judge model {i} #######")

    for i in range(2, 10):
        print(f"####### Starting evaluation for classifier model {i}... #######")
        evaluation("models/classifier/simplet5-" + str(i), "dataset/sarcasm_detection_shared_task_twitter_testing.jsonl", batch_size=8)
        print(f"####### Finished evaluation for classifier model {i} #######")
