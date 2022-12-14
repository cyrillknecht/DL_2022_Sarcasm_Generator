import pandas as pd
from simplet5 import SimpleT5
from sklearn.metrics import precision_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split


class Classifier:
    """
    A classifier based on T5 that can be finetuned using different datasets.
    """

    def __init__(self):
        self.classifier = SimpleT5()
        self.dataset_path = None
        self.train_data = None
        self.test_data = None
        self.trained_classifier_path = None

    def load_dataset(self, dataset_path):
        self.dataset_path = dataset_path
        dataset = pd.read_json(self.dataset_path, lines=True)

        X_train, X_test, y_train, y_test = train_test_split((dataset['context'] + dataset['text']).tolist(), dataset['label'].tolist(),
                                    shuffle=True, test_size=0.05, random_state=42, stratify=dataset['label'])
      
        self.train_data = pd.DataFrame({'source_text': X_train, 'target_text': y_train})
        self.test_data = pd.DataFrame({'source_text': X_test, 'target_text': y_test})

        return

    def finetune(self, max_epochs=5, early_stopping_patience_epochs=2, output_dir="outputs"):
        if self.train_data is None:
            print("Please load the dataset first.")
            return

        self.classifier.from_pretrained(model_type="t5", model_name="t5-base")
        self.classifier.train(train_df=self.train_data,
                              eval_df=self.test_data,
                              source_max_token_len=300,
                              target_max_token_len=200,
                              batch_size=8,
                              max_epochs=max_epochs,
                              outputdir=output_dir,
                              use_gpu=True,
                              save_only_last_epoch=False,
                              early_stopping_patience_epochs=early_stopping_patience_epochs
                              )

        return

    def evaluate_classifier(self, trained_classifier_path):
        self.trained_classifier_path = trained_classifier_path

        self.classifier.load_model("t5", self.trained_classifier_path, use_gpu=True)

        predictions = []
        for index, row in self.test_data.iterrows():
            prediction = self.classifier.predict(row['source_text'])[0]
            predictions.append(prediction)

        result_df = self.test_data.copy()
        result_df['predicted'] = predictions
        result_df['original'] = result_df['target_text']

        f1 = f"F1: {f1_score(result_df['original'], result_df['predicted'], average='macro')}"
        precision = f"Precision: {precision_score(result_df['original'], result_df['predicted'], average='macro')}"
        accuracy = f"Accuracy: {accuracy_score(result_df['original'], result_df['predicted'])}"

        return f1, precision, accuracy

    def classify_tweet(self, conversation, trained_classifier_path):
        self.trained_classifier_path = trained_classifier_path

        self.classifier.load_model("t5", self.trained_classifier_path, use_gpu=False)
        prediction = self.classifier.predict(conversation)[0]

        return True if prediction == "SARCASM" else False

