# Fine-tuned Classifiers

In this directory one can find two fine-tuned classifiers, generated using the functionality
provided in *Classifiers.py*.

### Training
Classifier 1 was trained on *first_half.jsonl*
and Classifier 2 on *second_half.jsonl*.

Those datasets are splits with equal size of the twitter training set
provided by [ETS](https://github.com/EducationalTestingService/sarcasm).

### Evaluation
Both classifiers were evaluated on *test_data.jsonl*. 
Which is just a processed version of the twitter test set
provided by [ETS](https://github.com/EducationalTestingService/sarcasm).

### Results
To reproduce the results using the functionality of *Classifier.py* please use the original 
[ETS](https://github.com/EducationalTestingService/sarcasm) twitter datasets. They will be pre-processed
to the form shown in *first_half.jsonl*, *second_half.jsonl*, *test_data.jsonl*.

|              | Accuracy | Precision | F1   |
|--------------|----------|-----------|------|
| Classifier 1 | 0.72     | 0.72      | 0.72 |
| Classifier 2 | 0.73     | 0.73      | 0.73 |