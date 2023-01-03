# Self-Augmentation Network for Sarcasm Generation with Synthetic Data
### Deep Learning Project AS 2022, ETH Zürich
Cyrill Knecht, Joel Neuner-Jehle, Justin Studer, David Zehnder

#### Project Description
This is the repository of our proposed sarcasm generation model using self-augmentation. It uses a pretrained discriminator that can distinguish between sarcasm and non-sarcasm to indirectly provide the generator with feedback on the quality of its outputs during training. Outputs classified as sarcasm with high confidence will be fed back to the generator for further training. The performance of our proposed model is measured for different fractions of mixing self-augmented data with real data and compared to the baseline performance of classic model fine-tuning with and without state-of-the-art data augmentation.

#### How to run the code
We provided a reusable ExperimentRunner.ipynb file, runnable e.g. via Google Colaboratory (Pro is required for sufficient resources). This notebook can be used to reproduce all our experiments.

Before running the experiments, additional pretrained models need to be downloaded via provided Google Drive folders. See "/models/classifier/" and "/models/judge/" for respective download links.
