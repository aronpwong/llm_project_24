# LLM Project

## Project Task
The project topic that I chose was sentiment analysis using the idmb dataset. Sentiment analysis will allow me to understand context-specific sentiments and slangs. Using the idmb dataset with a pre-trained language model, the goal is to classify reviews as either positive or negative.

## Dataset
The dataset is called IMDB Movie Reviews and contains 50,000 reviews. 25,000 are for training and 25,000 for testing. You can find the dataset here: [https://huggingface.co/datasets/stanfordnlp/imdb]

## Pre-trained Model
The pre-trained model used is DistilBERT which is a distilled version of BERT. It is a smaller, faster, cheaper and lighter version that run 60% faster while perserving 95% of BERT's performances.

## Performance Metrics
The performance metrics for this model will be evaluated with the following metrics: Accuracy, Precision, Recall and F1 Score.

The results for the datasets can be viewed in the following notebooks: 2-representation and 3-pre-trained-models.

TF-IDF Models
Logistic Regression:
Accuracy: 0.879
Precision: 0.878
Recall: 0.881
F1 Score: 0.879

Random Forest:
Accuracy: 0.847
Precision: 0.858
Recall: 0.831
F1 Score: 0.844

Gradient Boosting:
Accuracy: 0.808
Precision: 0.778
Recall: 0.861
F1 Score: 0.817

Support Vector Machine (SVM):
Accuracy: 0.875
Precision: 0.882
Recall: 0.867
F1 Score: 0.874

BoW Model
Logistic Regression:
Accuracy: 0.879
Precision: 0.878
Recall: 0.881
F1 Score: 0.879

Random Forest:
Accuracy: 0.847
Precision: 0.858
Recall: 0.831
F1 Score: 0.844

Gradient Boosting:
Accuracy: 0.808
Precision: 0.778
Recall: 0.861
F1 Score: 0.817

Support Vector Machine (SVM):
Accuracy: 0.875
Precision: 0.882
Recall: 0.867
F1 Score: 0.874

DistilBERT Results
Train Dataset Evaluation:
Accuracy: 0.796
Precision: 0.9183627317955676
Recall: 0.64976
F1 Score: 0.7610569715142429

Test Dataset Evaluation:
Accuracy: 0.79916
Precision: 0.9206884913938576
Recall: 0.65472
F1 Score: 0.7652531675160129

## Hyperparameters
**Optimization Results**

Epoch	Training Loss	Validation Loss	Accuracy
1	    0.255900  0.300769	      0.875560

The model was trained for just one epoch which I did due to time constraint and cause the model to converge quickly. A better method for the future would be adding additional epochs to improve the model further. Training loss is lower than validation loss with both values relatively close. This suggest the model has learned a general pattern and is not heavily overfitting. With an accuracy of 87.56%, this model has shown decent performance despite just one epoch.

Analyzing the evaluation results, it indicates the model is not performing well on the evaluation dataset. This may be due to Google Colab restarting my session due to inactivity. Training the model took me 5 hours each code so there is a chance of inconsistency. To fix this error in the future, I would revisit training, model tuning and data issues on Google Colab Pro.

Hyperparameters that were in relevant while optimizing model were the following:
- Learning Rate: 5e-5
  Finding an optimal learning rate will avoid coverging too quickly for suboptimal solutions and/or too slow causing it to get stuck in a local minimum [like I did :( ]
- Batch Size: 16
  Smaller batches can introduce more noise and help model generalize better but a larger batch size can lead to a stable training but will require more memory.
- Number of Epochs: 1
  Too few epoch can lead to underfitting while too many can cause overfitting.
- Optimizer: Adam
Choice of optimizer can affect the speed and quality of convergence. Adam is highly preferred for adaptive learning rate, which can accelerate convergence.
