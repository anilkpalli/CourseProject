# Objective:
The objective of this task is to detect sarcasm in tweets. 

# Datasets:
* Train: 5000 records (Columns: response, context, and label)
* Test: 1000 records (Columns: response, context)

# Tools Used:
* [Google colab notebook (free version)](https://colab.research.google.com/notebooks/intro.ipynb)
* Language: Python
* Libraries: torch, sklearn, transformers (ver=3.0.0), numpy, pandas, json, time

# Approach:
RoBERTa (developed as an extension of BERT) pre-trained model is used as the starting point. Further the model is fine-tuned using Sarcasm train dataset and the final optimal model is used for predicting the labels on Sarcasm test dataset. The code is split into 6 sections. In order to get the final predictions the code needs to be just executed in sequence as described below.

## Section: 1 - Read input data
The train and test json files are stored in google drive and are directly imported into google colab by providing appropriate credentials. Refer to this short [tutorial](https://www.youtube.com/watch?v=oqMImCeXi6o) for importing data into colab

## Section: 2 - Preprocess data
The input data (train) consists of 3 columns: response, context and label. In this step the response and context are combined into one single sentence in the order of last to first conversation sequence. Further analysis is done using this single sentence approach which captures both response and context as one complete conversation.

_Note_: 
* Tried out using response alone without context and it didn’t perform better than using response and context together
* Tried out data cleaning steps like removing stop words, special characters, urls etc.. but they didn't prove to be any useful in improving the accuracy while using this approach so removed those steps from final code

## Section: 3 - Prepare data for modeling
* _Split the data into training and validation_: 80% of the data from train set is used for training the model and 20% of train dataset is hold-out for validation purpose.
* _Tokenize and encode sequences_ of both training and validation sets: The conversations are of varying lengths and therefore have to be truncated and padded to equal lengths. Based on the distribution of lengths, max length is selected as 200 since most of the conversations are covered within this range. Any higher number could possibly cause model to train slower or run out of memory due to colab limitations in its free version
* _Create train and validation tensor datasets_: Tensor datasets are created to work efficiently with torch framework while building the model

## Section: 4 - Define model build functions
Define functions to initialize the pre-trained roberta-base model and fine-tune the parameters as needed to fit the data in hand. In-line documentation of these functions is available in the code.

_Note_: 
* For the per-trained model, tried out bert-base-uncased, bert-large-uncased, roberta-base, roberta-large. Among these the _large_ variants ran out of memory quickly and among the base variants roberta-base performed better, so retained it in the final submission
* Make sure to set appropriate seed values to reproduce the results

## Section: 5 - Build the model
* _Define model parameters_: Includes defining optimizer, loss functions, number of epochs and batch size. Used Adam optimizer and a suitable learning rate to tune roberta-base for 10 epochs. Negative Log-likelihood loss (alternatively cross-entropy loss) is used as the loss function.
* _Run the model_ and store the best model: The model is iterated for each epoch while optimizing the parameters. During training, the model parameters are evaluated against the validation set. Saved the model each time the validation accuracy increases so that the model with the highest validation accuracy is identified and stored. The train and validation metrics (loss and accuracy) are captured and stored for all the epochs. 

## Section: 6 - Test predictions
* _Load the best model_ - Load the model having the highest validation accuracy that is stored in prior step for predicting the test set
* _Prepare and manipulate test data_ - The test set should undergo same data preprocessing and preparation steps as the training which includes: combining response and context into one sentence, tokenize and encode test conversations
* _Split the test dataset_ into 2 parts to overcome limited space issue in colab - There are 1800 records in test set. Due to the space limitations in colab where majority was already utilized for loading and tuning pre-trained RoBERTa, the test set was split into 2 sets of 1000 and 800 records respectively.
* _Get the predictions_ for the 2 test sets and combine into one final dataframe - The test sets are scored using the best model and the predictions are combined into one single dataframe and then to csv to submit the result in appropriate format
