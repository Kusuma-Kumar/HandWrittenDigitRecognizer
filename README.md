Since Git does not allow us to push large files, please dowload the datasets from Kaggle using the link below. Add it to your digit-recognizer folder and unzip these files.

Data source
    - MNIST data set in csv from Kaggel(https://www.kaggle.com/datasets/oddrationale/mnist-in-csv?select=mnist_train.csv)
    - The mnist_train.csv file contains the 60,000 training examples and labels.
    - The mnist_test.csv contains 10,000 test examples and labels. Each row consists of 785 values: the first value is the label (a number from 0 to 9) and the remaining 784 values are the pixel values (a number from 0 to 255).

naive_bayes.py
    - implements a naive bayes model without using any available package
    - uses the train data to train the model and the test data to evaluate the model
    - uses the trained model to predict the labels of the test data
    - prints the accuracy and confusion matrix of the model