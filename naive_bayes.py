from typing import Hashable, Iterable, Dict, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

class naive_bayes:

    def __init__(self, labels: Iterable[Hashable]):
        """
        Initializes following:
        labels: A list of values range(0-9) for the possible classification outcomes.
        label_count: A dictionary containing the count of each label.
        model: A 2D NumPy array where rows correspond to labels and columns to features.
        """
        self.labels = labels
        self.label_count = {label: 0 for label in labels} 
        self.model = {label: [] for label in labels}

    def add_example(self, filename: str):
        """
        Reads csv training data file. 
        For each row, add the example to the model and update necessary variables.
        """
        # Read the CSV file
        train_df = pd.read_csv(filename)

        # Iterate over each row in the dataframe
        for _, row in train_df.iterrows():
            # The first column is the label
            label = row.iloc[0]   
            # The rest are the features(pixel0 - pixel783), normalize to [0, 1] because max value is 255, threshold as the middle 127
            features = row.iloc[1:].values / 127.0

            self.label_count[label] += 1
            # Dynamically initialize feature list if not already done
            if not self.model[label]:
                self.model[label] = [0.0] * len(features)

            # Add features to the corresponding label in the model
            self.model[label] = [
                self.model[label][i] + features[i] for i in range(len(features))
            ]

    def test_input(self, filename: str):
        """
        Reads csv test data file. 
        For each row, add the example to the model and update necessary variables.
        """
        # Read the CSV file
        test_df = pd.read_csv(filename)

        true_labels = []  
        pred_labels = []
        for _, row in test_df.iterrows():
            label = row.iloc[0]   
            features = row.iloc[1:].values / 127.0
            
            true_labels.append(label)
            pred_labels.append(self.predict(features, psuedo=0.5))
        
        # Compute metrics
        return self.compute_metrics(true_labels, pred_labels)

    def calculate_prior(self,label_count:Dict[Hashable, int]):
        """
        Calculates the prior probabilities P(y) for each label.

        return: A dictionary with prior probabilities for each label.
        """
        total_training_samples= sum(label_count.values())
        prior = {label: count / total_training_samples for label, count in label_count.items()}
        log_prior = {label: np.log(prob) for label, prob in prior.items()}
        
        return log_prior

    def calculate_uniuqe_feature_count(self, model: Dict[Hashable, List[float]]) -> Dict[Hashable, int]:
        """
        Calculates the count of unique features for all labels in the model.

        return: A dictionary containing the number of unique features seen for each label in the model.
        """
        unique_feature_count = {label: 0 for label in self.labels}
        for label in self.labels:
            for feature in model[label]:
                if feature:
                    unique_feature_count[label] += 1
        return unique_feature_count


    def predict(self, input_features: List[float], psuedo: float = 0.0001) -> int:
        """
        Predicts the label for a given input feature vector using Naive Bayes classification.
        
        :input_features: A list of pixel values representing the input.
        
        :return: The predicted label.
        """
        log_prior = self.calculate_prior(self.label_count)
        unique_feature_count = self.calculate_uniuqe_feature_count(self.model)
        
        likelihoods = {label: 0 for label in self.labels} 
        for feature in range(len(input_features)):
            if input_features[feature]:
                for label in self.labels:
                    likelihoods[label] += np.log((self.model[label][feature] + psuedo)/(self.label_count[label] + (unique_feature_count[label] * psuedo)))
        
        # part in denominaters log(x.y) = log(x) + log(y)
        for label in self.labels:
            likelihoods[label] += log_prior[label]

        denominator = np.logaddexp.reduce([likelihoods[label] for label in self.labels])
        for label in self.labels:
            likelihoods[label] = np.exp(likelihoods[label] - denominator)
        
        # self.display_image(input_features)
        return max(likelihoods, key=likelihoods.get)
    
    def display_image(self, pixel_values: List[float]) -> None:
        """
        Display an image given a list of 784 pixel values.
        
        Args:
        - pixel_values (list): A list of 784 integers ranging from 0 to 255, 
                            representing grayscale intensity.
        
        """
        
        # Convert the list to a 28x28 NumPy array
        image_array = np.array(pixel_values).reshape((28, 28))
        
        # Plot the image using matplotlib
        plt.imshow(image_array, cmap="gray")
        plt.show(block=False)
        plt.pause(3)
        plt.close()
    
    def compute_metrics(self, y_true: List[int], y_pred: List[int]) -> Dict[str, float]:
        """Compute metrics to evaluate binary classification accuracy

        Args:
            y_true: Array-like ground truth (correct) target values.
            y_pred: Array-like estimated targets as returned by a classifier.

        Returns:
            dict: Dictionary of metrics in including confusion matrix, accuracy, recall, precision and F1
        """
        print("accuracy:", metrics.accuracy_score(y_true, y_pred))
        cm = metrics.confusion_matrix(y_true, y_pred, labels=self.labels)
        # Display the confusion matrix
        metrics.ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.labels).plot(cmap="viridis")
        plt.title("Confusion Matrix")
        plt.show(block=False)
        plt.pause(5)
        plt.close()
        

if __name__ == "__main__":
    model = naive_bayes(labels=range(10))
    filename = 'digit-recognizer/mnist_train.csv'
    model.add_example(filename)
    model.test_input('digit-recognizer/mnist_test.csv')