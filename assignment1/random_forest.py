from decision_tree import DecisionTree
from util import confusion_matrix_, eval_metrics, predicted_value_counts
import csv
import numpy as np
import ast


class RandomForest(object):
    num_trees = 0
    decision_trees = []
    # Includes bootstrapped datasets. List of list
    bootstrap_datasets = []
    # Corresponding class labels for above bootstraps_datasets
    bootstrap_labels = []

    def __init__(self, n_trees):
        self.n_trees = n_trees
        self.decision_trees = [DecisionTree() for i in range(n_trees)]

    def create_bootstrap_dataset(self, XX, n):
        """
        In this method, create sample datasets with size n by sampling with replacement from XX.
        You can prefer to use X and y instead of XX. It depends on your implementation.
        Args:
            XX: original dataset (includes target feature y)
            n: sampling size

        Returns:
            samples: sampled data (excluding target feature y)
            labels:

        """

        samples = []  # sampled dataset
        labels = []  # class labels for the sampled records
        num_records = len(XX)
        indices = np.random.choice(num_records, n, replace=True)

        samples = np.array(XX)[indices, :-1]  # Convert to NumPy array and then perform indexing
        labels = np.array(XX)[indices, -1]    # Convert to NumPy array and then perform indexing

        return samples, labels

    def bootstrap(self, XX):
        """
        This method initializes the bootstrap datasets for each tree
        Args:
            XX
        """
        for i in range(self.n_trees):
            data_sample, data_label = self.create_bootstrap_dataset(XX, len(XX))
            self.bootstrap_datasets.append(data_sample)
            self.bootstrap_labels.append(data_label)

    def fitting(self):
        """
        This method train each decision tree (with number of n_trees) using the bootstraps datasets and labels
        """
        print("Fitting function is called.")
        for i in range(self.n_trees):
            print(f"Training tree waiting {i+1}")
            tree = self.decision_trees[i]
            bootstrapData = self.bootstrap_datasets[i]
            bootstrapLabels = self.bootstrap_labels[i]

            # Train the decision tree with the current bootstrap dataset
            tree.train(bootstrapData, bootstrapLabels)
            print(f"Training tree completed. {i+1}")
        
        print("Fitting process completed.")  

    def majority_voting(self, X):
        """
        This method predicts labels for X using all the decision tree we fit.
        Args:
            X: dataset

        Returns:
            y: predicted value for each data point (includes the prediction of each decision tree)

        Explanation:
            After finding the predicted labels by all the decision tree, use majority voting to choose one class label
            Example: if 3 decision tree find the class labels as [1, 1, 0],
                    the majority voting should find the predicted label as 1
                    because the number of 1's is bigger than the number of 0's
        """
        
        print("Voting starting..")
        predictions = np.zeros((len(self.decision_trees), len(X)))
        
        
        # Get predictions from each decision tree
        
        
        for i in range(len(self.decision_trees)):
            print(f"Voting {i+1} working...")
            tree = self.decision_trees[i]
            predictions[i] = [tree.classify(records) for records in X]
            print(f"Voting {i+1} Done.")
        
        
        # Perform majority voting
        y = np.zeros(len(X))
        for i in range(len(X)):
            unique_labels, counts = np.unique(predictions[:, i], return_counts=True)
            y[i] = unique_labels[np.argmax(counts)]

        return y


def read_dataset():
    X = list()
    y = list()
    XX = list()  # Contains data features and data labels

    # Loading data set
    print("...Reading Airline Passenger Satisfaction Dataset...")
    with open("airline_passenger_satisfaction.csv") as f:
        next(f, None)
        for line in csv.reader(f, delimiter=","):
            xline = []
            for i in range(len(line)):
                xline.append(ast.literal_eval(line[i]))

            X.append(xline[:-1])
            y.append(xline[-1])
            XX.append(xline[:])


    return X, y, XX


# Do not change below. You can try different forest_size values
def main():
    """
    In this homework, you are only using train set. You don't have to split your data into train and test set
    Only use XX, X and y.
    X: matrix with n rows and d columns where n is the number of rows and d is the number of features
    y: target features with length n
    XX: X + y
    """
    X, y, XX = read_dataset()

    #  You can change below
    forest_size = 10

    # Initializing a random forest.
    randomForest = RandomForest(forest_size)
    randomForest.bootstrap(XX)

    print("...Fit your forest (all your decision trees)...")
    randomForest.fitting()

    print("...Make Prediction...")
    y_predicted = randomForest.majority_voting(X)

    matr = confusion_matrix_(y_predicted, y)
    accuracy, recall, precision, f1_score = eval_metrics(matr)

    class_0_count, class_1_count = predicted_value_counts(y_predicted)
    print(matr)
    print("How many value is predicted as class 0: ", class_0_count)
    print("How many value is predicted as class 1: ", class_1_count)
    print("accuracy: %.4f" % accuracy)
    print("recall: %.4f" % recall)
    print("precision: %.4f" % precision)
    print("f1_score: %.4f" % f1_score)


if __name__ == "__main__":
    main()
