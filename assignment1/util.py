import numpy as np

def entropy(class_y):
    """
    This method calculates entropy
    Args:
        class_y: list of class labels
    Returns:
        entropy: entropy value

    Example: entropy for [0,0,0,1,1,1] should be 1.
    """
    # Count the occurrences of each class in the data
    class_y = np.array(class_y)

    temp, count = np.unique(class_y, return_counts = True)
    
    # Calculate probability for each class
    probabilities = count / len(class_y)
    
    # Avoid log(0) by setting log(0) = 0
    probabilities[probabilities == 0] = 1  # Log(1) is 0
    
    # Calculate entropy
    entropy = -np.sum(probabilities * np.log2(probabilities))
    
    return entropy


def information_gain(previous_y, current_y):
    """
    This method calculates information gain. In this method, use the entropy function you filled
    Args:
        y_before: the distribution of target values before splitting
        y_splitted: the distribution of target values after splitting

    Returns:
        information_gain

    Example: if y_before = [0,0,0,1,1,1] and y_splitted = [[0,0],[0,1,1,1]], information_gain = 0.4691

    """
    systemEntropy = entropy(previous_y)
    weighted = 0
    for y_splitted in current_y:
        if len(y_splitted) == 0:
            continue
        weighted += (len(y_splitted) / len(previous_y)) * entropy(y_splitted)
    info_gain = systemEntropy - weighted
    return info_gain


def split_node(X, y, split_feature, split_value):
    """
    This method implements binary split to your X and y.
    Args:
        X: dataset without target value
        y: target labels
        split_feature: column index of the feature to split on
        split_value: value that is used to split X and y into two parts

    Returns:
        X_left: X values for left subtree
        X_right: X values for right subtree
        y_left: y values of X values in the left subtree
        y_right: y values of X values in the right subtree

    Notes:  Implement binary split.
            You can use mean for split_value or you can try different split_value values for better Random Forest results
            Assume you are only dealing with numerical features. You can ignore the case there are categorical features.
            Example:
                Divide X into two list.
                X_left: where values are <= split_value.
                X_right: where values are > split_value.
    """
    
    X_left = []
    X_right = []
    y_left = []
    y_right = []
    
   # Iterate through each row in X and corresponding label in y
    for i in range(len(X)):
        if X[i, split_feature] <= split_value:
            X_left.append(X[i])
            y_left.append(y[i])
        else:
            X_right.append(X[i])
            y_right.append(y[i])

    # Convert lists to numpy arrays
    X_left = np.array(X_left)
    X_right = np.array(X_right)
    y_left = np.array(y_left)
    y_right = np.array(y_right)

    return X_left, X_right, y_left, y_right



def confusion_matrix_(y_predicted, y):
    """
    Args:
        y_predicted: predicted labels
        y: your true labels

    Returns:
        confusion_matrix: with shape (2, 2)
    """
    Confusion = np.zeros((2, 2), dtype=int)  # Ensure the matrix contains integer values
    
    for predicted, true in zip(y_predicted, y):
        predicted = int(predicted)  # Cast predicted value to integer
        true = int(true)  # Cast true value to integer
        Confusion[predicted][true] += 1

    return Confusion


def predicted_value_counts(y_predicted):
    """
    Args:
        y_predicted: predicted labels

    Returns:
        val_1 = count of class 0
        val_2 = count of class 1
    """
    y_predicted = y_predicted.astype(int)  # Convert to integer type
    counts = np.bincount(y_predicted)

    value0 = counts[0] if len(counts) > 0 else 0
    value1 = counts[1] if len(counts) > 1 else 0

    return value0, value1



def eval_metrics(conf_matrix):
    """
    Args:
        conf_matrix: Use confusion matrix you calculated

    Returns:
        accuracy, recall, precision, f1_score

    """

    TP = conf_matrix[1, 1]
    FP = conf_matrix[0, 1]
    FN = conf_matrix[1, 0]
    TN = conf_matrix[0, 0]


    accuracy, recall, precision, f1_score = 0, 0, 0, 0
    
    
    sum = (TP + TN + FP + FN)
    if sum != 0:
        accuracy = (TP + TN) / sum
    if (TP + FN) != 0:
        recall = TP / (TP + FN)
    if (TP + FP) != 0:
        precision = TP / (TP + FP)
    if (precision + recall) != 0:
        f1_score = 2 * (precision * recall) / (precision + recall)

    return accuracy, recall, precision, f1_score
