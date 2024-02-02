from util import entropy, information_gain, split_node
import numpy as np



class DecisionTree(object):
    def __init__(self):
        self.tree = {} # you can use different data structure to build your tree
        self.max_depth = 10

    # Inside DecisionTree class

    def train(self, X, y, depth=0):
        """
        This method trains decision tree (trains = construct = build)
        Args:
            X: data excluded target feature
            y: target feature

        Returns:

        NOTES:  You can add more parameter to build your algorithm if necessary.
                You will have to use the functions from util.py
                Construct your tree using a dictionary or any other data structure.
                Each key should represent a property of your tree, and the value is the corresponding value for your key.
        IMPORTANT:  ADD RANDOMNESS
                    You should add randomness to your decision tree for random forest.
                    At each node: select random 5 features from the feature list (22 features) and
                    compare the information gain of only 5 randomly selected features to select the splitting attribute with the highest information gain.
                    The selected random features should change at every node choice.
        Example: You can think each node as a Node object, and the node object should have some properties.
                A node should have split_value and split_attribute properties that give us the information of that node.
                (Below example is just an example; each tree should have more properties)
                Like: tree["split_value"] = split_value_you_find
                    tree["split_attribute"] = split_feature_with_highest_information_gain
        """
        bestInfoGain = -1
        bestSplittingFeature = None
        bestSplitingValue = None

        
        #print("Step 1..")#Checkpointsss
        if len(X) == 0:
            self.tree["value"] = np.random.choice(2)
            return
        #print("Step 2..")#Checkpointsss
        if depth >= self.max_depth or len(np.unique(y)) == 1:
            unique_labels, counts = np.unique(y, return_counts=True)
            self.tree["value"] = unique_labels[np.argmax(counts)]
            return
        
        

        numberofFeature = X.shape[1]
        selected_features = np.random.choice(numberofFeature, min(5, numberofFeature), replace=False)

        

        for split_feature in selected_features:
            unique_values = np.unique(X[:, split_feature])
            
            #print("Step 3..")#Checkpointsss
            if len(unique_values) > 100:
                unique_values = np.random.choice(unique_values, 100, replace=False)

            for split_value in unique_values:
                # Split the data
                X_left, X_right, y_left, y_right = split_node(X, y, split_feature, split_value)

                # Calculate information gain
                currentInfoGain = information_gain(y, [y_left, y_right])

                # Update best split if the current one is better
                if currentInfoGain > bestInfoGain:
                    bestInfoGain = currentInfoGain
                    bestSplittingFeature = split_feature
                    bestSplitingValue = split_value
        X_left, X_right, y_left, y_right = split_node(X, y, bestSplittingFeature, bestSplitingValue)

        #print("Step 4..")#Checkpointsss
        if bestSplittingFeature is None:
            if len(np.unique(y)) == 1:
                self.tree['value'] = y[0]
            else:
                unique_labels, counts = np.unique(y, return_counts=True)
                most_common_label = unique_labels[np.argmax(counts)]
                self.tree['value'] = most_common_label
        
        else:
            self.tree['feature_index'] = bestSplittingFeature
            self.tree['threshold'] = bestSplitingValue

            X_left, X_right, y_left, y_right = split_node(X, y, bestSplittingFeature, bestSplitingValue)

            self.tree['left'] = DecisionTree()
            self.tree['left'].train(X_left, y_left, depth + 1)

            self.tree['right'] = DecisionTree()
            self.tree['right'].train(X_right, y_right, depth + 1),
            

        return self.tree

       

    def classify(self, record):
        """
        This method classifies the record using the tree you build and returns the predicted la
        Args:
            record: each data point

        Returns:
            predicted_label

        """
        if 'value' in self.tree:
            return self.tree['value']

        feature = record[self.tree['feature_index']]


        if feature <= self.tree['threshold']:
            next_tree = self.tree['left']
        else:
            next_tree = self.tree['right']

        return next_tree.classify(record) if next_tree else None



        
        

        
        
