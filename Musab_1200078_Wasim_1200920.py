import csv
import sys
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

# Note :The code may take some time when run

TEST_SIZE = 0.3
K = 3

class NN:
    def __init__(self, trainingFeatures, trainingLabels) -> None:
        self.trainingFeatures = trainingFeatures
        self.trainingLabels = trainingLabels

    def euclidean_distance(self, point1, point2):
        distance = sum([(a - b) ** 2 for a, b in zip(point1, point2)])
        return distance ** 0.5

    def predict(self, testFeatures, k):
        """
        Given a list of feature vectors for testing examples,
        return the predicted class labels (list of either 0s or 1s)
        using the k nearest neighbors.
        """
        predictions = []
        for test_instance in testFeatures:
            distances = []
            for i in range(len(self.trainingFeatures)):
                dist = self.euclidean_distance(test_instance, self.trainingFeatures[i])
                distances.append((self.trainingLabels[i], dist))
            distances.sort(key=lambda x: x[1])
            k_nearest_labels = [label for label, dist in distances[:k]]
            prediction = max(set(k_nearest_labels), key=k_nearest_labels.count)
            predictions.append(prediction)
        return predictions


def load_data(filename):
    """
     Load spam data from a CSV file `filename` and convert into a list of
     features vectors and a list of target labels. Return a tuple (features, labels).

     features vectors should be a list of lists, where each list contains the
     57 features vectors

     labels should be the corresponding list of labels, where each label
     is 1 if spam, and 0 otherwise.
     """
    features = []
    labels = []

    with open(filename, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            features.append([float(i) for i in row[:-1]])
            labels.append(int(row[-1]))

    return features, labels


def preprocess(features):
    """
    Normalize each feature by subtracting the mean value in each
    feature and dividing by the standard deviation.
    """

    features = np.array(features)
    means = np.mean(features, axis=0)
    stds = np.std(features, axis=0)
    normalized_features = (features - means) / stds

    return normalized_features.tolist()


def train_mlp_model(features, labels):
    """
    Given a list of features lists and a list of labels, return a
    fitted MLP model trained on the data using sklearn implementation.
    """
    mlp = MLPClassifier(hidden_layer_sizes=(10, 5), activation='logistic', max_iter=2000)
    mlp.fit(features, labels)

    return mlp




def evaluate(labels, predictions):
    """
      Given a list of actual labels and a list of predicted labels,
      return (accuracy, precision, recall, f1).

      Assume each label is either a 1 (positive) or 0 (negative).
      """
    TP = FN = FP = TN = 0
    for i in range(len(labels)):
        if labels[i] == 1 and predictions[i] == 1:
            TP += 1
        elif labels[i] == 1 and predictions[i] == 0:
            FN += 1
        elif labels[i] == 0 and predictions[i] == 1:
            FP += 1
        else:
            TN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = TP / (TP + FP)
    recall = TP / (TP + FN)
    f1 = 2 * precision * recall / (precision + recall)

    return accuracy, precision, recall, f1



def main():

    # Check command-line arguments
    if len(sys.argv) != 2:
         sys.exit("Usage: python template.py ./spambase.csv")

    # Load data from spreadsheet and split into train and test sets
    features, labels = load_data(sys.argv[1])
    features = preprocess(features)
    X_train, X_test, y_train, y_test = train_test_split(
        features, labels, test_size=TEST_SIZE)

    # Train a k-NN model and make predictions
    model_nn = NN(X_train, y_train)
    predictions = model_nn.predict(X_test, K)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** 1-Nearest Neighbor Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)

    # Train an MLP model and make predictions
    model = train_mlp_model(X_train, y_train)
    predictions = model.predict(X_test)
    accuracy, precision, recall, f1 = evaluate(y_test, predictions)

    # Print results
    print("**** MLP Results ****")
    print("Accuracy: ", accuracy)
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)



if __name__ == "__main__":
    main()
