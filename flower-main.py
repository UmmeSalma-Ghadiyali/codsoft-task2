import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
df = pd.read_csv('flower-dataset.csv')

# Display the first few rows of the dataset
print(df.head())

# Prepare the data
X = df[['sepal_length', 'sepal_width', 'petal_length', 'petal_width']]
y = df['species']

# Split the data into training and testing sets manually
# Using 80% of the data for training and 20% for testing
train_size = int(0.8 * len(df))
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Define a function to compute Euclidean distance between two points
def euclidean_distance(x1, x2):
    return np.sqrt(np.sum((x1 - x2)**2))

# Define the KNN classifier
class KNNClassifier:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        y_pred = [self._predict(x) for x in X]
        return np.array(y_pred)

    def _predict(self, x):
        # Compute distances between x and all examples in the training set
        distances = [euclidean_distance(x, x_train) for x_train in self.X_train]
        # Sort by distance and return indices of the first k neighbors
        k_indices = np.argsort(distances)[:self.k]
        # Extract the labels of the k nearest neighbor training samples
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Return the most common class label among the k nearest neighbors
        most_common = max(set(k_nearest_labels), key=k_nearest_labels.count)
        return most_common

# Instantiate the KNN classifier with k=3
knn = KNNClassifier(k=3)

# Train the classifier
knn.fit(X_train.values, y_train.values)

# Make predictions on the test set
y_pred = knn.predict(X_test.values)

# Evaluate the model
def accuracy(y_true, y_pred):
    return np.sum(y_true == y_pred) / len(y_true)

print(f'Accuracy: {accuracy(y_test.values, y_pred):.2f}')

# Plot the confusion matrix
def plot_confusion_matrix(y_true, y_pred, classes):
    unique_classes = np.unique(classes)
    cm = np.zeros((len(unique_classes), len(unique_classes)), dtype=int)
    for i in range(len(y_true)):
        true_index = np.where(unique_classes == y_true[i])[0][0]
        pred_index = np.where(unique_classes == y_pred[i])[0][0]
        cm[true_index][pred_index] += 1

    plt.imshow(cm, interpolation='nearest', cmap='viridis')
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(unique_classes))
    plt.xticks(tick_marks, unique_classes, rotation=45)
    plt.yticks(tick_marks, unique_classes)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')

plot_confusion_matrix(y_test.values, y_pred, df['species'].unique())
plt.show()

# Define the rows for classification
rows_for_classification = [2, 52, 102, 3, 54, 107, 5, 60, 120, 15, 92, 142]

# Make predictions for the selected rows
y_pred_specific = knn.predict(X.iloc[rows_for_classification].values)

# Display the classification results in a table
results_specific_df = pd.DataFrame({'Row No.': rows_for_classification, 'True Label': y.iloc[rows_for_classification].values, 'Predicted Label': y_pred_specific})
print('\nSpecific Classification Results:')
print(results_specific_df)