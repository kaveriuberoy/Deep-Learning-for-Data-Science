import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ucimlrepo import fetch_ucirepo 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
  
# fetching dataset 
covertype = fetch_ucirepo(id=31) 
  
# data (features and targets)
features = covertype.data.features 
targets = covertype.data.targets # cover type
  
# inspecting the dataset
print(covertype.metadata) 
print(covertype.variables) 
print("Feature shape:", features.shape)
print("Target shape:", targets.shape)

# selecting two cover types

data = features.copy()
data["Cover_Type"] = targets["Cover_Type"]

# keeping only classes 1 and 2 (spruce/fir and lodgepole pine)
binary_data = data[data["Cover_Type"].isin([1, 2])].copy()

print("\nBinary dataset shape:", binary_data.shape)
print("\nClass distribution:")
print(binary_data["Cover_Type"].value_counts())

# separating features (x) and target (y).

X = binary_data.drop(columns=["Cover_Type"])

# cover type 1 (spruce/fir) => 0
# cover type 2 (lodgepole pine)=> 1
y = binary_data["Cover_Type"].map({1: 0, 2: 1})

# split into 80% training and 20% testing, keeping proportions of each y the same in the training and set sets. 

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=1,
    stratify=y
)

print("\nTraining set:", X_train.shape)
print("Testing set:", X_test.shape)

print("\nTraining class dist:")
print(y_train.value_counts())

print("\nTesting class dist:")
print(y_test.value_counts())

# standardize features in both training and test sets

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# perceptron classifier

class Perceptron:
    
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.learning_rate = learning_rate # arbitrary learning rate chosen
        self.n_epochs = n_epochs # 100 passes through data. arbitrary value chosen
        self.weights = None
        self.bias = None
        self.errors = []

    def fit(self, X, y):

        # initialize weights and bias to 0 to begin 
        self.weights = np.zeros(X.shape[1])
        self.bias = 0

        # train for a fixed number of epochs
        for epoch in range(self.n_epochs):
            
            errors = 0

            for i in range(X.shape[0]):
                
                # calculate weighted sum (z)
                linear_output = np.dot(X[i], self.weights) + self.bias

                # perceptron prediction
                if linear_output >= 0:
                    prediction = 1
                else:
                    prediction = 0

                # calculate error 
                error = y[i] - prediction

                # update weights and bias if prediction (error) is wrong
                if error != 0:
                    self.weights += self.learning_rate * error * X[i]
                    self.bias += self.learning_rate * error
                    errors += 1

            self.errors.append(errors)

            # if no mistake were made across one whole epoch through the training data, break early, otherwise keep going through epochs
            if errors == 0:
                break

        return self

    def predict(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return (linear_output >= 0).astype(int)


# training the perceptron
perceptron = Perceptron(
    learning_rate=0.1,
    n_epochs=100
)

perceptron.fit(X_train_scaled, y_train.to_numpy())

print("\nPerceptron training complete.")
print("Number of epochs cycled through:", len(perceptron.errors))
print("Final number of errors:", perceptron.errors[-1])

# predictions of true positives, true negatives, false positives, and false negatives for evaluating the perceptron model.

y_pred = perceptron.predict(X_test_scaled)

TP = 0
TN = 0
FP = 0
FN = 0

for actual, predicted in zip(y_test, y_pred):
    
    if actual == 1 and predicted == 1:
        TP += 1
        
    elif actual == 0 and predicted == 0:
        TN += 1
        
    elif actual == 0 and predicted == 1:
        FP += 1
        
    elif actual == 1 and predicted == 0:
        FN += 1

print("\nConfusion Matrix:")
print("True Positives:", TP)
print("True Negatives:", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

# accuracy, preicision, recall and f1 score for perceptron

perceptron_accuracy = (TP + TN) / (TP + TN + FP + FN)
perceptron_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
perceptron_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
perceptron_f1 = (
    2 * perceptron_precision * perceptron_recall/ (perceptron_precision + perceptron_recall)
    if (perceptron_precision + perceptron_recall) > 0
    else 0
)


print("\nPerceptron Performance:")
print("Accuracy :", perceptron_accuracy)
print("Precision:", perceptron_precision)
print("Recall   :", perceptron_recall)
print("F1-score :", perceptron_f1)

# selecting 2 features for visualization of decision boundary (elevation, horizontal distance to hydrology)

feature_1 = "Elevation"
feature_2 = "Horizontal_Distance_To_Hydrology"

feature_indices = [
    X.columns.get_loc(feature_1),
    X.columns.get_loc(feature_2)
]

print("\nFeatures selected for visualization:")
print(feature_1)
print(feature_2)

X_train_2d = X_train_scaled[:, feature_indices]
X_test_2d = X_test_scaled[:, feature_indices]

# elevation, horizontal distance to hydrology for visualization of their decision boundary.
# the remaining 52 standardized features are held constant at 0, their training mean

feature_1 = "Elevation"
feature_2 = "Horizontal_Distance_To_Hydrology"

feature_indices = [
    X.columns.get_loc(feature_1),
    X.columns.get_loc(feature_2)
]

# create ranges for the two features I selected 
x_min = X_test_scaled[:, feature_indices[0]].min() - 1
x_max = X_test_scaled[:, feature_indices[0]].max() + 1

y_min = X_test_scaled[:, feature_indices[1]].min() - 1
y_max = X_test_scaled[:, feature_indices[1]].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# creating a grid containing all 54 features, startimg with all features equal to 0.
grid = np.zeros((xx.ravel().shape[0], X_train_scaled.shape[1]))

# the two visualization features go into the grid.
grid[:, feature_indices[0]] = xx.ravel()
grid[:, feature_indices[1]] = yy.ravel()

# using the Perceptron trained all 54 features, as clarified in the hw questions. 
Z = perceptron.predict(grid)
Z = Z.reshape(xx.shape)

# plot the decision regions
plt.figure(figsize=(10, 7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.2,
    cmap=plt.cm.coolwarm
)

# plot the actual test observations
plt.scatter(
    X_test_2d[y_test.to_numpy() == 0, 0],
    X_test_2d[y_test.to_numpy() == 0, 1],
    color="blue",
    alpha=0.5,
    label="Spruce/Fir (Class 0)"
)

plt.scatter(
    X_test_2d[y_test.to_numpy() == 1, 0],
    X_test_2d[y_test.to_numpy() == 1, 1],
    color="red",
    alpha=0.5,
    label="Lodgepole Pine (Class 1)"
)

# drawing the actual decision boundary
plt.contour(
    xx,
    yy,
    Z,
    levels=[0.5],
    colors="black",
    linewidths=2
)

plt.xlabel(feature_1 + " (standardized)")
plt.ylabel(feature_2 + " (standardized)")
plt.title(
    "Perceptron Decision Boundary"
)
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# logistic regression classifier

class LogisticRegression:
    
    def __init__(self, learning_rate=0.1, n_epochs=100):
        self.learning_rate = learning_rate
        self.n_epochs = n_epochs
        self.weights = None
        self.bias = None
        self.losses = []

    # sigmoid function
    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    # train the model
    def fit(self, X, y):
        
        # number of training samples and features
        n_samples, n_features = X.shape

        # initialize weights and bias to 0 to begin
        self.weights = np.zeros(n_features)
        self.bias = 0

        # updating weights 
        for epoch in range(self.n_epochs):

            # calculate linear output (z)
            linear_output = np.dot(X, self.weights) + self.bias

            # convert linear output to probability vals using sigmoid function
            predictions = self.sigmoid(linear_output)

            # calculate gradients for weights and biases (averaging)
            dw = (1 / n_samples) * np.dot(X.T, (predictions - y))
            db = (1 / n_samples) * np.sum(predictions - y)

            # update weights and bias
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db

            # calculate log loss, preventing predictions from being exactly 0 or 1 for logs
            epsilon = 1e-15
            predictions_clipped = np.clip(
                predictions,
                epsilon,
                1 - epsilon
            )

            loss = -(1 / n_samples) * np.sum(
                y * np.log(predictions_clipped)
                + (1 - y) * np.log(1 - predictions_clipped)
            )

            self.losses.append(loss)

        return self

    # predict probabilities
    def predict_probability(self, X):
        linear_output = np.dot(X, self.weights) + self.bias
        return self.sigmoid(linear_output)

    # predict class labels
    def predict(self, X, threshold=0.5):
        probabilities = self.predict_probability(X)
        return (probabilities >= threshold).astype(int)
    
# train the logistic classifier
logistic_model = LogisticRegression(
    learning_rate=0.1,
    n_epochs=100
)

logistic_model.fit(
    X_train_scaled,
    y_train.to_numpy()
)

print("\nLogistic Regression training complete.")
print("Number of epoch cycled through:", len(logistic_model.losses))
print("Final loss:", logistic_model.losses[-1])

# make predictions
y_pred = logistic_model.predict(X_test_scaled)
# get predicted probabilities
y_prob = logistic_model.predict_probability(X_test_scaled)

print("\nFirst 10 predicted probabilities:")
print(y_prob[:10])

print("\nFirst 10 predicted classes:")
print(y_pred[:10])

TP = 0
TN = 0
FP = 0
FN = 0

for actual, predicted in zip(y_test.to_numpy(), y_pred):

    if actual == 1 and predicted == 1:
        TP += 1

    elif actual == 0 and predicted == 0:
        TN += 1

    elif actual == 0 and predicted == 1:
        FP += 1

    elif actual == 1 and predicted == 0:
        FN += 1


print("\nConfusion Matrix:")
print("True Positives :", TP)
print("True Negatives :", TN)
print("False Positives:", FP)
print("False Negatives:", FN)

# accuracy, preicision, recall and f1 score 
logistic_accuracy = (TP + TN) / (TP + TN + FP + FN)
logistic_precision = TP / (TP + FP) if (TP + FP) > 0 else 0
logistic_recall = TP / (TP + FN) if (TP + FN) > 0 else 0
logistic_f1 = (
    2 * logistic_precision * logistic_recall  / (logistic_precision + logistic_recall )
    if (logistic_precision + logistic_recall ) > 0
    else 0
)

print("\nLogistic Regression Performance:")
print("Accuracy :", logistic_accuracy)
print("Precision:", logistic_precision)
print("Recall   :", logistic_recall )
print("F1-score :", logistic_f1)

# training loss
plt.figure(figsize=(8, 5))

plt.plot(
    range(1, len(logistic_model.losses) + 1),
    logistic_model.losses
)

plt.xlabel("Epoch")
plt.ylabel("Log Loss")
plt.title("Logistic Regression Training Loss")
plt.grid(alpha=0.3)

plt.show()

# Visualizing the decision boundary of the Logistic Regression model trained using all 54 features, visualizing:
# Elevation and Horizontal_Distance_To_Hydrology. The remaining 52 standardized features are held constant at 0.

feature_1 = "Elevation"
feature_2 = "Horizontal_Distance_To_Hydrology"

feature_indices = [
    X.columns.get_loc(feature_1),
    X.columns.get_loc(feature_2)
]

# create ranges for the two visualization features
x_min = X_test_scaled[:, feature_indices[0]].min() - 1
x_max = X_test_scaled[:, feature_indices[0]].max() + 1

y_min = X_test_scaled[:, feature_indices[1]].min() - 1
y_max = X_test_scaled[:, feature_indices[1]].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 500),
    np.linspace(y_min, y_max, 500)
)

# create a grid containing all 54 features
grid = np.zeros(
    (xx.ravel().shape[0], X_train_scaled.shape[1])
)

# set the two visualization features
grid[:, feature_indices[0]] = xx.ravel()
grid[:, feature_indices[1]] = yy.ravel()

# using model trained on all 54 features
Z_prob = logistic_model.predict_probability(grid)

# convert probabilities to class predictions
Z = (Z_prob >= 0.5).astype(int)
Z = Z.reshape(xx.shape)

# plot decision regions
plt.figure(figsize=(10, 7))

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.2,
    cmap=plt.cm.coolwarm
)

# plotting actual test observations
plt.scatter(
    X_test_2d[y_test.to_numpy() == 0, 0],
    X_test_2d[y_test.to_numpy() == 0, 1],
    color="blue",
    alpha=0.5,
    label="Spruce/Fir (Class 0)"
)

plt.scatter(
    X_test_2d[y_test.to_numpy() == 1, 0],
    X_test_2d[y_test.to_numpy() == 1, 1],
    color="red",
    alpha=0.5,
    label="Lodgepole Pine (Class 1)"
)

# logistic regression decision boundary occurs where predicted probability is 0.5
plt.contour(
    xx,
    yy,
    Z_prob.reshape(xx.shape),
    levels=[0.5],
    colors="black",
    linewidths=2
)

plt.xlabel(feature_1 + " (standardized)")
plt.ylabel(feature_2 + " (standardized)")
plt.title(
    "Logistic Regression Decision Boundary\n"
    "(Model trained using all 54 features)"
)
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# BCE loss
y_prob = logistic_model.predict_probability(X_test_scaled)

# to avoid log(0) and log(1) numerical problems
epsilon = 1e-15
y_prob_clipped = np.clip(y_prob, epsilon, 1 - epsilon)

# calculate BCE loss on the test set
y_test_array = y_test.to_numpy()

bce = -np.mean(
    y_test_array * np.log(y_prob_clipped)
    + (1 - y_test_array) * np.log(1 - y_prob_clipped)
)

print("\nBinary Cross-Entropy Loss:")
print("BCE:", bce)

# create a range of predicted probabilities
probabilities = np.linspace(0.001, 0.999, 500)

# BCE when the actual class is 1
bce_class_1 = -np.log(probabilities)

# BCE when the actual class is 0
bce_class_0 = -np.log(1 - probabilities)

# Create the plot
plt.figure(figsize=(10, 6))

plt.plot(
    probabilities,
    bce_class_1,
    color="red",
    linewidth=2,
    label="Actual class = 1"
)

plt.plot(
    probabilities,
    bce_class_0,
    color="blue",
    linewidth=2,
    label="Actual class = 0"
)

plt.xlabel("Predicted Probability of Class 1")
plt.ylabel("Binary Cross-Entropy Loss")
plt.title("Binary Cross-Entropy Loss vs Predicted Probability")
plt.legend()
plt.grid(alpha=0.3)

plt.show()

# comparison table for perceptron and logistic regression

comparison = pd.DataFrame({
    "Model": [
        "Perceptron",
        "Logistic Regression"
    ],
    
    "Accuracy": [
        perceptron_accuracy,
        logistic_accuracy
    ],
    
    "Precision": [
        perceptron_precision,
        logistic_precision
    ],
    
    "Recall": [
        perceptron_recall,
        logistic_recall
    ],
    
    "F1-score": [
        perceptron_f1,
        logistic_f1
    ],
    
    "BCE Loss": [
        np.nan,
        bce
    ]
})

print("\nModel Comparison:")
print(comparison.to_string(index=False))
