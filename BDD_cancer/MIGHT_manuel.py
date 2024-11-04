import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# Define a custom MIGHT Tree class
class MightTree:
    def __init__(self, max_depth=None, random_state=None):
        self.architecture_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.leaf_probs = None  # Stores the probability of positive class for each leaf node
        self.random_state = random_state

    def fit(self, X_arch, y_arch, X_leaf, y_leaf):
        # Use the Architecture Group to train the architecture tree
        self.architecture_tree.fit(X_arch, y_arch)

        # Get the leaf node indices for the Leaf Group
        leaf_indices = self.architecture_tree.apply(X_leaf)

        # Calculate the probability of positive class for each leaf node
        self.leaf_probs = {}
        for leaf_index in np.unique(leaf_indices):
            leaf_samples = (leaf_indices == leaf_index)
            # Calculate the mean of the positive class labels in the leaf node
            self.leaf_probs[leaf_index] = y_leaf[leaf_samples].mean()

    def predict(self, X_test):
        # Traverse the architecture tree to get the leaf node indices for the test data
        leaf_indices = self.architecture_tree.apply(X_test)
        return np.array([self.leaf_probs.get(leaf, 0.5) for leaf in leaf_indices])

# Define a custom MIGHT Forest class
class MightForest:
    def __init__(self, n_trees=100, max_depth=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.forest = []

    def fit(self, X, y):
        n_samples = len(X)
        for i in range(self.n_trees):
            # Set the random state for reproducibility
            rs = self.random_state + i if self.random_state is not None else None

            # Randomly sample 160 samples with replacement
            sample_indices = np.random.choice(n_samples, size=160, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # Divide the sample into Architecture and Leaf groups
            X_arch, X_temp, y_arch, y_temp = train_test_split(
                X_sample, y_sample, test_size=0.6, stratify=y_sample, random_state=rs
            )
            X_leaf, X_test, y_leaf, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=rs
            )

            # Build a MIGHT Tree using the Architecture and Leaf groups
            tree = MightTree(max_depth=self.max_depth, random_state=rs)
            tree.fit(X_arch, y_arch, X_leaf, y_leaf)

            # Store the trained tree and the Leaf group
            self.forest.append((tree, X_test))

    def predict(self, X):
        # Initialize arrays to store the sum of scores and the count of predictions for each sample
        all_scores = np.zeros(len(X))
        counts = np.zeros(len(X))

        for tree, X_test in self.forest:
            # Get the scores for the Leaf group from the MIGHT Tree
            scores = tree.predict(X_test)

            # For each test sample, add the score to the sum and increment the count
            for j, x_test in enumerate(X_test):
                idx = np.where((X == x_test).all(axis=1))[0]
                if len(idx) > 0:
                    all_scores[idx[0]] += scores[j]
                    counts[idx[0]] += 1

        # Calculate the average score for each sample
        avg_scores = np.zeros(len(X))
        avg_scores[counts > 0] = all_scores[counts > 0] / counts[counts > 0]
        avg_scores[counts == 0] = 0.5  # Fill missing values with 0.5

        return avg_scores

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(X, y, test_size=0.2, random_state=425, stratify=y)

# Initialize and train the MIGHT Forest model
might_forest = MightForest(n_trees=100, max_depth=5, random_state=42)
might_forest.fit(X_train_full, y_train_full)

# Predict the probabilities for the test set
y_pred_might = might_forest.predict(X_test_full)

# Calculate the ROC AUC score for the MIGHT model
roc_auc_might = roc_auc_score(y_test_full, y_pred_might)
print(f"MIGHT模型测试集 AUC: {roc_auc_might:.2f}")