import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import random as sparse_random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# Sparse projection matrix generating function
def sparse_projection_matrix(n_features, density=0.1, random_state=None):
    """Generate a sparse projection matrix with the given density."""
    rng = np.random.default_rng(random_state)
    sparse_matrix = sparse_random(n_features, n_features, density=density, random_state=rng, format='csr')
    return sparse_matrix.toarray()

# Define a custom sparse projection decision tree classifier
class SparseProjectionDecisionTreeClassifier:
    def __init__(self, max_depth=None, density=0.1, random_state=None):
        self.base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.projection_matrix = None
        self.density = density
        self.random_state = random_state

    def fit(self, X, y):
        n_features = X.shape[1]
        # Generate a sparse projection matrix
        self.projection_matrix = sparse_projection_matrix(n_features, density=self.density, random_state=self.random_state)
        
        # Project the data into the sparse space
        X_projected = X @ self.projection_matrix
        self.base_tree.fit(X_projected, y)

    def predict(self, X):
        # Project the input data into the sparse space and make predictions
        X_projected = X @ self.projection_matrix
        return self.base_tree.predict(X_projected)

# Define a custom sparse projection random forest classifier
class SparseProjectionRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, density=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.density = density
        self.random_state = random_state
        self.forest = []

    def fit(self, X, y):
        # Train a sparse projection decision tree for each estimator
        for i in range(self.n_estimators):
            tree = SparseProjectionDecisionTreeClassifier(max_depth=self.max_depth, density=self.density, random_state=self.random_state)
            tree.fit(X, y)
            self.forest.append(tree)

    def predict(self, X):
        # Make predictions for each tree in the forest and return the average
        predictions = np.array([tree.predict(X) for tree in self.forest])
        return np.mean(predictions, axis=0).round()

# Load the breast cancer dataset
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Train a sparse projection random forest classifier
sporf_clf = SparseProjectionRandomForestClassifier(n_estimators=10, max_depth=10, density=0.1, random_state=42)
sporf_clf.fit(X_train, y_train)

# Evaluate the classifier on the test set
y_pred = sporf_clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"测试集 AUC: {roc_auc:.2f}")