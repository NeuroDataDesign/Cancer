import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from scipy.sparse import random as sparse_random
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

# 稀疏投影矩阵生成函数
def sparse_projection_matrix(n_features, density=0.1, random_state=None):
    """生成稀疏投影矩阵"""
    rng = np.random.default_rng(random_state)
    sparse_matrix = sparse_random(n_features, n_features, density=density, random_state=rng, format='csr')
    return sparse_matrix.toarray()

# 自定义稀疏投影决策树分类器
class SparseProjectionDecisionTreeClassifier:
    def __init__(self, max_depth=None, density=0.1, random_state=None):
        self.base_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.projection_matrix = None
        self.density = density
        self.random_state = random_state

    def fit(self, X, y):
        n_features = X.shape[1]
        # 生成稀疏投影矩阵
        self.projection_matrix = sparse_projection_matrix(n_features, density=self.density, random_state=self.random_state)
        
        # 投影数据并训练决策树
        X_projected = X @ self.projection_matrix
        self.base_tree.fit(X_projected, y)

    def predict(self, X):
        # 将数据投影到稀疏空间
        X_projected = X @ self.projection_matrix
        return self.base_tree.predict(X_projected)

# 自定义稀疏投影随机森林分类器
class SparseProjectionRandomForestClassifier:
    def __init__(self, n_estimators=100, max_depth=None, density=0.1, random_state=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.density = density
        self.random_state = random_state
        self.forest = []

    def fit(self, X, y):
        # 为每棵树生成稀疏投影矩阵并训练
        for i in range(self.n_estimators):
            tree = SparseProjectionDecisionTreeClassifier(max_depth=self.max_depth, density=self.density, random_state=self.random_state)
            tree.fit(X, y)
            self.forest.append(tree)

    def predict(self, X):
        # 收集所有树的预测结果并求平均
        predictions = np.array([tree.predict(X) for tree in self.forest])
        return np.mean(predictions, axis=0).round()

# 测试SPORF实现
data = load_breast_cancer()
X, y = data.data, data.target
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 初始化和训练SPORF分类器
sporf_clf = SparseProjectionRandomForestClassifier(n_estimators=10, max_depth=10, density=0.1, random_state=42)
sporf_clf.fit(X_train, y_train)

# 预测并计算AUC
y_pred = sporf_clf.predict(X_test)
roc_auc = roc_auc_score(y_test, y_pred)
print(f"测试集 AUC: {roc_auc:.2f}")