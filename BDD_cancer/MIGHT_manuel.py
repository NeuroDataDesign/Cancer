import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 定义MIGHT Tree类
class MightTree:
    def __init__(self, max_depth=None, random_state=None):
        self.architecture_tree = DecisionTreeClassifier(max_depth=max_depth, random_state=random_state)
        self.leaf_probs = None  # 存储每个叶节点的概率
        self.random_state = random_state

    def fit(self, X_arch, y_arch, X_leaf, y_leaf):
        # 使用Architecture Group训练决策树
        self.architecture_tree.fit(X_arch, y_arch)

        # 获取Leaf Group的叶节点索引
        leaf_indices = self.architecture_tree.apply(X_leaf)

        # 计算每个叶节点中的正类概率
        self.leaf_probs = {}
        for leaf_index in np.unique(leaf_indices):
            leaf_samples = (leaf_indices == leaf_index)
            # 计算叶节点中正类样本的比例
            self.leaf_probs[leaf_index] = y_leaf[leaf_samples].mean()

    def predict(self, X_test):
        # 预测Test Group中每个样本的概率
        leaf_indices = self.architecture_tree.apply(X_test)
        return np.array([self.leaf_probs.get(leaf, 0.5) for leaf in leaf_indices])

# 定义MIGHT Forest类
class MightForest:
    def __init__(self, n_trees=100, max_depth=None, random_state=None):
        self.n_trees = n_trees
        self.max_depth = max_depth
        self.random_state = random_state
        self.forest = []

    def fit(self, X, y):
        n_samples = len(X)
        for i in range(self.n_trees):
            # 设置随机状态以保证每棵树的随机性
            rs = self.random_state + i if self.random_state is not None else None

            # 随机有放回抽样160个样本
            sample_indices = np.random.choice(n_samples, size=160, replace=True)
            X_sample = X[sample_indices]
            y_sample = y[sample_indices]

            # 分割为Architecture Group（40%）、Leaf Group（40%）和Test Group（20%）
            X_arch, X_temp, y_arch, y_temp = train_test_split(
                X_sample, y_sample, test_size=0.6, stratify=y_sample, random_state=rs
            )
            X_leaf, X_test, y_leaf, y_test = train_test_split(
                X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=rs
            )

            # 构建并训练MIGHT Tree
            tree = MightTree(max_depth=self.max_depth, random_state=rs)
            tree.fit(X_arch, y_arch, X_leaf, y_leaf)

            # 存储树和对应的测试数据
            self.forest.append((tree, X_test))

    def predict(self, X):
        # 初始化每个样本的得分和计数
        all_scores = np.zeros(len(X))
        counts = np.zeros(len(X))

        for tree, X_test in self.forest:
            # 获取当前树对X_test的预测得分
            scores = tree.predict(X_test)

            # 对于X_test中的每个样本，累加其得分到all_scores
            for j, x_test in enumerate(X_test):
                idx = np.where((X == x_test).all(axis=1))[0]
                if len(idx) > 0:
                    all_scores[idx[0]] += scores[j]
                    counts[idx[0]] += 1

        # 计算平均得分
        avg_scores = np.zeros(len(X))
        avg_scores[counts > 0] = all_scores[counts > 0] / counts[counts > 0]
        avg_scores[counts == 0] = 0.5  # 对于未出现的样本，设置默认概率为0.5

        return avg_scores

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 将数据集拆分为训练集和测试集
X_train_full, X_test_full, y_train_full, y_test_full = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

# 初始化并训练MIGHT Forest
might_forest = MightForest(n_trees=100, max_depth=5, random_state=42)
might_forest.fit(X_train_full, y_train_full)

# 在测试集上进行预测
y_pred_might = might_forest.predict(X_test_full)

# 计算AUC值
roc_auc_might = roc_auc_score(y_test_full, y_pred_might)
print(f"MIGHT模型测试集 AUC: {roc_auc_might:.2f}")