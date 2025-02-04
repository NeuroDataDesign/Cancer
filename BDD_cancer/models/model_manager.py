from models.train_models import train_and_save_model
from models.test_models import test_model

def train_model(model_name, X_train, y_train):
    """封装训练函数"""
    train_and_save_model(model_name, X_train, y_train)

def evaluate_model(model_name, X_test, y_test):
    """封装测试函数"""
    return test_model(model_name, X_test, y_test)