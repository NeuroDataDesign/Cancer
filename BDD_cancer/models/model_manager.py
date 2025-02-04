import os
from models.train_models import train_and_save_model
from models.test_models import test_model

def train_model(model_name):
    """Main file calls this method to train the specified model"""
    train_and_save_model(model_name)

def evaluate_model(model_name):
    """Main file calls this method to evaluate the specified model"""
    test_model(model_name)