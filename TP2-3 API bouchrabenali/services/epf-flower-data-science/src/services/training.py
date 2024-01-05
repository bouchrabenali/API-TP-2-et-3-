""""
import joblib
from sklearn.ensemble import RandomForestClassifier
#from src.config import load_model_parameters
#from config import load_model_parameters

from ..config.config import load_model_parameters

def train_iris_model(X_train, y_train):
    params = load_model_parameters()

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    joblib.dump(model, 'src/models/iris_model.pkl')

    return model

"""
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from ..config.config import load_model_parameters

def train_iris_model(X_train, y_train):
    params = load_model_parameters()

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    model_dir = 'src/models'
    if not os.path.isdir(model_dir):
        os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, 'iris_model.pkl')

    joblib.dump(model, model_path)

    return model


