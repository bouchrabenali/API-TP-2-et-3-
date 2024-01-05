""""
import json

def load_model_parameters():
    with open('config/model_parameters.json') as f:
        parameters = json.load(f)
    return parameters
"""

import json
import os

def load_model_parameters():
    current_dir = os.path.dirname(__file__)
    file_path = os.path.join(current_dir, 'model_parameters.json')
    with open(file_path) as f:
        params = json.load(f)
    return params
