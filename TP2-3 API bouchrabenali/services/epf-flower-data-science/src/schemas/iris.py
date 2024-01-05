from pydantic import BaseModel

class IrisModel(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float
    species: str

class IrisModelPrediction(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float