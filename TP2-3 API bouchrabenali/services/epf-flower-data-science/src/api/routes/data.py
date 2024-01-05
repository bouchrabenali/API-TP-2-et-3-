import opendatasets as od
import pandas as pd
from fastapi import HTTPException
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from fastapi import APIRouter
from src.services.training import train_iris_model
from src.schemas.iris import IrisModel
from src.schemas.iris import IrisModelPrediction
from src.services.firebase_setup import initialize_firestore
from typing import List
import logging
import joblib
from fastapi import APIRouter, HTTPException
from src.services.utils import setup_logger
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from src.api.routes.authentication import create_access_token, get_current_user
from src.services.rate_limiter import limiter
from fastapi import APIRouter, Request
from src.services.rate_limiter import limiter, get_remote_address
from fastapi import HTTPException

router = APIRouter()
logger = setup_logger()

@router.post("/token")
async def generate_token(form_data: OAuth2PasswordRequestForm = Depends()):
    if form_data.username == "bouchra" and form_data.password == "bouchra":
        access_token = create_access_token(data={"sub": form_data.username})
        return {"access_token": access_token, "token_type": "bearer"}
    else:
        raise HTTPException(status_code=401, detail="Invalid credentials")



@router.get("/download-iris")
async def download_iris_dataset():
    logger.info("Download Iris Data")
    try:
        dataset_url = 'https://www.kaggle.com/datasets/uciml/iris'
        od.download(dataset_url, 'src/data/')
        logger.info("Dataset downloaded successfully")
        return {"message": "Dataset downloaded successfully"}
    except Exception as e:
        logger.error(f"Error in /download-iris route: {e}")
        raise HTTPException(status_code=500, detail="An internal error occurred")

@router.get("/load-iris")
async def load_iris_dataset():
    logger.info("Loading Iris Data")
    try:
        df = pd.read_csv('src/data/iris/Iris.csv')
        logger.info("Iris Data loaded successfully")
        return df.to_dict(orient='records')
    except FileNotFoundError as e:
        logger.error(f"Error in /load-iris route: {e}")
        raise HTTPException(status_code=404, detail="File not found")

@router.post("/process-iris")
async def process_iris_dataset(iris_data_list: List[IrisModel]):
    logger.info("Processing Iris Data")
    try:
        df = pd.DataFrame([iris.dict() for iris in iris_data_list])
        df_numeric = df.drop(columns=['species'])
        scaler = StandardScaler()
        df_scaled = scaler.fit_transform(df_numeric)
        df_scaled = pd.DataFrame(df_scaled, columns=df_numeric.columns).to_dict(orient='records')
        logger.info("Iris Data processed successfully")
        return df_scaled
    except Exception as e:
        logger.error(f"Error in /process-iris route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/split-iris")
async def split_iris_dataset(df: List[IrisModel], test_size: float = 0.2):
    logger.info("Splitting Iris Data")
    try:
        X_train, X_test = train_test_split(df, test_size=test_size)
        logger.info("Iris Data split successfully")
        return {"train": X_train, "test": X_test}
    except Exception as e:
        logger.error(f"Error in /split-iris route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/train-model")
async def train_model(data: List[IrisModel], current_user: str = Depends(get_current_user)):
    logger.info("Training Iris Model")
    try:
        df = pd.DataFrame([item.dict() for item in data])
        X = df.drop(columns=['species'])
        y = df['species']
        model = train_iris_model(X, y)
        logger.info("Iris Model trained successfully")
        return {"message": "Model trained successfully"}
    except Exception as e:
        logger.error(f"Error in /train-model route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/predict")
@limiter.limit("5/minute")
async def make_prediction(request: Request, data: List[IrisModelPrediction]):
    logger.info("Making predictions")
    try:
        model = joblib.load('src/models/iris_model.pkl')
        df = pd.DataFrame([item.dict() for item in data])
        predictions = model.predict(df)
        logger.info("Predictions made successfully")
        return {"predictions": predictions.tolist()}
    except Exception as e:
        logger.error(f"Error in /predict route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

db = initialize_firestore()

@router.get("/get-parameters")
async def get_parameters():
    logger.info("Getting parameters")
    try:
        params_ref = db.collection('parameters').document('model_params')
        params = params_ref.get()
        if params.exists:
            logger.info("Parameters retrieved successfully")
            return params.to_dict()
        else:
            return {"message": "No parameters found"}
    except Exception as e:
        logger.error(f"Error in /get-parameters route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.put("/update-parameters")
async def update_parameters(new_params: dict):
    logger.info("Updating parameters")
    try:
        params_ref = db.collection('parameters').document('model_params')
        params_ref.update(new_params)
        logger.info("Parameters updated successfully")
        return {"message": "Parameters updated"}
    except Exception as e:
        logger.error(f"Error in /update-parameters route: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/add-parameters")
async def add_parameters(new_params: dict):
    logger.info("Adding new parameters")
    try:
        params_ref = db.collection('parameters').document('new_model_params')
        params_ref.set(new_params)
        logger.info("New parameters added successfully")
        return {"message": "Parameters added"}
    except Exception as e:
        logger.error(f"Error in /add-parameters route: {e}")
        raise HTTPException(status_code=500, detail=str(e))
