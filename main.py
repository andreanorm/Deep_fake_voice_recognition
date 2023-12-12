import pandas as pd
import numpy as np
import librosa
import os
# from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, StratifiedKFold, cross_validate, RepeatedStratifiedKFold
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import time
# from tpot import TPOTClassifier
# from tqdm.auto import tqdm
# from google.cloud import storage

from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
import joblib

app = FastAPI()

# Allowing all middleware is optional, but good practice for dev purposes
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

def initialize_empty_df():
    df = pd.DataFrame({
        "chroma_stft":[],
        "rms":[],
        "spectral_centroid":[],
        "spectral_bandwidth":[],
        "rolloff":[],
        "zero_crossing_rate":[]
    })
    for mfcc in [f"mfcc{i+1}" for i in range(20)]:
        df[mfcc] = ""
    df["LABEL"] = ""
    return df

df_columns = initialize_empty_df().columns

def preprocess_data(y, sr, label):
    iter_ = int(y.shape[0]/sr)
    features = []
    for i in range(iter_):
        y_seg = y[i*sr:(sr*i+sr)]
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_seg, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y_seg))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_seg, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_seg, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_seg))
        mfcc = np.mean(librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=20),axis=1)
        features.append([chroma_stft,rms,spec_cent,spec_bw,rolloff,zcr,*mfcc, label])

    df_indiv = pd.DataFrame(features, columns = df_columns)
    return df_indiv

def load_model():
    latest_model = joblib.load("models/20231212-123732.pkl")
    return latest_model

app.state.model = load_model()

@app.post("/predict/")
async def predict(
        file: UploadFile
    ):
    """
    Make a single audio prediction.
    """
    with open(file.filename, "wb") as f:
        f.write(file.file.read())

    model = app.state.model
    assert model is not None

    y, sr = librosa.load(file.filename)
    df_processed = preprocess_data(y, sr, 0)
    X_processed = df_processed.drop(columns="LABEL")

    y_pred = pd.DataFrame(model.predict(X=X_processed)).value_counts(normalize=True)
    if y_pred.index[0][0] == 0:
        prediction = "REAL"
    else:
        prediction = "FAKE"
    proba = y_pred[0]

    return JSONResponse(content=jsonable_encoder({"prediction":prediction, "probability":round(proba,2)}))


@app.get("/")
def root():
    return {'greeting': "Hello everyone ! :)"}
