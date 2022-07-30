#paquetes necesarios para el funcionamiento del programa
from cmath import nan
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
import numpy as np
import pandas as pd

from sklearn import svm
from sklearn import linear_model
from sklearn import tree
from sklearn import ensemble
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.multioutput import MultiOutputRegressor

from datetime import datetime
from datetime import timedelta

#import requests
import json
import pickle
#import psycopg2

app = FastAPI()


#**********************************************************************************************************************************************
#*****************************************************************PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************

@app.get("/")#cuando se realiza un get en la api no se realiza ninguna accion
def read_root():
	return {"No se realiza ninguna acción"}
