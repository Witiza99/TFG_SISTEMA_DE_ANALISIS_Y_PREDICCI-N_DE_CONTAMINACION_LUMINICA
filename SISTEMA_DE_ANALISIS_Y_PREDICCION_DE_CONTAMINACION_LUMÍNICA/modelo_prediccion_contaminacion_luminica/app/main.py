#paquetes necesarios para el funcionamiento del programa
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from typing import Optional
from cmath import nan
import numpy as np
import pandas as pd

import mysql.connector #paquete para insertar en la bd de mysql
"""
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
"""




app = FastAPI()


#**********************************************************************************************************************************************
#*****************************************************************PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************

@app.get("/")#cuando se realiza un get en la api no se realiza ninguna accion
def read_root():
	return {"No se realiza ninguna acción"}


@app.get("/insertarPatrones/")#funcion para insertar datos en la bd
def read_root(base_de_datos: str, archivo_datos: str):

	#guardamos los datos en un dataset de tipo string
	try:
		Dataset = pd.read_csv(archivo_datos)
		#cambio a float aquellos datos guardados en int64
		Dataset['ReflectanciaSuelo'] = Dataset['ReflectanciaSuelo'].astype('float64')
		Dataset['AlturaLuminaria'] = Dataset['AlturaLuminaria'].astype('float64')
		Dataset['FlujoLuminicoTotal'] = Dataset['FlujoLuminicoTotal'].astype('float64')
		Dataset['TCC'] = Dataset['TCC'].astype('float64')
		Dataset['Espectro'] = Dataset['Espectro'].astype('float64')
	except:
		return {"Error con el fichero de datos introducido"}
	print(Dataset)

	#creamos conexion con la bd
	conn = mysql.connector.connect(
	   user='root', password='password', host='mysql', port=3306, database=base_de_datos)

	#creamos un cursor
	cursor = conn.cursor()

	for i in Dataset.index:#recorremos toda la database para almacenarla en la bd
		# Preparamos una solicitud SQL para insertar en la bd.
		sql = """INSERT INTO datos_luminaria(
		   TipoLuminaria, AlturaLuminaria, FlujoLuminicoTotal, TCC, IluminanciaAbajo, Espectro,
		   ColorSuelo, ReflectanciaSuelo, IluminanciaSuperior, FlujoSuperior, FHSI)
		   VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)"""
		
		try:
			# Se ejecuta el comando
			cursor.execute(sql, (Dataset["TipoLuminaria"][i], Dataset["AlturaLuminaria"][i], Dataset["FlujoLuminicoTotal"][i],
			    Dataset["TCC"][i], Dataset["IluminanciaAbajo"][i], Dataset["Espectro"][i],
			    Dataset["ColorSuelo"][i], Dataset["ReflectanciaSuelo"][i], Dataset["IluminanciaSuperior"][i],
			    Dataset["FlujoSuperior"][i], Dataset["FHSI"][i]))

			conn.commit()#guardamos los cambios
		except:
			# Rollback en caso de error con la insercion
   			conn.rollback()

	# Se cierra la conexion
	conn.close()
	
	return {"Exito en la insercion de patrones"}