#paquetes necesarios para el funcionamiento del programa
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import numpy as np
import pandas as pd
from typing import Optional

import mysql.connector #paquete para insertar en la bd de mysql

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

#import requests
import json
import pickle


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



@app.get("/entrenar/")#get que entrena el modelo
def read_root(base_de_datos: str, crear_nuevo_fichero_entrenamiento: Optional[bool] = True):

	#creamos el fichero para entrenar con los datos que deseamos de la base de datos
	if crear_nuevo_fichero_entrenamiento:
		if -1 == crear_fichero_entrenamiento(base_de_datos):#funcion que crea el fichero con los datos de la bd
			return {"Error al crear el archivo con los patrones de la bd"}

	fichero_entrenamiento = "db.csv"

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(fichero_entrenamiento)
	if Dataset is None or train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	listaErrorMSE = np.zeros(shape=(5), dtype=float)

	#creamos una lista con el nombre de todos los modelos
	NombreModelo = ["SVR", "DecisionTreeRegressor", "RandomForest", "LinearRegression", "SGDRegressor"]

	#entrenamos con los distintos modelo para ver cual es el mejor
	print("Entrenando " + NombreModelo[0])
	listaErrorMSE[0] = entrenarSVR(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[1])
	listaErrorMSE[1] = entrenarArbolDecision(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[2])
	listaErrorMSE[2] = entrenarRandomForest(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[3])
	listaErrorMSE[3] = entrenarRegresionLineal(scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[4])
	listaErrorMSE[4] = entrenarSDGRegressor(scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	mejorModelo = 999999
	menorMSE = 999999

	for iterador in range(len(listaErrorMSE)):#comprobamos cual genero un MSE menor
		if listaErrorMSE[iterador] < menorMSE:
			menorMSE = listaErrorMSE[iterador]
			mejorModelo = iterador

	print("El mejor modelo es " + NombreModelo[mejorModelo] + " con un MSE de: " + str(menorMSE))

	with open(NombreModelo[iterador]+".pickle", 'rb') as fr: #cargamos el modelo ya entrenado, para guardarlo con el nombre de mejor modelo
		modelo = pickle.load(fr)

	#se almacena el mejor modelo
	guardar_modelo_y_scaler(modelo, scaler, "mejorModelo")

	return{"Modelos guardados con exito"}



@app.get("/predecir/")#get que predice la iluminancia superior

def read_root(ColorSuelo: str, AlturaLuminaria: str, FlujoLuminicoTotal: str, TCC: str,
	IluminanciaAbajo: str, Espectro: str, ReflectanciaSuelo: str, IluminanciaSuperior: str):

	patronesDict = {
		'IluminanciaSuperior':''
	}

	fichero_entrenamiento = "db.csv"
	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs_inutilizado, train_outputs_inutilizado, test_inputs_inutilizado, test_outputs_inutilizado, scaler_inutilizado= read_data(fichero_entrenamiento, prediccion = True)
	if Dataset is None:
		print("Error con el fichero")
		return {"Error con el fichero"}


	with open("mejorModelo.pickle", 'rb') as fr: #cargamos el modelo ya entrenado
		model = pickle.load(fr)
	with open("mejorModeloscaler.pickle", 'rb') as fr: #cargamos el scaler del modelo para poder preprocesar el patron
		scaler = pickle.load(fr)

	vector_datos_cliente = [ColorSuelo,AlturaLuminaria,FlujoLuminicoTotal,TCC,IluminanciaAbajo,Espectro,ReflectanciaSuelo,IluminanciaSuperior]
	patron_del_cliente = pd.DataFrame([vector_datos_cliente], columns = ['ColorSuelo','AlturaLuminaria','FlujoLuminicoTotal',
		'TCC','IluminanciaAbajo','Espectro','ReflectanciaSuelo','IluminanciaSuperior'])

	patrones_fusionados = patron_cliente_mas_dataset(Dataset,patron_del_cliente)
	patrones_fusionados = patrones_fusionados[-1:]
	patrones_fusionados = scaler.transform(patrones_fusionados)[:,:-1]#preprocesado y escalado para ajustar el patron a predecir
	
	#se predicen resultados, se extrane los resultados que buscamos y se desescala para que tenga sentido
	valor_prediccion = model.predict(patrones_fusionados)#predecimos la iluminancia superior
	valor_prediccion = np.column_stack((patrones_fusionados,valor_prediccion))
	valor_prediccion = scaler.inverse_transform(valor_prediccion)[:,-1]#realizamos el inverso al preprocesado y el escalado
	valor_prediccion = np.round_(valor_prediccion, decimals=3)

	patronesDict["IluminanciaSuperior"] = float(valor_prediccion)

	#Devuelvo el resultado
	return patronesDict

#**********************************************************************************************************************************************
#*************************************************************FIN_PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************




#**********************************************************************************************************************************************
#*************************************************************FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************

#funcion que extrae un fichero a partir de nuestra bd
def crear_fichero_entrenamiento(base_de_datos):

	try:

		#creamos conexion con la bd
		conn = mysql.connector.connect(#conexion de la bd
		   user='root', password='password', host=mysql, port=3306, database=base_de_datos)

		#se realiza la peticion de sql para extraer los datos de la bd y exportarlos a fichero csv
		temp = pd.read_sql_query("Select ColorSuelo, AlturaLuminaria, FlujoLuminicoTotal, TCC, IluminanciaAbajo, Espectro, ReflectanciaSuelo, IluminanciaSuperior FROM datos_luminaria", conn)
		temp.to_csv('db.csv', sep=',', index=False, encoding='utf-8')#como read_sql_query no funciona del todo bien con connect es necesario quitar el index

		conn.close()#cerramos conexion con la base de datos
		return 0

	except:
		return -1


#funcion para leer datos del fichero
def read_data(fichero_datos, prediccion = False):

	try:

		Dataset = None
		Inputs = None
		Outputs = None
		train_inputs = None
		train_outputs = None
		test_inputs = None
		test_outputs = None 
		scaler = None

		#guardamos los datos en un dataset de tipo string
		Dataset = pd.read_csv(fichero_datos)
		#reordeno las columnas para poner los cardinales en las primeras filas para usar el columnTransformer
		titulos_columnas = ["ColorSuelo", "AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"]
		Dataset=Dataset.reindex(columns=titulos_columnas)

		if prediccion == False:
			ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(1)))], remainder= 'passthrough')
			Dataset = np.array(ct.fit_transform(Dataset))#transformo en numpy y necesito cambiar a pandas
			Dataset = pd.DataFrame(Dataset,columns=["Negro", "GrisOscuro", "GrisClaro", "Blanco", "Morado", "Rojo", "VerdeClaro", "Verde", "VerdeOscuro", "MarrorClaro", "MarronOscuro",
				"AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"])

			#dividimos en 90% de datos para entreno y el resto para test
			train, test = train_test_split(Dataset, test_size= 0.10)

			#realizamos un preprocesamiento para normalizar los datos
			scaler = MinMaxScaler().fit(train)
			train = scaler.transform(train)
			test = scaler.transform(test)

			#finalmente se dividen los datos en inputs y outputs
			train_inputs=train[:, :-1]
			train_outputs=train[:, -1:]
			test_inputs=test[:, :-1]
			test_outputs=test[:, -1:]

			#transforma a una unica columna para evitar warnings
			train_outputs = train_outputs.ravel() 
			test_outputs = test_outputs.ravel()

		return Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler

	except:

		Dataset = None
		train_inputs = None
		train_outputs = None
		test_inputs = None
		test_outputs = None
		scaler = None

		return Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler


#funcion para guardar un modelo con su scaler
def guardar_modelo_y_scaler(modelo, scaler, name):

	#guardamos el modelo y el scaler
	with open(name+'.pickle', 'wb') as fw:
		pickle.dump(modelo, fw)
	with open(name+'scaler.pickle', 'wb') as fw:
		pickle.dump(scaler, fw)


#funcion que entrena con el modelo SVR
def entrenarSVR(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosParametrosSVR", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	vector = np.array([1e-2,1e-1,1e0,1e1,1e2,1e3])
	mejor_mse = 999999
	mae = 999999
	mejor_C = 999999
	mejor_Gamma = 999999

	for C in vector:#entrenamos probando distintas c y gammas
		for Gamma in vector:
			print("C=%f y Gamma=%f" % (C, Gamma))

			# Entrenamos el modelo
			modelo = svm.SVR(kernel='rbf',C=C, gamma=Gamma)
			modelo.fit(train_inputs, train_outputs)
			y_pred=modelo.predict(test_inputs)
			test_mse = mean_squared_error(test_outputs, y_pred) #MSE
			test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

			f.write("MSE & MAE Final con C=%f y G=%f: \t%f %f\n" % (C, Gamma, test_mse, test_mae))#lo vamos almacenando en un fichero

			if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
				mejor_mse = test_mse
				mejor_C = C
				mejor_Gamma = Gamma
				mae = test_mae

				#se almacena el modelo y el scaler
				guardar_modelo_y_scaler(modelo,scaler,"SVR")


	f.write("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que entrena con el modelo ArbolDecision
def entrenarArbolDecision(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosParametrosArbolDecision", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_splitter = {"best", "random"}
	v_min_samples_split = {0.1 ,0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	v_min_samples_leaf = {0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8}
	mejor_mse = 99999
	mae = 999999
	mejor_splitter = None
	mejor_min_samples_split = 999999
	mejor_min_samples_leaf = 999999


	#entrenamos probando distintas combinaciones
	for splitter in v_splitter:
		for min_samples_split in v_min_samples_split:
			for min_samples_leaf in v_min_samples_leaf:
				print("splitter=%s, min_samples_split=%f, min_samples_leaf=%f\n" % (splitter, min_samples_split, min_samples_leaf))

				#Entrenamos el modelo
				modelo = tree.DecisionTreeRegressor(splitter=splitter, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf)
				modelo.fit(train_inputs, train_outputs)
				y_pred=modelo.predict(test_inputs)
				test_mse = mean_squared_error(test_outputs, y_pred) #MSE
				test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

				f.write("MSE & MAE Final con splitter=%s, min_samples_split=%f y min_samples_leaf=%f: \t%f %f\n" % (splitter, min_samples_split, min_samples_leaf, test_mse, test_mae))

				if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
					mejor_mse = test_mse
					mejor_splitter = splitter
					mejor_min_samples_split = min_samples_split
					mejor_min_samples_leaf = min_samples_leaf
					mae = test_mae

				#se almacena el modelo y el scaler
				guardar_modelo_y_scaler(modelo,scaler,"ArbolDecision")

	f.write("Los mejores parametros para el arbol de decision es splitter->"+ str(mejor_splitter)+", min_samples_split->"+ str(mejor_min_samples_split)+", min_samples_leaf->"+ str(mejor_min_samples_leaf)+ " con un MSE de:"+ str(mejor_mse) + " y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el arbol de decision es splitter->"+ str(mejor_splitter)+", min_samples_split->"+ str(mejor_min_samples_split)+", min_samples_leaf->"+ str(mejor_min_samples_leaf)+ " con un MSE de:"+ str(mejor_mse) + " y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que entrena con el modelo RandomForest
def entrenarRandomForest(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosParametrosRandomForest", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_n_estimators = {100,300}
	v_max_features = {0.01, 0.05, 0.1, 0.2, 0.3}
	v_min_samples_split = {2, 50, 100}
	v_min_samples_leaf = {2, 50, 100}
	mejor_mse = 99999
	mae = 999999
	mejor_n_estimators = 999999
	mejor_min_samples_split = 999999
	mejor_min_samples_leaf = 999999
	mejor_max_features = 999999

	#entrenamos probando distintas combinaciones
	for n_estimators in v_n_estimators:
		for max_features in v_max_features:
			for min_samples_split in v_min_samples_split:
				for min_samples_leaf in v_min_samples_leaf:
					print("n_estimators=%d, max_features=%f, min_samples_split=%f, min_samples_leaf=%f \n" % (n_estimators, max_features, min_samples_split, min_samples_leaf))

					#Entrenamos el modelo
					modelo = ensemble.RandomForestRegressor(min_weight_fraction_leaf=0.2, max_samples= 0.5,n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=True, n_jobs=4)
					modelo.fit(train_inputs, train_outputs)
					y_pred=modelo.predict(test_inputs)
					test_mse = mean_squared_error(test_outputs, y_pred) #MSE
					test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

					f.write("MSE & MAE Final con n_estimators=%d, v_max_features=%f, min_samples_split=%f, min_samples_leaf=%f= \t%f %f\n" % (n_estimators, max_features, min_samples_split, min_samples_leaf, test_mse, test_mae))

					if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
						mejor_mse = test_mse
						mae = test_mae
						mejor_n_estimators = n_estimators
						mejor_min_samples_split = min_samples_split
						mejor_min_samples_leaf = min_samples_leaf
						mejor_max_features = max_features

						#se almacena el modelo y el scaler
						guardar_modelo_y_scaler(modelo,scaler,"RandomForest")

	f.write("Los mejores parametros para el random forest es n_estimators->"+ str(mejor_n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el random forest es n_estimators->"+ str(mejor_n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que entrena con el modelo RegresionLineal
def entrenarRegresionLineal(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosParametrosRegresionLineal", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	mejor_mse = 999999
	mae = 999999

	#Entrenamos el modelo
	modelo = linear_model.LinearRegression()
	modelo.fit(train_inputs, train_outputs)
	y_pred=modelo.predict(test_inputs)
	test_mse = mean_squared_error(test_outputs, y_pred) #MSE
	test_mae = mean_absolute_error(test_outputs, y_pred) #MAE
	
	mejor_mse = test_mse
	mae = test_mae

	f.write("MSE & MAE Final con para la regresion lineal: \t%f %f\n" % (test_mse, test_mae))
	f.close()

	#se almacena el modelo y el scaler
	guardar_modelo_y_scaler(modelo,scaler,"RegresionLineal")
	
	print("******************")
	print("La regresion lineal no tiene parametros que buscar, siendo su MSE: "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse


#funcion que entrena con el modelo SDGRegressor
def entrenarSDGRegressor(scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosParametrosSGCRegressor", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_validation = {0.1, 0.2, 0.3, 0.4}
	v_early_stopping = {True, False}
	v_shuffle = {True, False}
	mejor_mse = 99999
	mae = 999999
	mejor_validation = 999999
	mejor_early_stopping = None
	mejor_shuffle = None
	mejor_alpha = 999999

	#entrenamos probando distintas combinaciones
	for early_stopping in v_early_stopping:
		for validation in v_validation:
			for shuffle in v_shuffle:
					for alpha in range(1,21,1):
						print("early_stopping=%r, validation=%f, shuffle=%r, alpha=%f \n" % (early_stopping, validation, shuffle, alpha/10000))

						#Entrenamos el modelo
						modelo = linear_model.SGDRegressor(early_stopping=early_stopping, validation_fraction=validation, shuffle=shuffle,alpha=alpha/10000)
						modelo.fit(train_inputs, train_outputs)
						y_pred=modelo.predict(test_inputs)
						test_mse = mean_squared_error(test_outputs, y_pred)
						test_mae = mean_absolute_error(test_outputs, y_pred) #MAE
						
						f.write("MSE & MAE Final con early_stopping=%r, validation=%f, shuffle=%r, alpha=%f= \t%f %f\n \n" % (early_stopping, validation, shuffle, alpha/10000, test_mse, test_mae))

						if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
							mejor_mse = test_mse
							mae = test_mae
							mejor_early_stopping = early_stopping
							mejor_validation = validation
							mejor_shuffle = shuffle
							mejor_alpha = alpha/10000

							#se almacena el modelo y el scaler
							guardar_modelo_y_scaler(modelo,scaler,"SGDRegressor")
							

	f.write("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



#funcion para juntar el patron del cliente con el dataset
def patron_cliente_mas_dataset(Dataset,patron_del_cliente):

	Dataset = pd.concat((Dataset, patron_del_cliente))

	#aplicamos el onehotencoder
	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(1)))], remainder= 'passthrough')
	Dataset = np.array(ct.fit_transform(Dataset))#transformo en numpy y necesito cambiar a pandas
	Dataset = pd.DataFrame(Dataset,columns=["Negro", "GrisOscuro", "GrisClaro", "Blanco", "Morado", "Rojo", "VerdeClaro", "Verde", "VerdeOscuro", "MarrorClaro", "MarronOscuro",
			 "AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"])

		
	return Dataset


#**********************************************************************************************************************************************
#*********************************************************FIN_FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************