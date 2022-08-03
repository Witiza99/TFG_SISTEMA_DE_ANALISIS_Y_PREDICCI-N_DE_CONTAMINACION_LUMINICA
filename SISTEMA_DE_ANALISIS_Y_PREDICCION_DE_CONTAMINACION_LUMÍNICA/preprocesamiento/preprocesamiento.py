#paquetes necesarios para el funcionamiento del programa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt#paquete para mostrar histogramas
import seaborn as sns#paquete para mostrar el mapa de correlaciones
from sklearn import preprocessing#paquete de scikit con el estandarizado, normalizacion y el onehotencoder
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

import mysql.connector #paquete para insertar en la bd de mysql


#funcion main para realizar el preporcesamiento necesario de los datos
def main():

	crear_fichero()#creamos un fichero extraido de la bd para trabajar
	fichero_datos = "db.csv"

	Dataset = read_data(fichero_datos)#se leen los datos del fichero creado anteriormente para insertarlo en un Dataset
	if Dataset is None:
		print("Error al hacer el dataset a partir del fichero creado de la bd")
		return

	Dataset.hist(figsize=(14,14), xrot=45)#muestro histograma del dataset
	plt.show()

	#guardamos las correlaciones y las mostramos en un mapa
	Datasetcorr = Dataset.corr()
	plt.figure(figsize=(14,14))
	sns.heatmap(Datasetcorr, cmap='RdBu_r')
	plt.show()
	
	#realizamos normalizacion y estandarización del dataset, es necesario que al estandarizar y normalizar este el atributo objetivo
	x = Dataset.values 
	DatasetNormalizado=x[:, :-1]
	DatasetEstandarizado=x[:, :-1]
	auxiliarEstandar = pd.DataFrame(DatasetEstandarizado)

	minmax_scaler = MinMaxScaler()
	x_scaled = minmax_scaler.fit_transform(DatasetNormalizado)
	DatasetNormalizado = pd.DataFrame(x_scaled)
	DatasetNormalizado.hist(figsize=(14,14), xrot=45)
	plt.show()

	estandarizado_scaled= StandardScaler() 
	estandarizadoFinal= estandarizado_scaled.fit_transform(DatasetEstandarizado)
	nombres=auxiliarEstandar.columns
	EstandarizadoFinal =pd.DataFrame(estandarizadoFinal, columns=nombres)
	EstandarizadoFinal.hist(figsize=(14,14), xrot=45)
	plt.show()


#funcion que extrae un fichero a partir de nuestra bd
def crear_fichero():
	#creamos conexion con la bd
	conn = mysql.connector.connect(#conexion de la bd
	   user='root', password='password', host="127.0.0.1", port=3306, database='db')

	#se realiza la peticion de sql para extraer los datos de la bd y exportarlos a fichero csv
	temp = pd.read_sql_query("Select TipoLuminaria, ColorSuelo, AlturaLuminaria, FlujoLuminicoTotal, TCC, IluminanciaAbajo, Espectro, ReflectanciaSuelo, IluminanciaSuperior FROM datos_luminaria", conn)
	temp.to_csv('db.csv', sep=',', index=False, encoding='utf-8')#como read_sql_query no funciona del todo bien con connect es necesario quitar el index

	conn.close()#cerramos conexion con la base de datos


#funcion para leer datos del fichero
def read_data(fichero_datos):

	Dataset = None

	#guardamos los datos en un dataset de tipo string
	Dataset = pd.read_csv(fichero_datos)
	#reordeno las columnas para poner los cardinales en las primeras filas para usar el columnTransformer
	titulos_columnas = ["TipoLuminaria", "ColorSuelo", "AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"]
	Dataset=Dataset.reindex(columns=titulos_columnas)

	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(2)))], remainder= 'passthrough')
	Dataset = np.array(ct.fit_transform(Dataset))#transformo en numpy y necesito cambiar a pandas
	Dataset = pd.DataFrame(Dataset,columns=["Tipo1", "Tipo2", "Tipo3", "Tipo4", "Tipo5", "Tipo6", "Tipo7", "Tipo8", 
		"Negro", "GrisOscuro", "GrisClaro", "Blanco", "Morado", "Rojo", "VerdeClaro", "Verde", "VerdeOscuro", "MarrorClaro", "MarronOscuro",
		 "AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"])

	return Dataset


if __name__ == "__main__":
	main()