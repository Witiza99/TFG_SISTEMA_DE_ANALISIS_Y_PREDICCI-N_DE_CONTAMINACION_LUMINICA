#paquete utilizados
import requests
import json
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

#funcion main para generar las graficas
def main():

	Dataset = read_data("test_tmp.csv")#se leen los datos del fichero creado anteriormente para insertarlo en un Dataset
	if Dataset is None:
		print("Error al hacer el dataset a partir del fichero")
		return

	IluminanciaSuperiorEsperada = []
	IluminanciaSuperiorObtenida = []
	FlujoSuperiorEsperado = []
	FlujoSuperiorObtenido = []
	FlujoLuminicoTotalEsperado = []
	IluminanciaAbajoEsperada = []
	FHSIEsperado = []
	FHSIObtenido = []

	#recorremos el Dataset leido para pasale los datos a la api
	for i in range(0, len(Dataset), 1):

		#parametros para para la peticion get
		getParameters_prediccion = {
		"base_de_datos": "db",
		"ColorSuelo": Dataset.loc[i]["ColorSuelo"],
		"AlturaLuminaria": Dataset.loc[i]["AlturaLuminaria"],
		"FlujoLuminicoTotal": Dataset.loc[i]["FlujoLuminicoTotal"],
		"TCC": Dataset.loc[i]["TCC"],
		"IluminanciaAbajo": Dataset.loc[i]["IluminanciaAbajo"],
		"Espectro": Dataset.loc[i]["Espectro"],
		"ReflectanciaSuelo": Dataset.loc[i]["ReflectanciaSuelo"],
		"IluminanciaSuperior": 0,
		}

		#get para pedir el patron a predecir para cada uno
		patron_con_prediccion = requests.get("http://127.0.0.1:8000/predecir",#peticion al modulo para obtener la prediccion
			params = getParameters_prediccion
			)
		if not patron_con_prediccion.status_code == 200:#se comprueba si ha fallado
			raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(patron_con_prediccion.status_code, patron_con_prediccion.text))
			return {"Error al extraer datos para la prediccion"}
		patron_solucion = (patron_con_prediccion.json())
		patron_solucion = patron_solucion['IluminanciaSuperior']

		#almacenamos datos necesarios para poder calcular despues el flujo superior y FHSI
		IluminanciaAbajoEsperada.append(Dataset.loc[i]["IluminanciaAbajo"])
		FlujoLuminicoTotalEsperado.append(Dataset.loc[i]["FlujoLuminicoTotal"])

		IluminanciaSuperiorEsperada.append(Dataset.loc[i]["IluminanciaSuperior"])
		IluminanciaSuperiorObtenida.append(patron_solucion)

	

	for w in range(len(IluminanciaAbajoEsperada)):
		#realizamos los calculos de FHSI y FlujoSuperior
		FlujoSuperiorEsperado.append(FlujoLuminicoTotalEsperado[w] * (IluminanciaSuperiorEsperada[w])/IluminanciaAbajoEsperada[w])
		FlujoSuperiorObtenido.append(FlujoLuminicoTotalEsperado[w] * (IluminanciaSuperiorObtenida[w])/IluminanciaAbajoEsperada[w])

		FHSIEsperado.append(round(((FlujoSuperiorEsperado[w]/FlujoLuminicoTotalEsperado[w]) * 100), 3))
		FHSIObtenido.append(round(((FlujoSuperiorObtenido[w]/FlujoLuminicoTotalEsperado[w]) * 100), 3))

	#grafica para la iluminancia
	fig, ax = plt.subplots(figsize=(20, 10))
	ax.autoscale(enable=None, axis="x", tight=True)
	ax.set_ylim(bottom=0, top=300)
	ax.plot( IluminanciaSuperiorObtenida, 'r-', label='predicho')
	ax.plot( IluminanciaSuperiorEsperada, 'b-', label='real')
	plt.ylabel('IluminanciaSuperior')
	plt.xlabel('patron')
	plt.xticks(rotation = '90'); 
	plt.legend()
	plt.savefig("testIluminacion.png",bbox_inches='tight', dpi=100)

	#grafica para la contaminacion luminica
	fig, ax = plt.subplots(figsize=(20, 10))
	ax.autoscale(enable=None, axis="x", tight=True)
	ax.set_ylim(bottom=0, top=30)
	ax.plot( FHSIObtenido, 'r-', label='predicho')
	ax.plot( FHSIEsperado, 'b-', label='real')
	plt.ylabel('ContaminacionLuminica')
	plt.xlabel('patron')
	plt.xticks(rotation = '90'); 
	plt.legend()
	plt.savefig("testContaminacionLuminica.png",bbox_inches='tight', dpi=100)



#funcion para leer datos del fichero
def read_data(fichero_datos):

	Dataset = None

	#guardamos los datos en un dataset de tipo string
	Dataset = pd.read_csv(fichero_datos)
	#reordeno las columnas para poner los cardinales en las primeras filas para usar el columnTransformer
	titulos_columnas = ["ColorSuelo", "AlturaLuminaria", "FlujoLuminicoTotal", "TCC", "IluminanciaAbajo", "Espectro", "ReflectanciaSuelo", "IluminanciaSuperior"]
	Dataset=Dataset.reindex(columns=titulos_columnas)

	return Dataset


if __name__ == "__main__":
	main()