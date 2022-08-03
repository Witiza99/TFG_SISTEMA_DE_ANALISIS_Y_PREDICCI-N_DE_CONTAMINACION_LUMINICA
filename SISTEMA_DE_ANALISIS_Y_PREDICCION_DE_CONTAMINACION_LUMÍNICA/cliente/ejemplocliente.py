import requests
import json
import numpy as np


#funcion main para realizar el preporcesamiento necesario de los datos
def main():

	"""TipoLuminaria = input("Introduzca TipoLuminaria: ")
	##controlar el color de suelo para que ponga tipo
	ColorSuelo = input("Introduzca ColorSuelo: ")
	AlturaLuminaria = input("Introduzca AlturaLuminaria: ")
	FlujoLuminicoTotal = input("Introduzca FlujoLuminicoTotal: ")
	TCC = input("Introduzca TCC: ")
	IluminanciaAbajo = input("Introduzca IluminanciaAbajo: ")
	Espectro = input("Introduzca Espectro: ")
	ReflectanciaSuelo = input("Introduzca ReflectanciaSuelo: ")"""

	ColorSuelo = "negro"
	AlturaLuminaria = 80
	FlujoLuminicoTotal = 5400
	TCC = 4257
	IluminanciaAbajo = 1633
	Espectro = 444
	ReflectanciaSuelo = 4



	#parametros para para la peticion get
	getParameters_prediccion = {
	"base_de_datos": "db",
	"ColorSuelo": ColorSuelo,
	"AlturaLuminaria": AlturaLuminaria,
	"FlujoLuminicoTotal": FlujoLuminicoTotal,
	"TCC": TCC,
	"IluminanciaAbajo": IluminanciaAbajo,
	"Espectro": Espectro,
	"ReflectanciaSuelo": ReflectanciaSuelo,
	"IluminanciaSuperior": 0,
	}


	#get para pedir el patron a predecir
	patron_con_prediccion = requests.get("http://127.0.0.1:8000/predecir",#peticion al modulo para obtener la prediccion
		params = getParameters_prediccion
		)
	if not patron_con_prediccion.status_code == 200:#se comprueba si ha fallado
		raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(patron_con_prediccion.status_code, patron_con_prediccion.text))
		return {"Error al extraer datos para la prediccion"}
	#se realiza el preprocesamiento necesario para su correcto funcionamiento
	patron_solucion = (patron_con_prediccion.json())
	IluminanciaSuperior = patron_solucion['IluminanciaSuperior']
	print("La IluminanciaSuperior es :", "{0:.3f}".format(IluminanciaSuperior))
	FlujoSuperior = FlujoLuminicoTotal * (IluminanciaSuperior/IluminanciaAbajo)
	print("La FlujoSuperior es: ", "{0:.3f}".format(FlujoSuperior))
	FHSI = (FlujoSuperior/FlujoLuminicoTotal) * 100
	print("La contaminación Lumincia es: ", "{0:.3f}".format(FHSI),"%")

	if FHSI > 25:
		print("Como la contaminacion lumincia es superior al 25%, dicha luminaria no podria usarse en ningun tipo de areas luminicas")
	elif FHSI > 15:
		print("Como la contaminacion lumincia es superior al 15%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4")
	elif FHSI > 5:
		print("Como la contaminacion lumincia es superior al 5%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4 y E3")
	elif FHSI >1:
		print("Como la contaminacion lumincia es superior al 1%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4, E3 y E2")
	else:
		print("Como la contaminacion lumincia es inferior al 1%, dicha luminaria podria usarse en todos los tipos de areas luminicas E4, E3, E2 y E1")


if __name__ == "__main__":
	main()