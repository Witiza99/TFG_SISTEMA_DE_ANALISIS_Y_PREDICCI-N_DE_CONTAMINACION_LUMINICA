import requests
import json
import numpy as np
import os


#funcion main para realizar el preporcesamiento necesario de los datos
def main():

	flag = True
	while (flag):

		print("Introduzca ColorSuelo: ")
		print("1.Negro\n2.GrisOscuro\n3.GrisClaro\n4.Blanco\n5.Morado\n6.Rojo\n7.VerdeClaro\n8.Verde\n9.VerdeOscuro\n10.MarrorClaro\n11.MarronOscuro\n")
		IndiceColorSuelo = input()
		os.system ("clear")
		print(IndiceColorSuelo)
		try:	
			print("hola")
			IndiceColorSuelo = int(IndiceColorSuelo)
			print(IndiceColorSuelo)
			if (IndiceColorSuelo > 0 and IndiceColorSuelo < 12):
				flag = False
				print("hola2")
				if IndiceColorSuelo == 1:
					ColorSuelo = "negro"
				elif IndiceColorSuelo == 2:
					ColorSuelo = "gris oscuro"
				elif IndiceColorSuelo == 3:
					ColorSuelo = "gris claro"
				elif IndiceColorSuelo == 4:
					ColorSuelo = "blanco"
				elif IndiceColorSuelo == 5:
					ColorSuelo = "morado"
				elif IndiceColorSuelo == 6:
					ColorSuelo = "rojo"
				elif IndiceColorSuelo == 7:
					ColorSuelo = "verde claro"
				elif IndiceColorSuelo == 8:
					ColorSuelo = "verde"
				elif IndiceColorSuelo == 9:
					ColorSuelo = "Verde oscuro"
				elif IndiceColorSuelo == 10:
					ColorSuelo = "marrón claro"
				elif IndiceColorSuelo == 11:
					ColorSuelo = "marrón oscuro"	
			else:
				print("Valor introducido erroneo.")
		except:
			print("Valor introducido erroneo.")
	os.system ("clear")
	AlturaLuminaria = input("Introduzca AlturaLuminaria: ")
	os.system ("clear")
	FlujoLuminicoTotal = input("Introduzca FlujoLuminicoTotal: ")
	os.system ("clear")
	TCC = input("Introduzca TCC: ")
	os.system ("clear")
	IluminanciaAbajo = input("Introduzca IluminanciaAbajo: ")
	os.system ("clear")
	Espectro = input("Introduzca Espectro: ")
	os.system ("clear")
	ReflectanciaSuelo = input("Introduzca ReflectanciaSuelo: ")
	os.system ("clear")

	"""#Philips Coreline Malaga LED BRP102 LED55/740 II DM,negro,160.0,4600.0,4033.0,773.0,452.0,4.0,64.74
	ColorSuelo = "negro"
	AlturaLuminaria = 160.0
	FlujoLuminicoTotal = 4600
	TCC = 4033.0
	IluminanciaAbajo = 773.0
	Espectro = 452.0
	ReflectanciaSuelo = 4.0"""

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
	patron_solucion = (patron_con_prediccion.json())
	IluminanciaSuperior = patron_solucion['IluminanciaSuperior']

	#mostramos el FlujoSuperior y el FHSI una vez calculado
	print("La IluminanciaSuperior es :", "{0:.3f}".format(IluminanciaSuperior))
	FlujoSuperior = float(FlujoLuminicoTotal) * (IluminanciaSuperior/float(IluminanciaAbajo))
	print("La FlujoSuperior es: ", "{0:.3f}".format(FlujoSuperior))
	FHSI = (FlujoSuperior/float(FlujoLuminicoTotal)) * 100
	print("La contaminación Luminica es: ", "{0:.3f}".format(FHSI),"%")

	#dependiendo del flujo mostramos diferentes mensajes
	if FHSI > 25:
		print("Como la contaminacion luminica es superior al 25%, dicha luminaria no podria usarse en ningun tipo de areas luminicas")
	elif FHSI > 15:
		print("Como la contaminacion luminica es superior al 15%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4")
	elif FHSI > 5:
		print("Como la contaminacion luminica es superior al 5%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4 y E3")
	elif FHSI >1:
		print("Como la contaminacion luminica es superior al 1%, dicha luminaria solo podria usarse en el tipo de areas luminicas E4, E3 y E2")
	else:
		print("Como la contaminacion luminica es inferior al 1%, dicha luminaria podria usarse en todos los tipos de areas luminicas E4, E3, E2 y E1")


if __name__ == "__main__":
	main()