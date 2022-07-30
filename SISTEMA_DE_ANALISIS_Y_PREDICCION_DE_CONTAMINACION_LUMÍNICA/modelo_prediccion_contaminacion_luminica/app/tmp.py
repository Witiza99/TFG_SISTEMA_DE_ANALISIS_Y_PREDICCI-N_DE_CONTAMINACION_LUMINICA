@app.get("/entrenar/")#get que dependiendo de los parametros de entrada realizara distintas funciones
"""
def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	"""if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}"""#en este caso no es necesario ya que se realiza por fichero directamente

	train_file = "DatasetGeneracion.csv"

	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file, semana, outputs)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	listaMSE = np.zeros(shape=(4), dtype=float)

	#creamos una lista con el nombre de todos los modelos
	NombreModelo = ["SVR", "LinearRegression", "RandomForest", "SGDRegressor"]

	#entrenamos con los distintos modelo para ver cual es el mejor
	print("Entrenando " + NombreModelo[0])
	listaMSE[0] = entrenarSVR(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[1])
	listaMSE[1] = entrenarRegresionLineal(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[2])
	listaMSE[2] = entrenarRandomForest(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)
	print("Entrenando " + NombreModelo[3])
	listaMSE[3] = entrenarSDGRegressor(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	menorMSE = 999999
	mejorModelo = 999999

	for iterador in range(len(listaMSE)):#comprobamos cual genero un MSE menor
		if listaMSE[iterador] < menorMSE:
			menorMSE = listaMSE[iterador]
			mejorModelo = iterador

	print("El mejor modelo es " + NombreModelo[mejorModelo] + " con un MSE de: " + str(menorMSE))

	if semana:
		with open(NombreModelo[iterador]+"semanal.pickle", 'rb') as fr: #cargamos el modelo ya entrenado, para guardarlo con el nombre de mejor modelo
			model = pickle.load(fr)
	else:
		with open(NombreModelo[iterador]+".pickle", 'rb') as fr: #cargamos el modelo ya entrenado, para guardarlo con el nombre de mejor modelo
			model = pickle.load(fr)

	#se almacena el mejor modelo
	guardar_modelo_y_scaler(model, scaler, "mejorModelo",semana)

	return{"Modelo guardado con exito"}


@app.get("/predecir/")#get que predice los precios a nivel semanal o diario

def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#necesitamos extraer el dataset para realizar el preprocesamiento
	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	train_file = "DatasetGeneracion.csv"
	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs_tmp, train_outputs_tmp, test_inputs_tmp, test_outputs_tmp, scaler_tmp= read_data(train_file, semana, outputs)
	if train_inputs_tmp is None or train_outputs_tmp is None or test_inputs_tmp is None or test_outputs_tmp is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	PrediccionDict = {
		"Name": "Predicción",
		"Description": "",
		"Data": []
	}

	dia = datetime.now()

	if semana:
		with open("mejorModelosemanal.pickle", 'rb') as fr: #cargamos el modelo ya entrenado
			model = pickle.load(fr)
		with open("mejorModelosemanalscaler.pickle", 'rb') as fr: #cargamos el scaler del modelo para poder preprocesar el patron
			scaler = pickle.load(fr)

		#extracion de patron a nivel de semana para poder predecir los precios de la siguiente semana
		v_patrones = get_semana(base_de_datos)#de la base de datos extraemos la última 
		#se realiza el preprocesamiento necesario para su correcto funcionamiento
		v_patrones = formatear_semana(v_patrones, mode='predict')
		v_patrones = preprocessing(Dataset,v_patrones,mode='predict')[-7:, :]
		v_patrones = scaler.transform(v_patrones)[:,:-24]
		#se predicen resultados, se extrane los resultados que buscamos y se desescala para que tenga sentido
		prediccion = model.predict(v_patrones)#predecimos los precios
		prediccion = np.column_stack((v_patrones,prediccion))
		prediccion = scaler.inverse_transform(prediccion)[:,-24:]
		Description = "Predicción semanal de precios"
		DataDict = []
		for j in range(0, len(prediccion), 1):
			ArrayOfValues = []
			for i in range(0, len(prediccion[0]), 1):
				ArrayOfValues.append({"hour": str(i)+":00", "value": str(prediccion[j][i])})

			dia = dia + timedelta(days=1)
			DataDictDay = {
				"Day": dia.strftime("%Y-%m-%-d"),
				"units": "€/MWh",
				"values": ArrayOfValues
			}
			DataDict.append(DataDictDay)

	else:
		with open("mejorModelo.pickle", 'rb') as fr: #cargamos el modelo ya entrenado
			model = pickle.load(fr)
		with open("mejorModeloscaler.pickle", 'rb') as fr: #cargamos el scaler del modelo para poder preprocesar el patron
			scaler = pickle.load(fr)

		#extracion de patron a nivel de dia para poder predecir los precios del dia siguiente
		patrones_get = requests.get("http://bdconapi:8001/patrongeneracion")#peticion a bdconapi para extraer el patron del dia actual
		if not patrones_get.status_code == 200:#se comprueba si ha fallado
			raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(patrones_get.status_code, patrones_get.text))
			return {"Error al extraer datos para obtener precios del dia siguiente"}
		#se realiza el preprocesamiento necesario para su correcto funcionamiento
		v_patrones = np.asarray(json.loads(patrones_get.json()))
		v_patrones = pd.DataFrame(v_patrones, columns = ['Day of Year','Year','Month','Day','First Hour of Period',
			'Is Daylight','Distance to Solar Noon','Average Temperature (Day)','Average Wind Direction (Day)','Average Wind Speed (Day)',
			'Sky Cover','Visibility','Relative Humidity','Average Wind Speed (Period)','Average Barometric Pressure (Period)','Power Generated'])
		v_patrones = preprocessing(Dataset,v_patrones,mode='predict')
		v_patrones = v_patrones[-1:]
		v_patrones = scaler.transform(v_patrones)[:,:-1]#preprocesado y escalado para ajustar los patrones del dia de hoy a predecir
		
		#se predicen resultados, se extrane los resultados que buscamos y se desescala para que tenga sentido
		prediccion = model.predict(v_patrones)#predecimos los precios
		prediccion = np.column_stack((v_patrones,prediccion))
		prediccion = scaler.inverse_transform(prediccion)[:,-1]#realizamos el inverso al preprocesado y el escalado
		prediccion = np.round_(prediccion, decimals=2)
		Description = "Predicción diaria de precios"
		
		ArrayOfValues = []
		for i in range(0, len(prediccion), 1):
			ArrayOfValues.append({"hour": str(i)+":00", "value": str(prediccion[i])})

		dia = dia + timedelta(days=1)
		DataDict = {
			"Day": dia.strftime("%Y-%m-%-d"),
			"units": "€/MWh",
			"values": ArrayOfValues
		}

	
	#cambiar formato
	PrediccionDict['Description'] = Description
	PrediccionDict['Data'] = DataDict
	PrediccionDict = jsonable_encoder(PrediccionDict)
	return JSONResponse(content=PrediccionDict)#devolvemos los precios 


#estas llamadas get son creadas para experimentar con un modelo en concreto
@app.get("/SVR/")#get que dependiendo de los parametros de entrada realizara distintas funciones

def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracion.csv"

	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file, semana, outputs)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando SVR")
	MSE = entrenarSVR(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/RegresionLineal/")#get que dependiendo de los parametros de entrada realizara distintas funciones

def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracion.csv"

	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file, semana, outputs)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando RegresionLineal")
	MSE = entrenarRegresionLineal(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/RandomForest/")#get que dependiendo de los parametros de entrada realizara distintas funciones

def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracion.csv"

	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file, semana, outputs)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando RandomForest")
	MSE = entrenarRandomForest(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}


@app.get("/SGDRegressor/")#get que dependiendo de los parametros de entrada realizara distintas funciones

def read_root(base_de_datos: str, semana: Optional[bool] = False):

	#creamos el fichero a entrenar con los datos que deseamos de la base de datos, aparte se actualiza la base de datos
	if 200 != crear_fichero_entrenamiento(base_de_datos):#funcion donde se actualiza la base de datos y se crea el fichero con los datos a trabajar
		return {"Error": "Error al crear el fichero con patrones"}

	train_file = "DatasetGeneracion.csv"

	if semana:#especificamos el numero de salidas del modelo dependiendo de si es semanal o diario
		outputs = 24
	else:
		outputs = 1

	#llamamos a funcion read_data para extraer los datos del fichero csv a un formato con el que podamos trabajar
	Dataset, train_inputs, train_outputs, test_inputs, test_outputs, scaler= read_data(train_file, semana, outputs)
	if train_inputs is None or train_outputs is None or test_inputs is None or test_outputs is None:
		print("Error con el fichero")
		return {"Error con el fichero"}

	print("Entrenando SGDRegressor")
	MSE = entrenarSDGRegressor(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs)

	return{"Modelo guardado con exito"}

#**********************************************************************************************************************************************
#*************************************************************FIN_PETICIONES_GET***************************************************************
#**********************************************************************************************************************************************



#**********************************************************************************************************************************************
#*************************************************************FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************

def entrenarSVR(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosSVR", "w"); ##archivo donde se almacenan los resultados de los parametros

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
			model = svm.SVR(kernel='rbf',C=C, gamma=Gamma)
			if semana:
				model = MultiOutputRegressor(model)
			model.fit(train_inputs, train_outputs)
			y_pred=model.predict(test_inputs)
			test_mse = mean_squared_error(test_outputs, y_pred) #MSE
			test_mae = mean_absolute_error(test_outputs, y_pred) #MAE

			f.write("MSE & MAE Final con C=%f y G=%f: \t%f %f\n" % (C, Gamma, test_mse, test_mae))#lo vamos almacenando en un fichero

			if mejor_mse > test_mse:#se va guardando la mejor combinacion de parametros
				mejor_mse = test_mse
				mejor_C = C
				mejor_Gamma = Gamma
				mae = test_mae

				#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
				guardar_modelo_y_scaler(model,scaler,"SVR",semana)


	f.write("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros son C:"+ str(mejor_C) +" y Gamma:"+ str(mejor_Gamma) +" con un MSE de "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



def entrenarRegresionLineal(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosRegresionLineal", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	mejor_mse = 999999
	mae = 999999

	#Entrenamos el modelo
	model = linear_model.LinearRegression()
	if semana:
		model = MultiOutputRegressor(model)
	model.fit(train_inputs, train_outputs)
	y_pred=model.predict(test_inputs)
	test_mse = mean_squared_error(test_outputs, y_pred) #MSE
	test_mae = mean_absolute_error(test_outputs, y_pred) #MAE
	
	mejor_mse = test_mse
	mae = test_mae

	f.write("MSE & MAE Final con para la regresion lineal: \t%f %f\n" % (test_mse, test_mae))
	f.close()

	#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
	guardar_modelo_y_scaler(model,scaler,"RegresionLineal",semana)
	
	print("******************")
	print("La regresion lineal no tiene parametros que buscar, siendo su MSE: "+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



def entrenarRandomForest(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosRandomForest", "w"); ##archivo donde se almacenan los resultados de los parametros

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
					model = ensemble.RandomForestRegressor(min_weight_fraction_leaf=0.2, max_samples= 0.5,n_estimators=n_estimators, max_features=max_features, min_samples_split=min_samples_split, min_samples_leaf=min_samples_leaf, bootstrap=True, n_jobs=4)
					if semana:
						model = MultiOutputRegressor(model)
					model.fit(train_inputs, train_outputs)
					y_pred=model.predict(test_inputs)
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

						#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
						guardar_modelo_y_scaler(model,scaler,"RandomForest",semana)

	f.write("Los mejores parametros para el random forest es n_estimators->"+ str(n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el random forest es n_estimators->"+ str(n_estimators) + "max_features->"+ str(mejor_max_features) + ", min_samples_split->"+ str(mejor_min_samples_split) +", min_samples_leaf->"+ str(mejor_min_samples_leaf) + " con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



def entrenarSDGRegressor(semana, scaler, train_inputs, train_outputs, test_inputs, test_outputs):

	f=open("ResultadosBusquedaParametrosSGCRegressor", "w"); ##archivo donde se almacenan los resultados de los parametros

	#inicializamos parametros que usamos en el entrenamiento
	v_validation = {0.1, 0.2, 0.3, 0.4}
	v_early_stopping = {True, False}
	v_shuffle = {True, False}
	#v_l1_ratio = {0, 0.1, 0.2, 0.3 ,0.4 ,0.5, 0.6 ,0.7 ,0.8 ,0.9 ,1}
	mejor_mse = 99999
	mae = 999999
	mejor_validation = 999999
	mejor_early_stopping = None
	mejor_shuffle = None
	#mejor_l1_ratio = 999999
	mejor_alpha = 999999

	#entrenamos probando distintas combinaciones
	for early_stopping in v_early_stopping:
		for validation in v_validation:
			for shuffle in v_shuffle:
				#for l1_ratio in v_l1_ratio:
					for alpha in range(1,21,1):
						print("early_stopping=%r, validation=%f, shuffle=%r, alpha=%f \n" % (early_stopping, validation, shuffle, alpha/10000))

						#Entrenamos el modelo
						model = linear_model.SGDRegressor(early_stopping=early_stopping, validation_fraction=validation, shuffle=shuffle,alpha=alpha/10000)
						if semana:
							model = MultiOutputRegressor(model)
						model.fit(train_inputs, train_outputs)
						y_pred=model.predict(test_inputs)
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

							#se almacena el modelo y el scaler para poder eliminar el preprocesamiento cuando sea necesario o aplicarlo
							guardar_modelo_y_scaler(model,scaler,"SGDRegressor",semana)
							

	f.write("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	f.close()

	print("******************")
	print("Los mejores parametros para el SGDRegressor es early_stopping->"+str(mejor_early_stopping)+ ", validation->"+ str(mejor_validation)+ ", shuffle->"+ str(mejor_shuffle)+ ", alpha->"+ str(mejor_alpha)+" con un MSE de:"+ str(mejor_mse)+" y un MAE de "+str(mae))
	print("******************")

	return mejor_mse



#funcion para leer los datos del fichero
def read_data(train_file, semana, outputs):

	try:
		Inputs = None
		Outputs = None

		#guardamos los datos en un dataset de tipo string
		Dataset = pd.read_csv('DatasetGeneracion.csv')

		#cambiamos la columna de bool a int
		Dataset["Is Daylight"] = Dataset["Is Daylight"].astype(int)

		if semana == True:
			Dataset = formatear_semana(Dataset)#a la hora de leer del fichero si es del tipo semana es necesario cambiar el formato

		Dataset_sin_procesar = Dataset.copy()

		Dataset = preprocessing(Dataset, mode='train')


		#dividimos en 70% de datos para entreno y el resto para test
		train, test = train_test_split(Dataset, test_size= 0.10)

		#realizamos un preprocesamiento para estandarizar los datos
		scaler = MinMaxScaler().fit(train)
		train = scaler.transform(train)
		test = scaler.transform(test)

		#finalmente se dividen los datos en inputs y outputs
		train_inputs=train[:, :-outputs]
		train_outputs=train[:, -outputs:]
		test_inputs=test[:, :-outputs]
		test_outputs=test[:, -outputs:]

		#transforma a una unica columna para evitar warnings (solo en el caso de una salida)
		if semana == False:
			train_outputs = train_outputs.ravel() 
			test_outputs = test_outputs.ravel()

		return Dataset_sin_procesar, train_inputs, train_outputs, test_inputs, test_outputs, scaler

	except:
		Dataset_cp = None
		train_inputs = None
		train_outputs = None
		test_inputs = None
		test_outputs = None
		scaler = None

		return Dataset_cp, train_inputs, train_outputs, test_inputs, test_outputs, scaler



def crear_fichero_entrenamiento(base_de_datos):

	try:
		#primero se intenta actualizar la base de datos
		actualizar = requests.get("http://bdconapi:8001/actualizar")
		#se comprueba si ha fallado
		if not actualizar.status_code == 200:
			raise Exception("Incorrect reply from Ree API. Status code: {}. Text: {}".format(actualizar.status_code, actualizar.text))
			return {"Error al actualizar"}

		# se crea la conexion a la base de datos para extraer los datos a un fichero
		conn = psycopg2.connect(database=base_de_datos, user="postgres", password="password", host="timescaledb", port="5432")
		cur = conn.cursor()#se crea cursor

		#se realiza la peticion de sql para extraer los datos de la bd y exportarlos a fichero csv
		#se podría traer toda la base de datos a un ficher si no se usa where
		sql = "COPY (SELECT * FROM precio) TO STDOUT WITH CSV DELIMITER ','" 

		with open("/usr/src/app/DatasetGeneracion.csv", "w") as file: 
			cur.copy_expert(sql, file)

		conn.close()#cerramos conexion con la base de datos

		return 200
	except:
		return 300



def guardar_modelo_y_scaler(model, scaler, name, semana):

	if semana:
		#guardamos el modelo y el scaler
		with open(name+'semanal'+'.pickle', 'wb') as fw:
			pickle.dump(model, fw)
		with open(name+'semanal'+'scaler.pickle', 'wb') as fw:
			pickle.dump(scaler, fw)
	else:
		#guardamos el modelo y el scaler
		with open(name+'.pickle', 'wb') as fw:
			pickle.dump(model, fw)
		with open(name+'scaler.pickle', 'wb') as fw:
			pickle.dump(scaler, fw)



def preprocessing(Dataset, outputs=None, mode='train'):#preprocesamiento de los datos donde se eliminan patrones que no nos sirver y se realiza codificacion de parametros categoricos

	if  mode == 'predict':
		Dataset = pd.concat((Dataset, outputs))

	#eliminamos filas vacias
	Dataset=Dataset.dropna()

	#no es necesario por ahora
	ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(sparse=False), list(range(5)))], remainder= 'passthrough')#realización del encoder 
	Dataset = ct.fit_transform(Dataset)
		
	return Dataset



def get_semana(base_de_datos):#funcion que extrae de la base de datos la ultima semana
	#try:
		fecha = datetime.now()
		fecha_inicial = fecha - timedelta(days=8)
		#conexion a la base de datos
		conn = psycopg2.connect(database=base_de_datos, user="postgres", password="password", host="timescaledb", port="5432")
		cur = conn.cursor()#se crea cursor

		#se realiza la peticion de sql para extraer los datos de la bd y exportarlos a fichero csv
		#se podría traer toda la base de datos a un ficher si no se usa where
		sql = "COPY (SELECT * FROM precio where fecha>'"+fecha_inicial.strftime("%Y-%m-%d 23:50")+"') TO STDOUT WITH CSV DELIMITER ','" 
		with open("/usr/src/app/semana.csv", "w") as file: 
			cur.copy_expert(sql, file)

		conn.close()#cerramos conexion con la base de datos

		#realizamos una lectura especial para este caso, en vez de usar read_data
		Dataset = np.genfromtxt('semana.csv', delimiter=',', usecols=range(1,19), dtype='str')

		#cambiamos lass variables 't' o 'f' por '1' o '0' respectivamente
		i = 0
		for x in Dataset[:,5]:
			if x == 't':
				x = '1'
			else:
				x = '0'
			Dataset[i,5] = x
			i+=1

		#pasamos el tipo a float
		Dataset = Dataset.astype(float)

		return Dataset


def formatear_semana(Dataset, mode='train'):#esta funcion formatea los datos para que tengan estructura semanal
	if mode == 'train':
		rows1 = np.where((Dataset[:,1] == 10.0) & (Dataset[:,2] == 31.0) & (Dataset[:,3] == 2))
		rows_change = (Dataset[rows1])[0]
		rows_change[2] = 27
		rows_change[0] = 2022
		rows_change[1] = 3
		Dataset = np.delete(Dataset, np.asarray(rows1[0])[0], axis = 0)
		rows1 = np.where((Dataset[:,1] == 3.0) & (Dataset[:,2] == 27.0) & (Dataset[:,3] == 1))
		Dataset = np.insert(Dataset, rows1[0]+1, np.asarray(rows_change), axis = 0)

	#toda la info menos la hora
	Dataset_dias = np.zeros(shape=(int(len(Dataset)/23), len(Dataset[0])-4))
	#24horas actules, 24 horas semana siguiente
	precios = np.zeros(shape=(int(len(Dataset)/23), 48))

	precio = np.zeros(shape=(48))

	dia_anterior = 0
	j = 0
	demanda = 0

	for i in range(0,len(Dataset),1):
		if Dataset[i,2] != dia_anterior:
			demanda /=24
			Dataset_dias[j] = Dataset[i-2,:-4]
			Dataset_dias[j, 13] = demanda
			demanda = 0
			j+=1
		dia_anterior = Dataset[i,2]
		demanda += Dataset[i,13]
		
	if mode == 'train':
		k = 2
		j = 0
		precio[0] = Dataset[0,-4]
		precio[24] = Dataset[(24*7),-2]
		precio[1] = Dataset[0,-3]
		precio[25] = Dataset[(24*7)+1,-2]
		for i in range(0,len(Dataset)-(24*7),1):
			precio[k] = Dataset[i,-2]
			precio[k+24] = Dataset[i+(24*7),-2]
			k=(k+1)%24
			if k == 0:
				precios[j] = precio
				j+=1
	elif mode == 'predict':
		k = 0
		j = 0
		for i in range(0,len(Dataset),1):
			precio[k] = Dataset[i,-2]
			k=(k+1)%24
			if k == 0:
				precios[j] = precio
				j+=1

	#Dataset_dias = np.delete(Dataset_dias,obj=3,axis=1)
	Dataset_dias = Dataset_dias[~np.all(Dataset_dias == 0, axis=1)]
	precios = precios[~np.all(precios == 0, axis=1)]
	Dataset_dias = Dataset_dias[:len(precios),:]

	Dataset = np.column_stack((Dataset_dias,precios))

	if mode == 'predict':
		Dataset = np.append(Dataset, [Dataset[0,:]], axis=0)
		Dataset = np.delete(Dataset,obj=0,axis=0)

	return Dataset
"""
#**********************************************************************************************************************************************
#*********************************************************FIN_FUNCIONES_DE_APOYO***************************************************************
#**********************************************************************************************************************************************