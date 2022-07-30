--------------------------------------------------------------------
Este fichero contiene información sobre el esquema de la base de datos.
--------------------------------------------------------------------

1. Tabla de datos_luminaria;

Esta tabla contiene información sobre los atributos extraidos para los experimentos.

- Variables:
	
	+ Nombre de luminaria usada en ese experimento
	+ Altura donde se coloca la luminaria en el experimento (cm)
	+ Flujo Luminico de la luminaria colocada (lm)
	+ TCC, Temperatura Correlada del Color (K)
	+ Iluminancia detectada a nivel de suelo (lux)
	+ Espectro producido por la luminaria(valor maximo nm)
	+ Color del suelo usado en el experimento
	+ Indice de reflectancia para el suelo puesto
	+ Iluminancia en el punto justo superior a la luminaria (lux)
	+ Flujo Luminico superior a la luminaria (lm)
	+ FHSI, Flujo Hemisferio Superior instalado, porcentaje de flujo que se desprende 
	por encima de la luminaria dependiendo del flujo total(%)
	
	
	
	    Field			|           Type		| Null		| Key		 | Default 	| Extra
--------------------------------------+------------------------------+--------------+----------------+-------------+-------+
 TipoLuminaria				| varchar(255)		 	| YES		| 		 |		|
 AlturaLuminaria			| float			| YES		|		 |		|
 FlujoLuminicoTotal			| float			| YES		| 		 |		|
 TCC					| float			| YES		|  		 |		|
 IluminanciaAbajo			| float			| YES		| 		 | 		|
 Espectro				| float			| YES		| 		 | 		|
 ColorSuelo				| varchar(255)			| YES		| 		 |		| 
 ReflectanciaSuelo			| float			| YES		|		 | 		|
 IluminanciaSuperior			| float			| YES		|		 | 		|
 FlujoSuperior				| float			| YES		| 		 | 		|
 FHSI					| float			| YES		|		 | 		|

