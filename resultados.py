#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 15 08:06:06 2026

@author: camiyteo
"""

import pandas as pd
import matplotlib.pyplot as plt

##ENTRENAMIENTO

numero_run=21
directorio_entrenamiento= "runs/RL_agente" + str(numero_run) 
ruta_csv=directorio_entrenamiento + "/monitor/train_env_3.csv.monitor.csv"

entrenamiento_df=pd.read_csv(ruta_csv,skiprows=1)

plt.figure()
plt.plot(entrenamiento_df["r"])
plt.xlabel("Episodio")
plt.ylabel("Retorno")
plt.title("Retorno por Episodio")
plt.ticklabel_format(style='plain', axis='y', useOffset=False)  
plt.show()


#TRAYECTORIA DE EVALUACIÓN

numero_eval=24
directorio_evaluaciones= "evaluaciones/EVALUACION_" + str(numero_eval)
ruta_trayectoria=directorio_evaluaciones + "/trayectoria.csv"

trayectoria_df=pd.read_csv(ruta_trayectoria)


