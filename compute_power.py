#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  1 08:40:15 2025

@author: invitado

FUNCION PARA CALCULAR LA POTENCIA Y PODER USARLA EN EL ENTORNO EN CADA TIMESTEP

"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def compute_power(Tvec,vel_vec,rho,data):

    """
    T:Magnitud del empuje
    Vx: componente en x de la velocidad
    Vz: componente en Z de la velocidad
    rho: densidad
    A: area total de los rotores
    omega: velocidad angular de los rotores
    R: radio de los rotores
    """
    R=data["R"] #Radio de los rotores
    CD0=data["CD0"] #CD0 del perfil de las palas
    solidity=data["solidity"] #Solidez
    masa=data["masa"]
    peso=masa*9.81
    kappa=data["kappa"] #Factor de correcion k (para el CPi)
    K=data["K"] #Factor de corrección K (para el CP0)
    f=data["f"] # Flat area equivalence
    V_tip=data["V_tip"] #Velocidad de punta de pala
    N=data["N"] #Número de rotores

    A=np.pi*R**2 # Area del rotor


    info_vi=vi(rho,A,Tvec,vel_vec,V_tip, N)

    inflow_ratio=info_vi["inflow_ratio"]
    Ct=info_vi["Ct"]
    advance_ratio=info_vi["advance_ratio"]



    #INDUCED POWER

    CPi=kappa*Ct*inflow_ratio

    #PARASITIC POWER
    CPp = 0.5 * (f/A) * advance_ratio**3

    #PROFILE POWER
    CP0=solidity*CD0*(1+K*advance_ratio**2)/8


    # Usando la misma n_thrust de vi()
    Vrel = -vel_vec          # viento relativo


    Thrust_magn = np.linalg.norm(Tvec)
    
    if Thrust_magn < 1e-5:
        return (
            {
                "P_total": 0.0,
                "Pi": 0.0,
                "Pp": 0.0,
                "P0": 0.0,
                "Pc": 0.0
            },
            {
                "Vi": 0.0,
                "inflow_ratio": 0.0,
                "Tmag": 0.0,
                "Ct": 0.0,
                "advance_ratio": 0.0,
                "alpha": 0.0
            }
        )
    

    Vel_normal = np.dot(Vrel, -Tvec/Thrust_magn)     # componente a lo largo del eje del rotor
    inflow_ratio_climb = Vel_normal / V_tip
    Cw=peso/(rho*A*V_tip**2)
    CPc = inflow_ratio_climb * Cw


    Pi=N*CPi*rho*A*V_tip**3
    Pp=CPp*rho*A*V_tip**3
    P0=N*CP0*rho*A*V_tip**3
    Pc=CPc*rho*A*V_tip**3

    CP=CPi+CP0+CPp

    Power=Pi+Pp+P0

    Potencias={"P_total":Power,
               "Pi":Pi,
               "Pp":Pp,
               "P0":P0,
               "Pc":Pc}


    return Potencias,info_vi

def vi(rho, Area, Tvec, vel_vec, V_tip, N,
       tol=1e-6,max_iter=50):

    
    Vrel = -vel_vec   # viento relativo que ve el rotor
    Vmag = np.linalg.norm(Vrel)

    Tmag = np.linalg.norm(Tvec)
    
    if Tmag < 1e-5:
        return {
            "Vi": 0.0,
            "inflow_ratio": 0.0,
            "Tmag": 0.0,
            "Ct": 0.0,
            "advance_ratio": 0.0,
            "alpha": 0.0
        }
    

    # Normal del rotor (dirección del empuje, normalizada)
    n_thrust = Tvec / Tmag
    n_inflow = -n_thrust   # flujo inducido va en dirección opuesta al empuje

    Vn = np.dot(Vrel, n_inflow)      # escalar: componente sobre la normal
    Vt_vec = Vrel - Vn * n_inflow    # vector en el plano del rotor
    Vt = np.linalg.norm(Vt_vec)      # magnitud tangencial
    alpha = np.arctan2(Vn, Vt)      # angulo entre la ve


    advance_ratio = abs(Vt) / V_tip   # μ



    Ct = (Tmag/N) / (rho * Area * V_tip**2)

    mu = max(advance_ratio, 0.0)


    def f(lmbda):
        g = np.sqrt(mu**2 + lmbda**2)
        return lmbda - mu*np.tan(alpha) - Ct/(2.0*g)

    def df(lmbda):
        g = np.sqrt(mu**2 + lmbda**2)
        return 1.0 + (Ct * lmbda) / (2.0 * g**3)

    # Valor inicial para λ
    if abs(mu) < 1e-3:
        lambda_i = np.sqrt(max(Ct, 0.0)/2.0)
    else:
        lambda_i = Ct / (2.0*mu)

     # Iteración de Newton
    for _ in range(max_iter):
      f_val = f(lambda_i)
      if abs(f_val) < tol:
          break
      df_val = df(lambda_i)
      lambda_i = lambda_i - f_val / df_val

    inflow_ratio = lambda_i

    # Devolvemos vi (no solo λ)
    vi_value = inflow_ratio * V_tip

    info={
        "Vi":vi_value,
        "inflow_ratio":inflow_ratio,
        "Tmag":Tmag,
        "Ct":Ct,
        "advance_ratio":mu,
        "alpha":alpha
    }

    return info