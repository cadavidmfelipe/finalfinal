#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 18:23:14 2026

@author: camiyteo
"""


import os
from dataclasses import dataclass, field
from typing import Dict, Callable, Any, List
import numpy as np
import matplotlib.pyplot as plt  
import csv

@dataclass
class EvalContext:
    """Guarda todos los arreglos disponibles del episodio/evaluación."""
    data: Dict[str, Any] = field(default_factory=dict)

    def require(self, keys: List[str]) -> Dict[str, Any]:
        """Devuelve los arrays requeridos o lanza un error claro si faltan."""
        missing = [k for k in keys if k not in self.data]
        if missing:
            raise KeyError(f"Faltan datos para la(s) gráfica(s): {missing}")
        return {k: self.data[k] for k in keys}

class EvalPlots:
    def __init__(self, **arrays):
        # Crea el contexto con todos los arrays que tengas disponibles.
        # Ej: t, xs, zs, reward, dist, vxs, vzs, etc.
        self.ctx = EvalContext(data=arrays)

        # Registro: nombre -> función de trazado
        self._registry: Dict[str, Callable[[], None]] = {
            "rew_vs_dist": self._plot_rew_vs_dist,
            "traj_xz": self._plot_traj_xz,
            "speed_vs_time": self._plot_speed_vs_time,
            "boxplot_recompensas":self._boxplot_reward,
            "acciones":self.visualizacion_accciones,
            "energia": self.energia_vs_tiempo,
            "power_vs_speed": self.power_vs_speed
            # agrega más acá...
        }

    # ========= API principal =========
    def graficar(self,save_path=None, **which_flags: bool):
        """
        Ejemplo: graficar(rew_vs_dist=True, traj_xz=False, speed_vs_time=True)
        Solo se ejecutan las que estén en True.
        """
        to_draw = [name for name, on in which_flags.items() if on]
        if not to_draw:
            raise ValueError("No activaste ninguna gráfica (pasa flags True).")

        # Dibuja una figura por gráfica (simple y modular)
        for name in to_draw:
            if name not in self._registry:
                raise ValueError(f"Gráfica desconocida: '{name}'. Opciones: {list(self._registry)}")
            self._registry[name](save_path=save_path)  # Llama la función de ploteo registrada

        plt.show()

    # ========= Gráficas concretas =========

    def _plot_rew_vs_dist(self,save_path=None): 
        need = self.ctx.require(["xs", "zs","distancias","recompensas","tiempo","xy_objetivo","distances","altitudes"]) 
        rewards_ep = (need["recompensas"])
        distancias_episodio = np.asarray(need["distancias"], dtype=float) 
        xy_objetivo = need["xy_objetivo"] 
        xs_ep = np.asarray(need["xs"], dtype=float) 
        zs_ep = np.asarray(need["zs"], dtype=float) 
        t = np.asarray(need["tiempo"], dtype=float) 
        distances=np.asarray(need["distances"],dtype=float)
        altitudes=np.asarray(need["altitudes"], dtype=float)
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), constrained_layout=True) 
        axes[0].plot(xs_ep[:], zs_ep[:], linewidth=2) 
        axes[0].plot(distances, altitudes, color='green')
        axes[0].fill_between(distances, altitudes, color='lightgreen', alpha=0.5)
        axes[0].scatter(xy_objetivo[0],xy_objetivo[1], marker="x") 
        axes[1].scatter(0, distancias_episodio[0], marker="x") 
        axes[1].plot(t,distancias_episodio) 
        axes[0].set_xlabel("x") 
        axes[0].set_ylabel("z") 
        axes[0].set_title("Trayectoria (episodio)") 
        axes[0].axis("equal") 
        axes[0].grid(True, alpha=0.3) 
        axL = axes[2] 
        for name, values in rewards_ep.items(): 
            y = np.asarray(values, dtype=float) 
            axL.plot(t, y, label=name) 
        axL.set_xlabel("Tiempo [s]") 
        axL.set_ylabel("Recompensas (adim.)") 
        axL.set_ylim(-10,100) 
        axL.grid(True, alpha=0.3)

        lines_left, labels_left = axL.get_legend_handles_labels() 
        #lines_right, labels_right = axR.get_legend_handles_labels() 
        axes[2].legend(lines_left,
                       #lines_right, 
                       labels_left,# + labels_right 
                       ncol=2, fontsize=9, loc="upper right") 
        self._guardar_fig(save_path, name="reward_vs_distance")
            
    def _plot_traj_xz(self):
        need = self.ctx.require(["xs", "zs"])
        xs = np.asarray(need["xs"], dtype=float)
        zs = np.asarray(need["zs"], dtype=float)

        plt.figure()
        plt.plot(xs, zs, linewidth=2)
        plt.xlabel("x")
        plt.ylabel("z")
        plt.title("Trayectoria x-z")
        # Punto objetivo opcional si está disponible
        if "xg" in self.ctx.data and "zg" in self.ctx.data:
            plt.scatter([self.ctx.data["xg"]], [self.ctx.data["zg"]], marker="x")

    def _plot_speed_vs_time(self,save_path=None):
        need = self.ctx.require(["tiempo"])
        t = np.asarray(need["tiempo"], dtype=float)

        # Si ya tienes speed, úsalo; si no, intenta derivarlo de vxs, vzs
        if "speed" in self.ctx.data:
            speed = np.asarray(self.ctx.data["speed"], dtype=float)
        else:
            need2 = self.ctx.require(["vxs", "vzs"])
            vxs = np.asarray(need2["vxs"], dtype=float)
            vzs = np.asarray(need2["vzs"], dtype=float)
            speed = np.sqrt(vxs**2 + vzs**2)

        plt.figure()
        plt.plot(t, speed)
        plt.xlabel("tiempo (s)")
        plt.ylabel("velocidad (m/s)")
        plt.title("Velocidad vs Tiempo")
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)  
        plt.show()
        
        
    def _boxplot_reward(self,save_path=None):
        need = self.ctx.require(["recompensas"]) 
        rewards_ep = (need["recompensas"])
        self._guardar_fig(save_path, name="Velocidad vs tiempo")
        

# lista en el orden que quieres graficar
        keys = rewards_ep.keys()
        
        data = [rewards_ep[k] for k in keys]
        
        plt.figure(figsize=(12,6))
        plt.boxplot(data, showfliers=False)  # showfliers=False para que no te meta outliers exagerados visualmente
        plt.xticks(range(1, len(keys)+1), keys, rotation=45)
        plt.ylabel("Reward value")
        plt.title("Distribución de cada término de recompensa")
        plt.tight_layout()
        plt.show()
        
        self._guardar_fig(save_path, name="boxplot_recompensas")
        
    def visualizacion_accciones(self, save_path):
        need = self.ctx.require(["acciones_norm","magnitud_real","angulo_real"]) 
        actions=(need["acciones_norm"])
        magnitud=(need["magnitud_real"])
        angulo=(need["angulo_real"])
        angulos = actions[:,0]
        magnitudes = actions[:,1]
        
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True) 
        
        axes[0,0].plot(angulos)
        axes[0,1].title("Ángulo de acción (Normalizada)")
        plt.set_xlabel("timestep")
        plt.set_ylabel("ángulo")
        
        axes[0,1].plot(magnitudes)
        axes[0,1].title("Magnitud de acción (Normalizada")
        plt.set_xlabel("timestep")
        plt.set_ylabel("Magnitud")
        
        axes[1,0].plot(angulo)
        axes[1,0].title("Ángulo de acción")
        plt.set_xlabel("timestep")
        plt.set_ylabel("ángulo")
        
        axes[1,1].plot(angulo)
        axes[1,1].title("Magnitud de acción")
        plt.set_xlabel("timestep")
        plt.set_ylabel("Magnitud")
        
        plt.tight_layout()
        plt.show()
        
    def energia_vs_tiempo(self, save_path):
        need = self.ctx.require(["potencia", "acumulada","cps"]) 
        potencias=(need["potencia"])
        energía_acumulada=(need["acumulada"])
        Cps=(need["cps"])
        
        plt.figure(figsize=(12,4))
        plt.subplot(1,3,1)
        plt.plot(potencias)
        plt.title("Potencia vs timestep")
        plt.xlabel("timestep")
        plt.ylabel("Potencia (hp)")
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)  # << aquí

        
        plt.subplot(1,3,2)
        plt.plot(Cps)
        plt.title("Coeficiente de potencia")
        plt.xlabel("timestep")
        plt.ylabel("Cp (adim)")
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)  # << aquí

        
        plt.subplot(1,3,3)
        plt.plot(energía_acumulada)
        plt.title("Energía acumulada")
        plt.xlabel("timestep")
        plt.ylabel("Energía (J)")
        
        plt.show()
                
    def _guardar_fig(self, save_path, name):
        if save_path is not None:
            final_path = os.path.join(save_path, f"{name}.png")
            plt.savefig(final_path, dpi=300)
            print(f"Imagen guardada en {final_path}")
        else:
            print("No se guardó la figura")
            
    def power_vs_speed(self,save_path=None):
        need = self.ctx.require(["speed", "potencia"]) 
        potencias=(need["potencia"])
        velocidades=(need["speed"])
        
        plt.figure()
        plt.plot(velocidades,potencias)
        plt.xlabel("velocidad (m/s)")
        plt.ylabel("potencia (hp)")
        plt.title("Potencia vs Velocidad")
        plt.ticklabel_format(style='plain', axis='y', useOffset=False)  
        plt.show()



def guardar_trayectoria_csv(save_path, xs, zs, vxs, vzs,
                            distancias, rewards, acciones,energia, dt):
    
    
    """
    Guarda en un CSV la información del episodio evaluado:
    posiciones, velocidades, distancia al objetivo, recompensas, energía y acción.
    """
    filename = save_path / "trayectoria.csv"

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        # Cabeceras
        columnas = [
            "timestep", "tiempo (s)",
            "x (m)", "z (m)",
            "vx (m/s)", "vz (m/s)",
            "distancia_objetivo (m)",
            "r_distance", "r_clearence", "r_energy",
            "r_terminal", "total_reward",
            "angulo empuje normalizado", "magnitud empuje normalizado",
            "energía_acumulada (J)"
        ]
        writer.writerow(columnas)

        # Iterar sobre los timesteps
        for i in range(len(xs)):
            writer.writerow([
                i,
                i * dt,
                xs[i], zs[i],
                vxs[i], vzs[i],
                distancias[i],
                rewards["r_distance"][i],
                rewards["r_clearence"][i],
                rewards["r_energy"][i],
                rewards["r_terminal"][i],
                rewards["total_reward"][i],
                acciones[i,0,0],
                acciones[i,0,1],
                energia[i]
            ])

    print(f"✅ Trayectoria guardada en: {filename}")