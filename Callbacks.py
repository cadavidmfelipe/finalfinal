#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 08:51:34 2025

@author: invitado

CALLBACKS

"""

from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import pandas as pd
import os
from collections import defaultdict, deque
from pathlib import Path
import csv
import math
import torch as th



class CurriculumCallback(BaseCallback):
    """
    Callback que actualiza el curriculum basado en tasa de éxito
    """
    def __init__(self, check_freq=100, verbose=1):
        super().__init__(verbose)
        self.check_freq = check_freq  # Cada cuántos episodios revisar
        self.episode_successes = []  # Buffer de éxitos
        self.episode_count = 0
        
    def _on_step(self) -> bool:
        # Se ejecuta después de cada step
        # Revisar si el episodio terminó
        if self.locals['dones'][0]:
            self.episode_count += 1
            
            # Obtener info del episodio
            info = self.locals['infos'][0]
            success = info.get('success', False)
            self.episode_successes.append(float(success))
            
            # Cada check_freq episodios, actualizar curriculum
            if self.episode_count % self.check_freq == 0:
                success_rate = np.mean(self.episode_successes[-self.check_freq:])
                self.training_env.env_method("update_curriculum", success_rate)
                vals = self.training_env.env_method("get_curriculum_vals")[0]

            
                
                if self.verbose > 0:
                    print(f"\n{'='*60}")
                    print(f"📚 CURRICULUM UPDATE - Episodio {self.episode_count}")
                    print(f"{'='*60}")
                    print(f"  Tasa de éxito (últimos {self.check_freq}): {success_rate*100:.1f}%")
                    print(f"  EMA de éxito: {vals['success_ema']*100:.1f}%")
                    print(f"  Radio objetivo: {vals['Radio']:.1f}m")
                    print(f"  Tolerancia: {vals['tolerancia']:.1f}m")

                    print(f"{'='*60}\n")
        
        return True
    
    
    
class EntropyLogger(BaseCallback):
    """
    Callback para registrar la entropía media de la política durante el entrenamiento.
    Guarda en un CSV: timestep, entropy_mean, ent_coef
    """
    def __init__(self, log_dir, check_freq=5000):
        super().__init__()
        self.log_dir = log_dir
        self.check_freq = check_freq
        self.data = []

    def _on_step(self):
        if self.n_calls % self.check_freq == 0:
            # Muestrea N observaciones del entorno
            obs = np.array([self.training_env.observation_space.sample() for _ in range(128)])
            with th.no_grad():
                dist = self.model.actor.get_distribution(obs)
                entropy = dist.entropy().cpu().numpy()
            entropy_mean = np.mean(entropy)
            ent_coef = float(self.model.ent_coef.item()) if hasattr(self.model.ent_coef, "item") else self.model.ent_coef
            self.data.append([self.num_timesteps, entropy_mean, ent_coef])
        return True

    def _on_training_end(self):
        df = pd.DataFrame(self.data, columns=["timesteps", "entropy_mean", "ent_coef"])
        os.makedirs(self.log_dir, exist_ok=True)
        df.to_csv(os.path.join(self.log_dir, "entropy_log.csv"), index=False)
        
        
        

class TrainingStatsCallback(BaseCallback):
    def __init__(self, 
                 csv_path="runs/train_stats.csv",
                 window_success=100,
                 T_MAX=1.0,                # <-- AJUSTA a tu escala de acción
                 THETA_MAX_DEG=90.0,       # <-- AJUSTA a tu escala de acción
                 EPS_SAT=0.02,             # 2% del rango se considera saturación
                 verbose=0):
        super().__init__(verbose)
        self.csv_path = Path(csv_path)
        self.window_success = window_success
        self.T_MAX = float(T_MAX)
        self.THETA_MAX_DEG = float(THETA_MAX_DEG)
        self.EPS_SAT = float(EPS_SAT)
        self.ep_buffers = defaultdict(lambda: defaultdict(float))  # por env_idx
        self.ep_counts = defaultdict(int)
        self.prev_T = defaultdict(lambda: None)
        self.prev_theta = defaultdict(lambda: None)
        self.sat_counts = defaultdict(int)
        self.deltaT_sum = defaultdict(float)
        self.deltaTheta_sum = defaultdict(float)
        self.step_counts = defaultdict(int)
        self.success_window = deque(maxlen=window_success)
        self._csv_inited = False

    def _init_csv(self):
        self.csv_path.parent.mkdir(parents=True, exist_ok=True)
        if not self.csv_path.exists():
            with self.csv_path.open("w", newline="") as f:
                w = csv.writer(f)
                w.writerow([
                    "global_step","episode_idx","env_idx",
                    "ep_len","ep_time_s",
                    "return_sum",
                    "r_distance_mean","r_direction_mean","r_proximity_mean","r_terminal_mean",
                    "success","end_reason",
                    "energy_kJ_final","power_max",
                    "sat_rate","dT_mean","dTheta_mean"
                ])
        self._csv_inited = True

    def _on_training_start(self):
        self._init_csv()

    def _maybe_log_scalar(self, tag, value, step):
        if self.logger is not None:
            try:
                self.logger.record(tag, float(value))
            except Exception:
                pass

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        # Hay múltiples envs: recorremos cada uno
        for env_idx, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            # Acumular métricas disponibles en info
            buf = self.ep_buffers[env_idx]
            # Sumas de recompensas parciales para promediar al final
            for k in ("r_distance","r_direction","r_proximity","r_terminal","total_reward"):
                if k in info and info[k] is not None:
                    buf[k] += float(info[k])

            # Energía/potencia
            if "energía kJ" in info:
                buf["energy_kJ_last"] = float(info["energía kJ"])
            if "power" in info:
                buf["power_max"] = max(buf.get("power_max", float("-inf")), float(info["power"]))

            # Acciones (para saturación y suavidad)
            T = info.get("T (acción)")
            theta = info.get("theta (acción)")
            if T is not None:
                T = float(T)
                self.step_counts[env_idx] += 1
                if self.T_MAX > 0:
                    if T >= self.T_MAX * (1.0 - self.EPS_SAT):
                        self.sat_counts[env_idx] += 1
                prevT = self.prev_T[env_idx]
                if prevT is not None:
                    self.deltaT_sum[env_idx] += abs(T - prevT)
                self.prev_T[env_idx] = T
            if theta is not None:
                theta = float(theta)
                prevTh = self.prev_theta[env_idx]
                if prevTh is not None:
                    self.deltaTheta_sum[env_idx] += abs(theta - prevTh)
                self.prev_theta[env_idx] = theta

            # Terminar episodio (Monitor añade "episode" en info) 
            # o si tu env pone flags en "done" con 'success', 'end_reason'
            if "episode" in info:  # vía Monitor
                ep_len = int(info["episode"]["l"])
                ep_ret = float(info["episode"]["r"])
                self.ep_counts[env_idx] += 1
                ep_idx = self.ep_counts[env_idx]

                success = int(info.get("success", 1 if ep_ret > 0 else 0))
                end_reason = info.get("end_reason", "unknown")

                # Promedios de r_* por paso (si se acumularon)
                denom = max(ep_len, 1)
                r_distance_mean  = buf.get("r_distance", 0.0)  / denom
                r_direction_mean = buf.get("r_direction", 0.0) / denom
                r_proximity_mean = buf.get("r_proximity", 0.0) / denom
                r_terminal_mean  = buf.get("r_terminal", 0.0)  / denom

                sat_rate = (self.sat_counts[env_idx] / max(self.step_counts[env_idx],1))
                dT_mean = self.deltaT_sum[env_idx] / max(self.step_counts[env_idx]-1,1)
                dTheta_mean = self.deltaTheta_sum[env_idx] / max(self.step_counts[env_idx]-1,1)

                energy_kJ_final = buf.get("energy_kJ_last", float("nan"))
                power_max = buf.get("power_max", float("nan"))

                # Tiempo de episodio si tu info lo trae
                ep_time_s = float(info.get("tiempo (s)", float("nan")))

                # CSV
                with self.csv_path.open("a", newline="") as f:
                    w = csv.writer(f)
                    w.writerow([
                        self.num_timesteps, ep_idx, env_idx,
                        ep_len, ep_time_s,
                        ep_ret,
                        r_distance_mean, r_direction_mean, r_proximity_mean, r_terminal_mean,
                        success, end_reason,
                        energy_kJ_final, power_max,
                        sat_rate, dT_mean, dTheta_mean
                    ])

                # Logger (para ver en tiempo real/TensorBoard)
                self._maybe_log_scalar("train/episode_return", ep_ret, self.num_timesteps)
                self._maybe_log_scalar("train/episode_len", ep_len, self.num_timesteps)
                self._maybe_log_scalar("train/r_distance_mean", r_distance_mean, self.num_timesteps)
                self._maybe_log_scalar("train/r_direction_mean", r_direction_mean, self.num_timesteps)
                self._maybe_log_scalar("train/r_proximity_mean", r_proximity_mean, self.num_timesteps)
                self._maybe_log_scalar("train/r_terminal_mean", r_terminal_mean, self.num_timesteps)
                self._maybe_log_scalar("train/success", success, self.num_timesteps)
                self._maybe_log_scalar("train/sat_rate", sat_rate, self.num_timesteps)
                self._maybe_log_scalar("train/dT_mean", dT_mean, self.num_timesteps)
                self._maybe_log_scalar("train/dTheta_mean", dTheta_mean, self.num_timesteps)
                if not math.isnan(energy_kJ_final):
                    self._maybe_log_scalar("train/energy_kJ_final", energy_kJ_final, self.num_timesteps)
                if not math.isnan(power_max):
                    self._maybe_log_scalar("train/power_max", power_max, self.num_timesteps)

                # Mantener ventana de éxitos (para tasa móvil fuera de TB si quieres)
                self.success_window.append(success)

                # Reset buffers por env
                self.ep_buffers.pop(env_idx, None)
                self.prev_T[env_idx] = None
                self.prev_theta[env_idx] = None
                self.sat_counts[env_idx] = 0
                self.deltaT_sum[env_idx] = 0.0
                self.deltaTheta_sum[env_idx] = 0.0
                self.step_counts[env_idx] = 0

        return True
