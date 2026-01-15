#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 16:53:33 2026

@author: camiyteo
"""

import os, re, json
import gymnasium as gym
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, ProgressBarCallback
from evtol_env import EVTOLFlightEnv  # tu entorno
import matplotlib.pyplot as plt
from pathlib import Path
from graficar import EvalPlots, guardar_trayectoria_csv

data={
      "R":4.0,
      "CD0": 0.008,
      "solidity": 0.055,
      "masa":2940,
      "f":1.2,
      "V_tip":167.64,
      "N":4,
      "p_disp_r":125278,
      "kappa":1,
      "K":4.5,
      "E_max":1325e6
      
      }


EVALS_DIR = Path("./evaluaciones")
SEED = 52  # fija el inicio para que xy_inicial sea reproducible
PREFIX = "EVALUACION_" 

info_keywords_eval = (
    "end_reason", "success",
    "distancia_final", "energía J", "tiempo (s)",
    "velocidad (m/s)", "longitud del vuelo", #"variacion de velocidad",
    #"action_sat_pct", "dvdt_max",
    "r_distance", "r_direction", "r_proximity",
    #"r_vel_approach", "r_vel_shaping", "r_smooth", "r_vel_bonus",
    "r_terminal", "total_reward"
    #"seed", "deterministic", "model_path",
    #"vecnorm_path", "vecnorm_training", "vecnorm_norm_reward"
)
"""

def nombrar(xy_obj, xy_ini):
    # nombre corto, sin caracteres raros; redondea para evitar ruido
    return (f"obj_{xy_obj[0]:.2f}_{xy_obj[1]:.2f}"
            f"__ini_{xy_ini[0]:.2f}_{xy_ini[1]:.2f}")

def carpeta_evaluaciones(base_dir: Path, xy_obj, xy_ini):
    base_dir.mkdir(parents=True, exist_ok=True)
    name = nombrar(xy_obj, xy_ini)
    eval_dir = base_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir
"""


def carpeta_evaluaciones(base_dir: Path):
    base_dir.mkdir(parents=True, exist_ok=True)
    nums = []
    for p in base_dir.iterdir():
        if p.is_dir():
            m = re.fullmatch(rf"{PREFIX}(\d+)", p.name)
            if m:
                nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    name = f"{PREFIX}{n:02d}"
    eval_dir = base_dir / name
    eval_dir.mkdir(parents=True, exist_ok=True)
    return eval_dir

base_dir=carpeta_evaluaciones(EVALS_DIR)

def make_eval_env():
    env = EVTOLFlightEnv(data)
    obs,_=env.reset(seed=SEED)
    
    xy_ini = (env.x, env.z)
    xy_obj = tuple(env.objetivo)
    

    monitor_file = base_dir / "monitor.csv"
    

    env = Monitor(env, filename=str(monitor_file)#,
                  #info_keywords=info_keywords_eval)
                  )

    env.eval_metadata = {"xy_obj": xy_obj, "xy_ini": xy_ini, "seed": SEED}

    return env


VECNORM_PATH = "runs/RL_AGENTE22/vecnorm.pkl"
save_path=base_dir



eval_env = DummyVecEnv([make_eval_env])
eval_env = VecNormalize.load(VECNORM_PATH, eval_env)
eval_env.training = False
eval_env.norm_reward = False  # MUY IMPORTANTE

model_path="runs/RL_AGENTE22/best/best_model.zip"
model_path2="runs/RL_AGENTE106/checkpoints/sac_evtol_300000_steps.zip"
model = SAC.load(model_path, env=eval_env)
#model = SAC.load("runs/RL_AGENTE105/last/model.zip", env=eval_env)

m = re.search(r'RL_AGENTE(\d+)', model_path)
model_number = int(m.group(1)) if m else None


n_episodes = 1


ep_lengths = []

ep_rewards= []
successes = 0
final_distances = []
final_energies = []


#POR EPISODIO
distancia_al_objetivo=[]

base_env = eval_env.venv.envs[0].env.unwrapped





for ep in range(n_episodes):
    obs= eval_env.reset()
    eval_env.seed(12)
    
    ep_rew = 0.0
    steps = 0
    xs, zs = [], []
    vxs,vzs = [], []
    distancia_al_objetivo=[]
    empujes=[]
    acciones=[]
    potencias=[]
    rewards = {
        "r_distance": [],
        "r_clearence":[],
        "r_energy":[],
        #"r_direction": [],
        #"r_proximity": [],
        #"r_vel_approach": [],
        #"r_vel_shaping": [],
        #"r_smooth": [],
        #"r_vel_bonus": [],
        "r_terminal": [],
        "total_reward": [],
    }
    
    total_energy = 0.0
    

    while True:
        action, _ = model.predict(obs, deterministic=True)
        #action=np.array([[0.0,0.2]])
        obs, reward, done, info = eval_env.step(action)
        info=info[0]
        ep_rew += float(reward[0])
        steps += 1

        try:
            state=info.get("state")
            dt = base_env.unwrapped.dt
            x = state["x"]
            z = state["z"]
            vx = state["vx"]
            vz = state["vz"]
            potencia=info.get("potencia")
            empuje=info.get("T (acción)")
            d_goal = info.get("distancia_final", None)
            distancia_al_objetivo.append(float(d_goal) if d_goal is not None else np.nan)
            vxs.append(vx)
            vzs.append(vz)
            xs.append(x)
            zs.append(z)
            empujes.append(empuje)
            acciones.append(action)
            potencias.append(potencia)
            for key in rewards:
                val = info.get(key, np.nan)
                rewards[key].append(float(val) if val is not None else np.nan)
            
        except Exception:
            pass

        power = info.get("power", None)
        if power is not None:
            total_energy += float(power) * dt

        if done:
            ep_rewards.append(ep_rew)
            ep_lengths.append(steps)
            successes += int(info.get("success", False))
            print(f"EP{ep} terminó por: {info.get('end_reason')}")
            
            if "distance" in info:
                final_distances.append(float(info["distance"]))
            else:
                try:
                    x, z = float(obs[0]), float(obs[1])
                    xg, zg = base_env.unwrapped.xy_objetivo
                    final_distances.append(float(np.hypot(xg - x, zg - z)))
                except Exception:
                    final_distances.append(np.nan)

            if total_energy > 0:
                final_energies.append(total_energy)

            if ep == 0:
                distancias_episodio=distancia_al_objetivo
                vxs_ep=vxs
                vzs_ep=vzs
                xs_ep=xs
                zs_ep=zs
                rewards_ep=rewards
                steps_ep=steps
                acciones_ep=np.array(acciones)
                potencias_ep=potencias
                xy_objetivo=base_env.unwrapped.objetivo
                print("valores actualizados")
                
            break
        
t = np.arange(steps_ep)*dt
#xy_objetivo=base_env.unwrapped.xy_objetivo
distances=base_env.unwrapped.distances
altitudes=base_env.unwrapped.altitudes

plots = EvalPlots(xs=xs_ep, zs=zs_ep, distancias=distancias_episodio,
                  recompensas=rewards_ep, tiempo=t, xy_objetivo=xy_objetivo,
                  distances=distances,altitudes=altitudes,
                  vxs=vxs_ep, vzs=vzs_ep)

plots.graficar(rew_vs_dist=True,
               boxplot_recompensas=False,
               speed_vs_time=True,
               save_path=save_path)
manifest = {
    "entorno": str(str(EVTOLFlightEnv.__module__)),
    "modelo": str(model_number)
    }

guardar_trayectoria_csv(save_path, xs_ep, zs_ep, vxs_ep, vzs_ep,
                        distancias_episodio, rewards_ep, acciones_ep, potencias_ep, dt)

print("retorno", sum(rewards_ep["total_reward"]))
print("retorno", sum(rewards_ep["total_reward"]))

with open(base_dir / "config.json", "w") as f:
    json.dump(manifest, f, indent=2)

out = {
    "episodes": n_episodes,
    "mean_reward": float(np.nanmean(ep_rewards)) if ep_rewards else np.nan,
    "std_reward": float(np.nanstd(ep_rewards)) if ep_rewards else np.nan,
    "mean_ep_length": float(np.nanmean(ep_lengths)) if ep_lengths else np.nan,
    "success_rate": float(successes) / float(n_episodes) if n_episodes > 0 else np.nan,
    "mean_final_distance": float(np.nanmean(final_distances)) if final_distances else np.nan,
    "mean_energy_J": float(np.nanmean(final_energies)) if final_energies else np.nan,
}