#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Sep 25 10:09:21 2025

@author: invitado
"""

import os, re, json, datetime, sys, platform, subprocess
from pathlib import Path
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor  import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnNoModelImprovement, CheckpointCallback, ProgressBarCallback
from evtol_env import EVTOLFlightEnv  # tu entorno
from Callbacks import CurriculumCallback, TrainingStatsCallback,EntropyLogger

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


SEED = 42
N_ENVS = 4  # usa 1 si no quieres paralelizar
TOTAL_STEPS = 2_000_00

RUNS_DIR = Path("./runs")
PREFIX = "RL_AGENTE"    # => RL_AGENTE01, RL_AGENTE02, ...

info_keywords_train = (
    "end_reason", "success",
    "distancia_final", "tiempo (s)",
    #"action_sat_pct",
    "r_distance_total", "r_clearance_total", "r_energy_total",
    "r_total", "r_terminal_total",
    #"distancia inicial","coordenadas meta","tolerancia"
)

info_keywords_eval = (
    "end_reason", "success",
    "distancia_final", "energy_used", "tiempo (s)",
    "longitud del vuelo", "lista con velocidades", "altitudes",
    #"action_sat_pct", "dvdt_max",
    "r_distance_total", "r_clearance_total", "r_energy_total",
    "r_total", "r_terminal_total",
    #"distancia inicial","coordenadas meta","tolerancia"
    #"seed", "deterministic", "model_path",
    #"vecnorm_path", "vecnorm_training", "vecnorm_norm_reward"
)


RUNS_DIR = Path("./runs")
PREFIX = "RL_AGENTE"    # => RL_AGENTE01, RL_AGENTE02, ...

def next_run_dir():
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    # Busca carpetas existentes con el patrón PREFIXNN
    nums = []
    for p in RUNS_DIR.iterdir():
        if p.is_dir():
            m = re.fullmatch(rf"{PREFIX}(\d+)", p.name)
            if m:
                nums.append(int(m.group(1)))
    n = (max(nums) + 1) if nums else 1
    name = f"{PREFIX}{n:02d}"
    run_dir = RUNS_DIR / name
    run_dir.mkdir()
    # subcarpetas
    (run_dir / "tb").mkdir()
    (run_dir / "eval").mkdir()
    (run_dir / "checkpoints").mkdir()
    (run_dir / "best").mkdir()
    (run_dir / "last").mkdir()
    (run_dir / "monitor").mkdir()
    (run_dir / "replay_buffer").mkdir()
    return run_dir

def sys_versions():
    # versiones de libs clave; si alguna falta, ignora
    import importlib
    pkgs = ["numpy","gymnasium","stable_baselines3"]
    out = {}
    for p in pkgs:
        try:
            m = importlib.import_module(p)
            v = getattr(m, "__version__", "unknown")
        except Exception:
            v = "not_installed"
        out[p] = v
    out["python"] = sys.version.replace("\n"," ")
    out["platform"] = platform.platform()
    # commit git opcional
    try:
        commit = subprocess.check_output(["git","rev-parse","--short","HEAD"]).decode().strip()
        out["git_commit"] = commit
    except Exception:
        pass
    return out



if __name__ == "__main__":
    np.random.seed(SEED)

    run_dir = next_run_dir()
    LOGDIR = str(run_dir / "tb")
    BEST_DIR = run_dir / "best"
    LAST_DIR = run_dir / "last"
    CKPT_DIR = run_dir / "checkpoints"
    MONITOR_DIR = run_dir / "monitor"
    EVAL_DIR = run_dir / "eval"
    VECNORM_PATH = str(run_dir / "vecnorm.pkl")
    REPLAY_PATH = str(run_dir / "replay_buffer" / "sac_replay.pkl")
    STATS_PATH = run_dir / "stats.csv"

    # === Envs de training ===
    def make_train_env(rank: int = 0):
        def _thunk():
            env = EVTOLFlightEnv(data)
            # Escribe CSV por worker
            monitor_file = str(MONITOR_DIR / f"train_env_{rank}.csv")
            env = Monitor(env, filename=monitor_file, info_keywords=info_keywords_train)
            return env
        return _thunk

    if N_ENVS > 1:
        train_env = SubprocVecEnv([make_train_env(i) for i in range(N_ENVS)], start_method="spawn")
    else:
        train_env = DummyVecEnv([make_train_env(0)])

    # Normalización
    train_env = VecNormalize(train_env, norm_obs=True, norm_reward=True, clip_obs=10.0, gamma=0.995)

    # === Env de evaluación ===
    def make_eval_env():
        env = EVTOLFlightEnv(data)
        # CSV de evaluación agregado aparte por claridad
        eval_monitor_file = str(MONITOR_DIR / f"eval_env.csv")
        env = Monitor(env, filename=eval_monitor_file, info_keywords=info_keywords_eval)
        return env

    eval_env_raw = DummyVecEnv([lambda: make_eval_env()])
    eval_env = VecNormalize(eval_env_raw, training=False, norm_obs=True, norm_reward=False, clip_obs=10.0, gamma=0.995)
    
    eval_env.obs_rms = train_env.obs_rms  # sincroniza stats


    # === Callbacks ===
    stop_cb = StopTrainingOnNoModelImprovement(
        max_no_improvement_evals=10,
        min_evals=10,
        verbose=1
    )
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=str(BEST_DIR),
        log_path=str(EVAL_DIR),
        eval_freq=20_000 // max(N_ENVS, 1),
        n_eval_episodes=1,
        deterministic=True,
        #callback_after_eval=stop_cb
    )
    ckpt_cb = CheckpointCallback(
        save_freq=100_000 // max(N_ENVS, 1),
        save_path=str(CKPT_DIR),
        name_prefix="sac_evtol"
    )
    
    cb_stats = TrainingStatsCallback(
    csv_path=STATS_PATH,
    T_MAX=float(2.5*2000*9.81),             # <- pon tu límite real de empuje (escala de acción)
    THETA_MAX_DEG=90.0,    # <- pon el límite angular real
    EPS_SAT=0.02           # 2% del rango como “casi saturado”
)
    
    
    curriculum_cb = CurriculumCallback(
    check_freq=50,  # Revisar cada 100 episodios
    verbose=1
)
    
    entropy_cb = EntropyLogger(log_dir=run_dir, check_freq=5000)

    
    #pbar_cb = ProgressBarCallback()

    # === Modelo ===
    # OPCIÓN A) continuar desde un modelo previo (cambios leves en recompensa)
    #OLD_model = SAC.load("runs/RL_AGENTE59/best/best_model.zip", env=train_env, print_system_info=True)
    #params=OLD_model.get_parameters()
    
    

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-5,
        buffer_size=1_000_000,
        batch_size=256,
        tau=0.005,
        gamma=0.995,
        train_freq=1,
        gradient_steps=1,
        learning_starts=10_000,
        policy_kwargs=dict(net_arch=[256,256]),
        verbose=1,
        tensorboard_log=LOGDIR,
        seed=SEED,
        device="auto",


    )

    #model.set_parameters(params)
    
    

    # Guarda un manifiesto del run
    manifest = {
        "run_name": run_dir.name,
        "datetime": datetime.datetime.now().isoformat(),
        "seed": SEED,
        "n_envs": N_ENVS,
        "total_steps": TOTAL_STEPS,
        "algo": "SAC",
       "policy": "MlpPolicy",
        "hyperparams": {
            "lr": 3e-5, "buffer_size": 1_000_000, "batch_size": 256,
            "tau": 0.005, "gamma": 0.995, "learning_starts": 10_000,
            "net_arch": [256,256]
        },
        "Espacio de observacion": str(train_env.observation_space),
        "versions": sys_versions(),
        "Objetivo": str(eval_env.venv.envs[0].env.unwrapped.destino),
        "Observacion": "recompensa 3 terminos",
        #"Pesos": str(eval_env.venv.envs[0].env.unwrapped.pesos),
        "Entorno":str(EVTOLFlightEnv.__module__)

    }
    with open(run_dir / "config.json", "w") as f:
        json.dump(manifest, f, indent=2)

    try:
        model.learn(total_timesteps=TOTAL_STEPS, callback=[eval_cb, ckpt_cb, #pbar_cb,
                                                           cb_stats, #curriculum_cb
                                                       ], log_interval=10)
    except Exception as e:
        print("\nERROR DURANTE ENTRENAMIENTO:", e)
        # Cierra la barra manualmente
        #pbar_cb.on_training_end()
        raise e   # para que sepas qué error fue

    # === Guardados finales ===
    model.save(str(LAST_DIR / "model.zip"))       # último
    train_env.save(VECNORM_PATH)                  # stats normalización

    # (Opcional) guardar replay buffer para “reanudar” entrenamiento
    try:
        model.save_replay_buffer(REPLAY_PATH)
    except Exception as e:
        print(f"[WARN] No se pudo guardar replay buffer: {e}")

    # Cierre
    train_env.close()
    eval_env.close()
