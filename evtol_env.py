# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from Perfil_topografico import read_data, compute_accumulated_distance
from compute_power import compute_power

class EVTOLFlightEnv(gym.Env):
    def __init__(self, data, render_mode=None):
        super().__init__()
        self.data=data
        #---------------- PARAMETROS DE LA SIMULACION --------------------------

        self.g=9.81 #Aceleración debido a la gravedad (m/s^2)
        self.dt=0.1 #Timestep (s)
        self.rho=1.00545 #Densidad (kg/m^3)


        self.radio=data["R"] #Radio de los rotores (m)
        self.CD0=data["CD0"] #CD0 del perfil de las palas
        self.solidity=data["solidity"] #Solidez
        self.masa=data["masa"] #(kg)
        self.f=data["f"] # Flat area equivalence (m^2)
        self.V_tip=data["V_tip"] #Velocidad de punta de pala (m/s)
        self.N=data["N"] #Número de rotores
        self.p_disp_r=data["p_disp_r"] #Potencia disponible por rotor (W)

        self.kappa=data["kappa"] #Factor de correcion k (para el CPi)
        self.K=data["K"] #Factor de corrección K (para el CP0)
        
        
        self.peso=self.masa*self.g #Peso (N)
        self.A_rotor=3.141592*(self.radio**2) #(m^2)
        
        self.modo="terreno real" #["terreno real" o "terreno plano]
        self.eleccion_objetivo="destino"

        #-------------- CARGA DEL TERRENO -----------------------
        if self.modo == "terreno real":
            latitudes, longitudes, self.altitudes = read_data('Perfil topografico.txt')
            self.distances = compute_accumulated_distance(latitudes,longitudes)
            self.origen=(self.distances[0], self.altitudes[0]) #Coordenadas del punto de origen
            self.destino=(self.distances[-1], self.altitudes[-1]) #Coordenadas del punto de destino
            indice=np.argmax(self.altitudes)
            self.punto_mas_alto=(self.distances[indice], self.altitudes[indice])  #Coordenadas del punto más alto
            self.objetivo_dict={
                "destino":self.destino,
                "punto mas alto":self.punto_mas_alto}
            
        #--------------- TERRENO PLANO PARA ENTRENAMIENTO --------------------
        elif self.modo == "terreno plano":
            self.distances=np.arange(0,1000,1 )
            self.altitudes=np.zeros(len(self.distances))
        #-------------------- CONSTANTES DE NORMALIZACION ----------------------

        self.E_max = data["E_max"]  # Energía máxima (julios)
        self.dist_max = max(self.distances)*1.5 # Distancia máxima esperada al objetivo (m)
        self.alt_max = max(self.altitudes)+1000      # Altitud máxima esperada (m)
        self.v_max = 100.0        # Velocidad máxima esperada (m/s)


        #---------------------- PESOS DE LA RECOMPENSA -------------------------
        self.w_dist=1.0
        self.w_altitud=0.3
        self.w_energy=0.2
        self.gamma=0.99
        self.R_crash=200
        self.R_success=200
        self.R_timeout=200
        self.R_sin_energia=200

        # ----------------------------- OBSERVACION ----------------------------

        self.L_scan = 500.0       # Longitud del perfil del terreno a escanear (m)
        self.N_scan = 20          # Número de puntos de muestreo del terreno
        obs_dim= 12 +self.N_scan

         # [dx, dz, vx, vz, dist, cos_angle, sin_angle,
         #  vel_error_x, vel_error_z, speed, speed_error]
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=(obs_dim,), dtype=np.float32)


        #-------------------------- ESPACIO DE ACCIONES-------------------------
        #action[0]=Tau -> {-1,1} ->[0,2*T_max]
        #action[1]=theta -> {-1,1} -> [-60,60] -> 0 grados con la vertical
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32)

        T_max_rotor = (self.p_disp_r * np.sqrt(2*self.rho*self.A_rotor)/self.kappa)**(2/3)
        self.T_max = T_max_rotor * self.N
        self.angulo_maximo=np.arccos(self.peso/self.T_max)


        #------------------- CURRICULUM PARA RANDOMIZACION ---------------------

        #SAMPLEO DE UN CIRCULO - CENTRADO EN UN PUNTO Y UN NUMERO RANDOM DE ANGULO Y DE RADIO
        self.radio_curr=20.0 #radio inicial (m)
        self.radio_minimo=10.0
        self.radio_maximo=1500.0

        # Para control automático del currículum (actualiza fuera con métricas)
        self.success_ema = 0.0
        self.success_beta = 0.02  # suavizado EMA


        self.randomize_start = False

    def _auto_set_max_steps(self, dist_inicial):
        """Establece max_steps basado en la distancia inicial"""
        # Asume velocidad promedio de 20 m/s
        tiempo_estimado = dist_inicial / 20.0
        self.max_steps = int(tiempo_estimado / self.dt * 2)  # Factor 2 de seguridad
        self.max_steps = max(self.max_steps, 500)  # Mínimo 500 steps

    def get_obs(self):
        # --- Posición y navegación ---
        dx = self.objetivo[0] - self.x
        dz = self.objetivo[1] - self.z
        dist = np.linalg.norm([dx, dz])
        cos_ψ = dx / dist if dist != 0 else 1.0
        sin_ψ = dz / dist if dist != 0 else 0.0

        # --- Velocidades ---
        vx, vz = self.vx, self.vz
        dvx = vx - self.vel_objetivo[0]
        dvz = vz - self.vel_objetivo[1]
        speed = np.linalg.norm([vx, vz])
        speed_error = np.linalg.norm([dvx, dvz])

        # --- Energía y SoC ---
        SoC = 1.0 - self.energy / self.E_max
        SoC = np.clip(SoC, 0.0, 1.0)

        # --- Altura del terreno actual ---
        z_terreno = np.interp(self.x, self.distances, self.altitudes)
        self.altura_relativa = self.z - z_terreno

        # --- Perfil del terreno hacia adelante ---
        dir_x=1.0 if dx >= 0 else -1.0 #en qué dirección se está moviendo

        perfil_delante = np.linspace(0, self.L_scan, self.N_scan)
        scan_x = self.x + perfil_delante * dir_x
        perfil_terreno = np.interp(scan_x, self.distances, self.altitudes)
        perfil_relativo = perfil_terreno - self.z
        perfil_norm = np.clip(perfil_relativo / self.alt_max, -1.0, 1.0)

        # --- Normalizaciones (si no usas VecNormalize) ---
        dx_norm = np.clip(dx / self.dist_max, -1.0, 1.0)
        dz_norm = np.clip(dz / self.alt_max, -1.0, 1.0)
        #dist_norm = np.clip(dist / self.dist_max, 0.0, 1.0)
        vx_norm = np.clip(vx / self.v_max, -1.0, 1.0)
        vz_norm = np.clip(vz / self.v_max, -1.0, 1.0)
        dvx_norm = np.clip(dvx / self.v_max, -1.0, 1.0)
        dvz_norm = np.clip(dvz / self.v_max, -1.0, 1.0)
        speed_norm = np.clip(speed / self.v_max, 0.0, 1.0)
        speed_error_norm = np.clip(speed_error / self.v_max, 0.0, 1.0)
        altura_relativa_norm = np.clip(self.altura_relativa / self.alt_max, 0.0, 1.0)


        # --- Vector final de observaciones ---
        obs_features = np.array([
            dx_norm, dz_norm , # REMOVIDOS (HER los añade)
            cos_ψ, sin_ψ,       # Dirección al objetivo
            vx_norm, vz_norm,
            dvx_norm, dvz_norm,
            speed_norm, speed_error_norm,
            altura_relativa_norm,
            SoC,
            *perfil_norm
        ], dtype=np.float32)

        return obs_features




    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.randomize_start:
            self.x, self.z = self.posibles_origenes[self.np_random.integers(len(self.posibles_origenes))]
            self.vx = 0.0
            self.vz = 0.0
        else:
            # Inicio fijo más cercano al objetivo
            self.x = 0.0
            self.z = self.altitudes[0] + 10.0
            self.vx = 0.0
            self.vz = 1.0
            self.objetivo=np.array(self.objetivo_dict[self.eleccion_objetivo])
            self.tolerancia_radio_objetivo = 20.0 #(m)
            self.vel_objetivo=np.array([0.0, 0.0])
            self.tolerancia_velocidad=None


        #-------------------- RETORNOS -------------------------
        #Para evaluar la recompensa en el entrenamiento
        self.retornos={"retorno_total":0,
                  "retorno_distancia":0,
                  "retorno_clearence":0,
                  "retorno_energia":0,
                      }

        #------------------- INICIALIZACION DE VARIABLES -----------------------
        self.SoC=1.0                                           #State of Charge (%)
        self.energy = 0.0                                      #Energía consumida (Julios)
        self.umbral = 0.05                                     # Para SoC mínimo
        self.step_count = 0                                    #Contador
        p = np.array([self.x, self.z])                         #Posicion x,z
        vel_vec=np.array([self.vx, self.vz])
        self.prev_pos = p.copy()                               #Para las recompensas (se actualiza cada timestep)
        self.d0 = float(np.linalg.norm(self.objetivo - p))     #Distancia inicial (metros)
        self.dist_prev=self.d0                                 #Para las recompensas (se actualiza cada timestep)
        self.prev_vel_error = np.linalg.norm(vel_vec - self.vel_objetivo) #Para las recompensas (se actualiza cada timestep)
        self.prev_speed=np.linalg.norm(vel_vec)#Para observar la variacion de velocidad
        self.longitud=0.0
        self.velocidades=[]
        self.alturas_relativas=[]
        self.velocidades.append(self.prev_speed)                #Estoy agregando la primera velocidad a la lista de velocidades
        self.variacion_velocidad=[]                             #Longitud de la trayectoria (metros)
        self.h_min = 500.0                                     # Altitud minima regulatoria (m)

        #---------------------FLAGS---------------------------
        self.verbose = False
        self.debug = True

        #-------------------- MAX STEPS en funcion de la distancia inicial---------------
        self._auto_set_max_steps(self.d0)

        #-------------------- OBSERVACION -----------------------
        obs = self.get_obs()

        return obs, {}


    def _propagacion_dinamica(self,Tvec,vel_vec):

      speed=np.linalg.norm(vel_vec)

      Tx=Tvec[0]
      Tz=Tvec[1]

      if speed > 1e-6:
          D_mag = 0.5 * self.rho * (speed**2) * self.f #magnitud del arrastre aerodinamico
          Fd = - D_mag * (vel_vec / speed)  #vector de arrastre (en dirección de la velocidad pero en sentido contrario)
      else:
          Fd= np.array([0.0, 0.0])


      ax = (Tx + Fd[0]) / self.masa
      az = (Tz + Fd[1] - self.peso) / self.masa

      # Integración
      vx_new = self.vx + ax * self.dt
      vz_new = self.vz + az * self.dt
      x_new = self.x + vx_new * self.dt
      z_new = self.z + vz_new * self.dt

      pos = np.array([x_new, z_new])
      vel_vec_new = np.array([vx_new, vz_new])
      speed_new = np.linalg.norm(vel_vec_new)

      return x_new, z_new, vx_new, vz_new, pos, vel_vec_new, speed_new

    def clearance_penalty(self, state):
        """Penalización por altura (banda óptima)"""
        z = state["z"]
        terrain_height = np.interp(state["x"], self.distances, self.altitudes)
        h_clr = z - terrain_height
        
        h_min = 500.0
        h_opt_low = 700.0
        h_opt_high = 1000.0
        h_max_eff = 1500.0
        
        if h_clr < h_min:
            return 5.0 * ((h_min - h_clr) / h_min) ** 2
        elif h_clr < h_opt_low:
            return 0.5 * ((h_opt_low - h_clr) / (h_opt_low - h_min)) ** 2
        elif h_clr <= h_opt_high:
            return 0.0
        elif h_clr <= h_max_eff:
            return 0.2 * (h_clr - h_opt_high) / (h_max_eff - h_opt_high)
        else:
            return 1.0 * ((h_clr - h_max_eff) / h_max_eff) ** 1.5
    
    def potential_distance(self, state):
        """Potencial solo de distancia al objetivo"""
        pos = np.array([state["x"], state["z"]])
        d = np.linalg.norm(pos - self.objetivo)
        phi_dist = - d / self.dist_max
        return phi_dist
    
    def potential_clearance(self, state):
        """Potencial solo de clearance"""
        C = self.clearance_penalty(state)
        phi_clr = - C  # Ya no necesita normalización adicional
        return phi_clr


    def step(self, action):
      self.step_count+=1 #Suma contador

      #Estado fisico
      x,z,vx,vz = self.x,self.z,self.vx,self.vz
      state = {"x":self.x, "z":self.z, "vx":self.vx, "vz":self.vz}

      #Recortar acción de -1, 1
      a=np.clip(action, -1.0, 1.0)

      theta=self.angulo_maximo*a[1] #angulo entre vector de empuje y la vertical

      T_hover=self.peso
      Tmag=np.clip(T_hover+T_hover*a[0], 0.0, self.T_max) #magnitud de empuje, centrado en T_hover

      Tz=Tmag*np.cos(theta)
      Tx=Tmag*np.sin(theta)

      #Vector de empuje
      Tvec=np.array([Tx,Tz])
      vel_vec=np.array([vx,vz])

      #Nuevo estado luego de la acción realizada
      x,z,vx,vz,pos,vel_vec,speed=self._propagacion_dinamica(Tvec,vel_vec)
      
      self.velocidades.append(speed)
      self.alturas_relativas.append(self.altura_relativa)

      #Calculo de longitud y variacion de la velocidad y de la distancia
      self.longitud+=np.linalg.norm(pos-self.prev_pos)
      self.variacion_velocidad.append(speed-self.prev_speed)
      dist_act=float(np.linalg.norm(self.objetivo-pos))

      #Actualizacion de posicion y velocidad
      self.prev_pos=pos.copy()
      self.prev_speed=speed


      #Actualización del estado
      self.x, self.z, self.vx, self.vz=x,z,vx,vz
      new_state = {"x":self.x, "z":self.z, "vx":self.vx, "vz":self.vz}


      #Calculo de la potencia y de la energía
      potencias,info_vi=compute_power(Tvec,vel_vec,self.rho,self.data)
      power=potencias["P_total"]
      delta_energy = max(power * self.dt, 0)  # No permitir energía negativa
      self.energy = self.energy + delta_energy

      #Actualizacion del SoC
      self.SoC=np.clip(1.0-self.energy/self.E_max , 0.0, 1.0)


      # Observación
      obs = self.get_obs()

      # ----------------------- RECOMPENSA ---------------------------------


        
      # 1. REWARD SHAPING DE DISTANCIA
      phi_dist_s = self.potential_distance(state)
      phi_dist_sp = self.potential_distance(new_state)
      r_distance_shaping = self.w_dist * (self.gamma * phi_dist_sp - phi_dist_s)
            
      # 2. REWARD SHAPING DE CLEARANCE
      phi_clr_s = self.potential_clearance(state)
      phi_clr_sp = self.potential_clearance(new_state)
      r_clearance_shaping = self.w_altitud * (self.gamma * phi_clr_sp - phi_clr_s)
        
      # 3. PENALIZACIÓN DE ENERGÍA
      r_energy = - self.w_energy * delta_energy / self.E_max


      #------------------------- TERMINALES --------------------------------
      terminated = False
      truncated = False
      success = False
      crash = False
      ooe= False #Out of energy
      r_terminal = 0.0
      end_reason = None

      #---------------------------CHOQUE ------------------------------------

      terrain_height = np.interp(self.x, self.distances, self.altitudes)
      crashed=(self.z < terrain_height) or (self.z < 0)

      if crashed or (abs(self.x) > self.dist_max) or (self.z > self.alt_max):
          terminated = True
          crash = True
          r_terminal = -self.R_crash
          end_reason = "Choque"


      #------------------------ EXITO ----------------------------------------
      if (dist_act <= self.tolerancia_radio_objetivo) and not crash:
          r_terminal = self.R_success
          success = True
          terminated = True
          end_reason = "Llegooooo"

      #----------------------- SIN ENERGIA ----------------------------------
      if self.SoC < self.umbral and not terminated:

          r_terminal = -self.R_sin_energia
          truncated = True
          ooe= True
          end_reason = "Sin energía"


      #------------------------ TIMEOUT -----------------------------------
      if self.step_count >= self.max_steps and not terminated:

          r_terminal = -self.R_timeout
          truncated = True
          end_reason = "sin tiempo"



      reward=r_distance_shaping + r_clearance_shaping +r_energy+r_terminal
      
      self.retornos["retorno_total"]+=reward
      self.retornos["retorno_distancia"]+=r_distance_shaping
      self.retornos["retorno_clearence"]+=r_clearance_shaping
      self.retornos["retorno_energia"]+=r_energy

             
      #Actualizacion de distancia previa
      self.dist_prev=dist_act

      #Variables para el logging
      info = {
            "success": success,
            "end_reason": end_reason,
            "distancia_final": dist_act,
            "SoC": self.SoC,
            "energy_used": self.energy,
            "potencia": power,
            "tiempo (s)": float(self.step_count*self.dt),
            "longitud del vuelo": float(self.longitud),
            
            # Velocidades para calcular la velocidad promedio y la variacion de velocidad en cada episodio
            "velocidad promedio":np.mean(self.velocidades),
            "altitud promedio":np.mean(self.alturas_relativas),

            #Información de las condiciones iniciales
            "coordenadas meta":self.objetivo,
            "distancia inicial": self.d0,
            
            #Recompensas aisladas por timestep- PARA LA EVOLUCIÓN DE UN EPISODIO
            "r_distance": r_distance_shaping,
            "r_clearence": r_clearance_shaping,
            "r_energy": r_energy,
            "r_terminal": r_terminal,
            "total_reward": reward,
            
            
            
            
            #Retornos por episodio - PARA EL ENTRENAMIENTO
            "r_distance_total": self.retornos["retorno_distancia"],
            "r_clearance_total": self.retornos["retorno_clearence"],
            "r_energy_total": self.retornos["retorno_energia"],
            "r_terminal_total": r_terminal,
            "r_total": self.retornos["retorno_total"],
            
            # Proporciones (para ver dominancia)
            "r_distance_pct": self.retornos["retorno_distancia"] / 
                              (abs(self.retornos["retorno_total"]) + 1e-8) * 100,
            "r_clearance_pct": self.retornos["retorno_clearence"] / 
                               (abs(self.retornos["retorno_total"]) + 1e-8) * 100,
            "r_energy_pct": self.retornos["retorno_energia"] / 
                            (abs(self.retornos["retorno_total"]) + 1e-8) * 100,
            
            "state": new_state
            
                            }

     

      return obs, float(reward), terminated, truncated, info

    def render(self):
        pass
