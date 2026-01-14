import numpy as np
import matplotlib.pyplot as plt
from geopy.distance import geodesic

def read_data(filename):
    data = np.loadtxt(filename)
    latitudes = data[:, 2]
    longitudes = data[:, 1]
    altitudes = data[:, 3]
    return latitudes, longitudes, altitudes

def compute_accumulated_distance(latitudes, longitudes):
    distances = [0.0]  # Start at 0 km
    for i in range(1, len(latitudes)):
        prev_point = (latitudes[i-1], longitudes[i-1])
        curr_point = (latitudes[i], longitudes[i])
        d = geodesic(prev_point, curr_point).meters
        distances.append(distances[-1] + d)
    return np.array(distances)

def plot_profile(distances, altitudes):
    min_alt = np.min(altitudes)
    max_alt = np.max(altitudes)
    margin = (max_alt - min_alt) * 0.25  # 5% margin

    plt.figure(figsize=(10, 5))
    plt.plot(distances, altitudes, color='green')
    plt.fill_between(distances, altitudes, color='lightgreen', alpha=0.5)
    plt.title('Perfil topográfico')
    plt.xlabel('Distancia acumulada (m)')
    plt.ylabel('Altitud (m)')
    plt.ylim(min_alt - margin, max_alt + margin)  # Set trimmed y-axis
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    filename = 'Perfil topografico.txt'  # Replace with your file path
    latitudes, longitudes, altitudes = read_data(filename)
    distances = compute_accumulated_distance(latitudes, longitudes)
    plot_profile(distances, altitudes)


def terrain_h_at(x, distances, altitudes):
    """Altura de terreno h(x) por interpolación lineal en el perfil 1D."""
    x = np.clip(x, distances[0], distances[-1])
    return float(np.interp(x, distances, altitudes))

def clearance_ahead_1d(x, z, distances, altitudes, L_ahead, n_samples=64):
    """
    Clearance mínima en [x, x + L_ahead] siguiendo el perfil 1D.
    Retorna (clear_min, z_profile_min, x_at_min)
    """
    x_end = np.clip(x + L_ahead, distances[0], distances[-1])
    if x_end <= x + 1e-6:
        h_here = terrain_h_at(x, distances, altitudes)
        return max(z - h_here, 0.0), h_here, x

    xs = np.linspace(x, x_end, n_samples)
    hs = np.interp(xs, distances, altitudes)
    clear = z - hs
    idx = int(np.argmin(clear))
    return float(max(clear[idx], 0.0)), float(hs[idx]), float(xs[idx])

def max_slope_ahead_1d(x, distances, altitudes, L_ahead, n_samples=128):
    """
    Pendiente máxima |dh/dx| en [x, x + L_ahead], aproximada por diferencias finitas.
    (unidades: rad ≈ atan(dh/dx) si quieres convertir luego)
    """
    x_end = np.clip(x + L_ahead, distances[0], distances[-1])
    if x_end <= x + 1e-6:
        return 0.0
    xs = np.linspace(x, x_end, n_samples)
    hs = np.interp(xs, distances, altitudes)
    dx = np.diff(xs)
    dh = np.diff(hs)
    slope_abs = np.max(np.abs(dh / (dx + 1e-8)))
    return float(slope_abs)

def lookahead_profile_1d(x, z, distances, altitudes, look_dists):
    """
    Vector de 'clearance' en puntos de avance específicos: clear_i = z - h(x + d_i).
    Retorna np.array con misma longitud que look_dists.
    """
    xs = np.array([np.clip(x + d, distances[0], distances[-1]) for d in look_dists], dtype=float)
    hs = np.interp(xs, distances, altitudes)
    clear = z - hs
    return clear.astype(float)



if __name__ == '__main__':
    main()


