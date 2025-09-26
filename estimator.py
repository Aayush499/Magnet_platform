import pyswarms as ps
import numpy as np
import pygame
import serial
import sys
import time
import math
import csv
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
import pygame
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
plt.ion()  # Interactive mode


#--- Visualization parameters
WIN_SIZE = 600
PLATFORM_SIZE = 0.26416  # meters (length of one platform side)
MARGIN = 60

#--- Initialize pygame for display
pygame.init()
screen = pygame.display.set_mode((WIN_SIZE, WIN_SIZE))
pygame.display.set_caption("Magnetic Stylus Tracker")
clock = pygame.time.Clock()


SERIAL_PORT = '/dev/ttyACM0'  # adjust as needed
BAUD_RATE = 230400
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10
MAGNET_STRENGTH1 = 1.1624647114675248 
MAGNET_STRENGTH_RATIO = .5

sensor_mask = [True, False, True, True, True, True, True, False]
WORKING_SENSOR_COUNT = sum(sensor_mask)
import numpy as np


def calibrate_ambient_field(ser, n_frames=10):
    """Calibrate by measuring average field at each sensor with no stylus present."""
    frames = []
    print("Calibrating ambient magnetic field...")
    for _ in range(n_frames):
        frame = read_sensor_frame(ser)  # your existing function
        #check if frame[working_indices] is not None
        if frame is not None and all(any(frame[i][j] is not None for j in range(3))  for i in working_indices):
            frames.append(frame)
    #for all indices i false in working_indices, set frame[i] to (0,0,0)

    frames = np.array(frames)  # shape (n_frames, sensor_count, 3)
    for i in range(SENSOR_COUNT):
        if i not in working_indices:
            frames[:, i] = (0, 0, 0)
    ambient_field = np.mean(frames, axis=0)  # shape (sensor_count, 3)
    #change it back to None for broken sensors
    for i in range(SENSOR_COUNT):
        if i not in working_indices:
            ambient_field[i] = (None, None, None)
    print("Ambient field (Earth + background):", ambient_field)
    return ambient_field


def dipole_field(m, r, r0):
    """
    Calculates magnetic field at positions r from dipole at r0 with moment m.
    m: (3,) moment vector
    r: (N,3) sensor positions
    r0: (3,) magnet position
    Returns: (N,3) predicted B vectors
    """
    mu0_4pi = 1e-7
    R = r - r0  # shape (N,3)
    normR = np.linalg.norm(R, axis=1)[:, None]
    mdotR = np.sum(m * R, axis=1)[:, None]
    B = mu0_4pi * (3 * mdotR * R / normR**5 - m / normR**3)
    return B

def moment_from_angles(pitch, yaw, mag=MAGNET_STRENGTH1):  # adjust mag as needed for your magnet
    mx = mag * np.sin(pitch) * np.cos(yaw)
    my = mag * np.sin(pitch) * np.sin(yaw)
    mz = mag * np.cos(pitch)
    return np.array([mx, my, mz])

def cost(params, sensor_positions, measured_B):
    x, y, z, pitch, yaw = params
    r0 = np.array([x, y, z])
    m = moment_from_angles(pitch, yaw)
    B_pred = dipole_field(m, sensor_positions, r0)
    return np.sum((B_pred - measured_B)**2)

def cost_pso(X):
    # X shape: (n_particles, 5)
    costs = np.zeros(X.shape[0])
    for i, params in enumerate(X):
        costs[i] = cost(params, sensor_positions, measured_B)
    return costs

# -- sensor_positions and measured_B must be defined each frame! --
# Example sensor positions (replace with your actual layout, units in meters):
full_sensor_positions = np.array([
    (0.0, 0.26416, 0.0),         # Sensor 0
    (0.0, 0.13208, 0.0),        # Sensor 1
    (0.0, 0.0, 0.0),        # Sensor 2
    (0.13208, 0.0, 0.0),       # Sensor 3
    (0.26416, 0.0, 0.0),       # Sensor 4
    (0.26416, 0.13208, 0.0),       # Sensor 5
    (0.26416, 0.26416, 0.0),        # Sensor 6
    (0.13208, 0.26416, 0.0),        # Sensor 7 (broken)
])

# measured_B = np.array(...)  # shape (7,3), filled from your Arduino/frame
sensor_positions = full_sensor_positions[np.array(sensor_mask)]
working_indices = [i for i, ok in enumerate(sensor_mask) if ok]
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def read_sensor_frame(ser):
    """Reads next complete set of magnetometer readings as (N,3) array."""
    values = [(None, None, None)] * SENSOR_COUNT # only for working sensors
    while True:
        line = ser.readline().decode().strip()
        if not line:
            continue
        if line == "BREAK":
            working_values = [v for v, ok in zip(values, sensor_mask) if ok]
            if all(v is not None for v in working_values):
                return np.array(values)
            else:
                # Skip incomplete frames
                values = [None] * SENSOR_COUNT
                continue
        # Handle sensor data lines
        parts = line.split(",")
        if len(parts) == 4:
            try:
                idx = int(parts[0])  # sensor index, can ignore or check order
                Bx = float(parts[1]) / 1e6
                By = float(parts[2]) / 1e6
                Bz = float(parts[3]) / 1e6
                values[idx] = (Bx, By, Bz)
            except ValueError:
                continue

#--- Helper: Map real coords to screen
def map_pos(x, y):
    # x, y in meters (platform coords: 0 to PLATFORM_SIZE)
    sx = int(MARGIN + x * (WIN_SIZE - 2*MARGIN) / PLATFORM_SIZE)
    sy = int(WIN_SIZE - (MARGIN + y * (WIN_SIZE - 2*MARGIN) / PLATFORM_SIZE))
    return sx, sy

def read_calibration_frame(ser, sensor_idx=0, avg_secs=4):
    """Reads and averages Bz value from the chosen sensor for avg_secs."""
    print(f"Hold magnet over sensor {sensor_idx} at known height, and press Enter to start calibration averaging for {avg_secs} seconds.")
    input("Press Enter when ready...")
    values = []
    t0 = time.time()
    while time.time() - t0 < avg_secs:
        frame = read_sensor_frame(ser)
        Bx, By, Bz = frame[sensor_idx]  # assuming frame in Tesla!
        values.append(Bz)
        time.sleep(0.05)
    mean_Bz = np.mean(values)
    print(f"Mean Bz over {avg_secs}s: {mean_Bz:.6e} T")
    return mean_Bz

def calibrate_moment(ser, sensor_idx=0, height_m=0.02, avg_secs=4):
    print(f"Calibration: Place magnet {height_m} meters above sensor {sensor_idx}.")
    mean_Bz_raw = read_calibration_frame(ser, sensor_idx=sensor_idx, avg_secs=avg_secs)
    mean_Bz = mean_Bz_raw - ambient_field[sensor_idx][2]  # subtract ambient field Bz
    # Dipole moment estimation
    mu0_over_4pi = 1e-7
    m = mean_Bz * (height_m ** 3) / (2 * mu0_over_4pi)  # SI units, A·m²
    print(f"Estimated magnetic moment: m = {m:.3e} A·m²")
    return m 


ambient_field = calibrate_ambient_field(ser)
ambient_field = ambient_field[np.array(sensor_mask)]  # filter by mask

# m = [None] * SENSOR_COUNT
# # for i in (range(SENSOR_COUNT)):
# #     m_true = calibrate_moment(ser, sensor_idx=i, height_m=0.0762, avg_secs=10)
# #     m[i] = m_true
 
# #save m
# # np.save("calibrated_moments.npy", m)
# m = np.load("calibrated_moments.npy", allow_pickle=True).tolist()
# m[4] = calibrate_moment(ser, sensor_idx=4, height_m=0.0762, avg_secs=10)
# np.save("calibrated_moments.npy", m)  # save updated moments
# #get average of m
# m_avg = np.mean([mi for mi in m ])
# print("Use this value as MAGNET_STRENGTH1:", m_avg)


 
def record_swarm_guesses_over_sensor(sensor_idx, duration_secs=10):
    input(f"Place stylus over sensor {sensor_idx}. Press Enter when ready to start recording for {duration_secs} seconds...")
    guesses = []
    t_start = time.time()
    while (time.time() - t_start) < duration_secs:
        measured_B_raw = read_sensor_frame(ser)
        measured_B_raw = measured_B_raw[np.array(sensor_mask)]  # filter by mask
        measured_B = measured_B_raw - ambient_field
        
        # run PSO:
        best_cost, best_pos = optimizer.optimize(cost_pso, iters=100)
        now = time.time() - t_start
        guesses.append({
            'timestamp': now,
            'sensor_idx': sensor_idx,
            'best_cost': best_cost,
            'x': best_pos[0],
            'y': best_pos[1],
            'z': best_pos[2],
            'pitch': best_pos[3],
            'yaw': best_pos[4]
        })
        # small sleep to avoid unnecessary oversampling (optional)
        time.sleep(0.1)
    # Save to CSV
    filename = f'swarm_guesses_sensor{sensor_idx}.csv'
    with open(filename, 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=guesses[0].keys())
        writer.writeheader()
        writer.writerows(guesses)
    print(f"Done. Saved guesses to {filename}.")




testing_loop = False
last_best_pos = None
alpha = 0.6  # Between 0 and 1
while True:
    measured_B_raw = read_sensor_frame(ser)  # updated with each "BREAK"
    measured_B_raw = measured_B_raw[np.array(sensor_mask)]  # filter by mask
    measured_B = measured_B_raw - ambient_field  # shape (7,3)
    # print(f"Measured B: {measured_B_raw}")

    # PSO options
    options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}
    # bounds = (
    #     np.array([0, 0, 0, -np.pi/2, -np.pi]),   # min: x,y,z,pitch,yaw
    #     np.array([0.3, 0.3, 0.2, np.pi/2, np.pi]) # max: x,y,z,pitch,yaw
    # )
    if last_best_pos is not None:
        center = np.array(last_best_pos)
        width = np.array([0.03, 0.03, 0.02, np.pi/12, np.pi/12])  # Reasonable max movement per frame
        bounds = (np.maximum(center - width, bounds[0]), np.minimum(center + width, bounds[1]))
        filtered_pos = best_pos
    else:
        bounds = (
            np.array([0, 0, 0, -np.pi/2, -np.pi]),
            np.array([0.3, 0.3, 0.2, np.pi/2, np.pi])
        )
        
    

    # Create a PSO optimizer for 5 parameters
    optimizer = ps.single.GlobalBestPSO(
        n_particles=40,
        dimensions=5,
        options=options,
        bounds=bounds,

    )

    if testing_loop:
        for sensor_to_test in working_indices:
            record_swarm_guesses_over_sensor(sensor_idx=sensor_to_test, duration_secs=30)
            print(f"Now, move to next sensor.")
        break  # exit after testing all sensors
    # Run optimization (per frame!)
    best_cost, best_pos = optimizer.optimize(cost_pso, iters=100, n_processes=10)
    if last_best_pos is None:
        filtered_pos = best_pos
    else:
        filtered_pos = alpha * np.array(best_pos) + (1 - alpha) * np.array(last_best_pos)
    # last_best_pos = filtered_pos
    # last_best_pos = best_pos

    # for event in pygame.event.get():
    #     if event.type == pygame.QUIT:
    #         pygame.quit()
    #         ser.close()
    #         sys.exit()

    # screen.fill((30, 30, 40))

    # # Draw platform boundary
    # top_left = map_pos(0, PLATFORM_SIZE)
    # bottom_right = map_pos(PLATFORM_SIZE, 0)
    # pygame.draw.rect(screen, (80, 80, 120), (*top_left, bottom_right[0]-top_left[0], bottom_right[1]-top_left[1]), 3)

    # # Draw sensors
    # for pos in sensor_positions:
    #     x, y, _ = pos
    #     sx, sy = map_pos(x, y)
    #     pygame.draw.circle(screen, (255, 255, 80), (sx, sy), 10)
    
    # # Draw stylus
    # stylus_x, stylus_y, stylus_z = best_pos[:3]
    # s_sx, s_sy = map_pos(stylus_x, stylus_y)

    # # Stylus orientation as a line segment ("direction" in the xy-plane)
    # pitch, yaw = best_pos[3], best_pos[4]
    # # Project stylus direction onto xy-plane
    # dx = math.cos(yaw) * 50  # arbitrary line length scaling
    # dy = math.sin(yaw) * 50
    # end_pt = (int(s_sx + dx), int(s_sy - dy))
    # pygame.draw.line(screen, (200, 70, 70), (s_sx, s_sy), end_pt, 6)

    # # Stylus location
    # pygame.draw.circle(screen, (70, 200, 255), (s_sx, s_sy), 14)

    # # Optionally draw z-height as circle size or color
    # stylus_color = (int(255-120*stylus_z), 70, int(255*stylus_z/0.2))
    # pygame.draw.circle(screen, stylus_color, (s_sx, s_sy), 10+int(10*stylus_z/0.2), 2)

    # pygame.display.flip()
    ax.clear()
    # Draw sensors
    for pos in sensor_positions:
        ax.scatter(*pos, color='gold', s=60)
    # Draw stylus
    x, y, z = filtered_pos[:3]

    # Stylus orientation vector (pitch/yaw)
    pitch, yaw = filtered_pos[3], filtered_pos[4]
    mag_length = 0.03   # show orientation as a 3cm stick
    dx = mag_length * math.sin(pitch) * math.cos(yaw)
    dy = mag_length * math.sin(pitch) * math.sin(yaw)
    dz = mag_length * math.cos(pitch)
    ax.quiver(x, y, z, dx, dy, dz, color='crimson', lw=2)

    ax.scatter(x, y, z, color='deepskyblue', s=120)
    ax.set_xlim([0, PLATFORM_SIZE])
    ax.set_ylim([0, PLATFORM_SIZE])
    ax.set_zlim([0, 0.2])

    plt.draw()
    plt.pause(0.001)
    print(best_pos)

    clock.tick(60)



    # print(f"Best Cost: {best_cost}")
    # print(f"Estimated Position: {best_pos[:3]}, Pitch: {best_pos[3]}, Yaw: {best_pos[4]}")


