import pyswarms as ps
import numpy as np
import pygame
import serial
import sys
import time
import math
import csv
  
import pygame
 
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

 

#--- Visualization parameters
WIN_SIZE = 600
PLATFORM_SIZE = 0.26416  # meters (length of one platform side)
MARGIN = 60




SERIAL_PORT = '/dev/ttyACM0'  # adjust as needed
BAUD_RATE = 1000000
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10
MAGNET_STRENGTH1 = 1.1624647114675248 
MAGNET_STRENGTH_RATIO = .5

sensor_mask = [True, True, True, True, True, True, True, False]
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




ambient_field = calibrate_ambient_field(ser)
ambient_field = ambient_field[np.array(sensor_mask)]  # filter by mask



def init_visualization():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(10, (display[0]/display[1]), 0.01, 2.0)
    glEnable(GL_DEPTH_TEST)
    global clock
    clock = pygame.time.Clock()
    # camera state if needed
    global camera_pos, camera_rot_x, camera_rot_y
    camera_pos = [0.0, 0.0, -0.05]
    camera_rot_x, camera_rot_y = 30, 0

def draw_sphere(pos, radius=0.01, color=(1,1,0)):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    quad = gluNewQuadric()
    gluSphere(quad, radius, 16, 12)
    glPopMatrix()

def draw_arrow(start, direction, length=0.04, color=(1,0,0)):
    glColor3f(*color)
    direction = np.array(direction) / np.linalg.norm(direction)
    end = start + direction * length
    glBegin(GL_LINES)
    glVertex3fv(start)
    glVertex3fv(end)
    glEnd()
    # Simple arrowhead (optional)
    arrow_dir = direction
    if not np.allclose(arrow_dir, [0,0,1]):
        side = np.cross(arrow_dir, [0, 0, 1])
    else:
        side = np.cross(arrow_dir, [0,1,0])
    side = side / np.linalg.norm(side)
    glBegin(GL_LINES)
    glVertex3fv(end)
    glVertex3fv(end - 0.01 * arrow_dir + 0.004 * side)
    glVertex3fv(end)
    glVertex3fv(end - 0.01 * arrow_dir - 0.004 * side)
    glEnd()

def get_direction_from_pitch_yaw(pitch, yaw):
    dx = math.sin(pitch) * math.cos(yaw)
    dy = math.sin(pitch) * math.sin(yaw)
    dz = math.cos(pitch)
    return np.array([dx, dy, dz])

init_visualization()

last_best_pos = None
alpha = 0.6  # Between 0 and 1
zoom_step = 0.02
rot_step = 5
pan_step = 0.02
fov = 45 
running = True
while running:
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

    
    best_cost, best_pos = optimizer.optimize(cost_pso, iters=100, n_processes=10)
    if last_best_pos is None:
        filtered_pos = best_pos
    else:
        filtered_pos = alpha * np.array(best_pos) + (1 - alpha) * np.array(last_best_pos)
    # last_best_pos = best_pos
    
    # stylus_pos = filtered_pos[:3]
    # pitch, yaw = filtered_pos[3], filtered_pos[4]
    stylus_pos = best_pos[:3]
    pitch, yaw = best_pos[3], best_pos[4]
    
    # (B) EVENT HANDLING:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == KEYDOWN:
            if event.key == pygame.K_w:
                camera_pos[2] += zoom_step  # Zoom in
            elif event.key == pygame.K_s:
                camera_pos[2] -= zoom_step  # Zoom out
            elif event.key == pygame.K_a:
                camera_pos[0] += pan_step   # Pan left
            elif event.key == pygame.K_d:
                camera_pos[0] -= pan_step   # Pan right
            elif event.key == pygame.K_q:
                camera_pos[1] += pan_step   # Pan up
            elif event.key == pygame.K_e:
                camera_pos[1] -= pan_step   # Pan down
            elif event.key == pygame.K_UP:
                camera_rot_x += rot_step    # Pitch up
            elif event.key == pygame.K_DOWN:
                camera_rot_x -= rot_step    # Pitch down
            elif event.key == pygame.K_LEFT:
                camera_rot_y += rot_step    # Yaw left
            elif event.key == pygame.K_RIGHT:
                camera_rot_y -= rot_step    # Yaw right

    
    # (C) VISUALIZATION:
    glClearColor(0.05,0.05,0.06,1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])
    glRotatef(camera_rot_x, 1, 0, 0)
    glRotatef(camera_rot_y, 0, 1, 0)

    # Sensors
    for pos in sensor_positions:
        draw_sphere(pos, radius=0.012, color=(1,1,0))

    # Stylus tip
    draw_sphere(stylus_pos, radius=0.016, color=(0,0.5,1))

    # Stylus orientation (arrow)
    stylus_dir = get_direction_from_pitch_yaw(pitch, yaw)
    draw_arrow(stylus_pos, stylus_dir, length=0.045, color=(1,0,0))

    
    
    pygame.display.flip()
    
    clock.tick(60)



    # print(f"Best Cost: {best_cost}")
    # print(f"Estimated Position: {best_pos[:3]}, Pitch: {best_pos[3]}, Yaw: {best_pos[4]}")


