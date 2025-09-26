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
NUM_GRID_LINES = 5 # Number of grid lines, includes left and right
CYLINDER_SECTOR_COUNT = 20 # Number of sectors on cylinder, how good the circle looks
Moving_average_window_size = 5  # Number of frames to average for smoothing

SERIAL_PORT = '/dev/ttyACM0'  # adjust as needed
BAUD_RATE = 115200
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10
# MAGNET_STRENGTH1 = 1.1624647114675248 
MAGNET_STRENGTH1 = 1.4
MAGNET_STRENGTH_RATIO = .5

sensor_mask = [True, True, True, True, True, True, True, False]
WORKING_SENSOR_COUNT = sum(sensor_mask)
TOTAL_SENSOR_COUNT = len(sensor_mask)

 
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

working_indices = [i for i, ok in enumerate(sensor_mask) if ok]
ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)

def calibrate_ambient_field(ser, n_frames=50):
    """Calibrate by measuring average field at each sensor with no stylus present."""
    frames = []
    print("Calibrating ambient magnetic field...")
    for _ in range(n_frames):
        frame = parse_output_matrix()  # your existing function
        #check if frame[working_indices] is not None
        if frame is not None :
            frames.append(frame)
    #for all indices i false in working_indices, set frame[i] to (0,0,0)

    frames = np.array(frames)  # shape (n_frames, sensor_count, 3)
     
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


# Centering the board positions for visualization
sensor_position_stats = {
    'mean': np.mean(full_sensor_positions, axis=0)
}
full_sensor_positions -= sensor_position_stats['mean']
sensor_position_stats.update({
    'min': np.min(full_sensor_positions, axis=0),
    'max': np.max(full_sensor_positions, axis=0)
})
default_bounds = (
    np.concat([sensor_position_stats['min'], [-np.pi/2, -np.pi]]),
    np.concat([sensor_position_stats['max'] + (0, 0, 0.2), [np.pi/2, np.pi]])
)

# Getting positions for grid lines
grid_xy = np.transpose(np.linspace(sensor_position_stats['min'], sensor_position_stats['max'], num=NUM_GRID_LINES)[:, :2])
grid_coordinates = np.column_stack(((
    np.concat((grid_xy[0], np.full(NUM_GRID_LINES, grid_xy[0][0]))),
    np.concat((np.full(NUM_GRID_LINES, grid_xy[1][0]), grid_xy[1])),
    np.concat((grid_xy[0], np.full(NUM_GRID_LINES, grid_xy[0][-1]))),
    np.concat((np.full(NUM_GRID_LINES, grid_xy[1][-1]), grid_xy[1]))
)))




def read_line(ser):
    """Read a line from the serial port."""
    while True:
        try:
            line = ser.readline().decode(errors='replace')
            if len(line.split('\n')[0].split()) != TOTAL_SENSOR_COUNT * 3:
                continue
            return line.split('\n')[0]
        except Exception as e:
            print(f"Error reading from serial port: {e}")
            return None
        
        
def parse_output_matrix():
    """Parse a line of output matrix data."""
    line = read_line(ser)
    if line:
        try:
            values = list(map(float, line.split()))
            if len(values) == TOTAL_SENSOR_COUNT * 3:
                matrix = [values[i:i+3] if i//3 in working_indices else (0, 0, 0) for i in range(0, len(values), 3)]
                #divide each value by 1e6
                matrix = [[v / 1e6 for v in row] for row in matrix]
                return np.array(matrix)
            else:
                print(f"Unexpected number of values: {len(values)}")
        except ValueError as ve:
            print(f"Value error: {ve}")
    return None

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
    # pygame.display.set_mode(flags=DOUBLEBUF|OPENGL|FULLSCREEN)
    # w, h = pygame.display.get_surface().get_size()
    # gluPerspective(10, (w/h), 0.01, 2.0)
    glEnable(GL_DEPTH_TEST)
    global clock
    clock = pygame.time.Clock()
    # camera state if needed
    global camera_pos, camera_rot_x, camera_rot_y
    camera_pos = [0.0, 0.0, 2]
    camera_rot_x, camera_rot_y = 0, 180
    global mouse_drag, mouse_pan
    mouse_drag, mouse_pan = False, False
    glClearColor(0.9,0.9,0.9,1)
    
    # Geometry precalculation
    calculate_box_vertices((0, 0, -0.01), (0.3, 0.3, 0.02))

def draw_sphere(pos, radius=0.01, color=(1,1,0)):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    quad = gluNewQuadric()
    gluSphere(quad, radius, 16, 12)
    glPopMatrix()

box_edges = ((0,1),(2,3),(4,5),(6,7),(0,2),(0,4),(2,6),(4,6),(1,3),(1,5),(3,7),(5,7))
box_surfaces = ((1,3,7,5),(0,1,3,2),(2,3,7,6),(6,7,5,4),(4,5,1,0),(0,2,6,4))
box_colors = ((0.5,0.5,0.5),(1,1,1),(0.5,0.5,0.5),(1,1,1))
vertices_precalc = []

def calculate_box_vertices(pos, size, isMiddle=True):
    min_coord = [pos[i] - (size[i] / 2) for i in range(3)] if isMiddle else pos
    max_coord = [pos[i] + (size[i] / 2) for i in range(3)] if isMiddle else [pos[i] + size[i] for i in range(3)]
    box_coord = tuple(zip(min_coord, max_coord))
    vertices = []
    for x in box_coord[0]:
        for y in box_coord[1]:
            for z in box_coord[2]:
                vertices.append((x, y, z))
    vertices_precalc.append(vertices)

def calculate_cylinder_vertices(pos, rt, rb):
    angles = np.linspace(0, 2 * math.pi, num=CYLINDER_SECTOR_COUNT + 1)[:-1]
    circ_xy = np.array((np.cos(angles) + pos[0], np.sin(angles) + pos[1]))
    print(circ_xy)

# calculate_cylinder_vertices((0, 0, 0), 1, 1)
# calculate_cylinder_vertices((1, 1, 0), 1, 1)

def draw_box(vertices, color=(0.5, 0.5, 0.5)):
    glBegin(GL_QUADS)
    for surface in box_surfaces:
        c = 0
        for v in surface:
            glColor3fv(box_colors[c])
            glVertex3fv(vertices[v])
            c += 1
    glEnd()
    glColor3f(*color)
    glBegin(GL_LINES)
    for edge in box_edges:
        for v in edge:
            glVertex3fv(vertices[v])
    glEnd()

def draw_line(x1, y1, z1, x2, y2, z2, color=(0, 0, 0)):
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y2, z2)
    glEnd()

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

sensor_positions = full_sensor_positions[np.array(sensor_mask)]

last_best_pos = None
alpha = 0.6  # Between 0 and 1
zoom_step = 1.2
rot_step = 5
pan_step = 0.002
fov = 45 
running = True

# PSO options
options = {'c1': 0.5, 'c2': 0.3, 'w': 0.9}

moving_average_window = []
while running:
    measured_B_raw = parse_output_matrix()  # updated with each "BREAK"
    measured_B_raw = measured_B_raw[np.array(sensor_mask)]  # filter by mask
    measured_B = measured_B_raw - ambient_field  # shape (7,3)
    if len(moving_average_window) <= Moving_average_window_size:
        
        moving_average_window.append(measured_B)
        continue
    else:
        moving_average_window.pop(0)
        
        moving_average_window.append(measured_B_raw)
    measured_B = np.mean(moving_average_window, axis=0)
    # print(f"Measured B: {measured_B_raw}")

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
        bounds = default_bounds

    # Create a PSO optimizer for 5 parameters
    optimizer = ps.single.GlobalBestPSO(
        n_particles=40,
        dimensions=5,
        options=options,
        bounds=bounds,
    )

    # TODO: Find typical bound change and try to form an equation to determine necessary n_particles or iters
    start_time = time.time_ns()
    best_cost, best_pos = optimizer.optimize(cost_pso, iters=50, verbose=False)
    # print((time.time_ns() - start_time) / 1000000)
    # print(bounds)
    stylus_pos = best_pos[:3]
    pitch, yaw = best_pos[3], best_pos[4]
    
    # (B) EVENT HANDLING:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == MOUSEBUTTONDOWN:
            match event.button:
                case 1:
                    mouse_drag = True
                    mouse_pos_prev_x, mouse_pos_prev_y = event.pos
                case 2:
                    mouse_pan = True
                    mouse_pos_prev_x, mouse_pos_prev_y = event.pos
                case 4:
                    camera_pos[2] *= zoom_step
                case 5:
                    camera_pos[2] /= zoom_step
        elif event.type == MOUSEBUTTONUP:
            if event.button == 1:
                mouse_drag = False
            if event.button == 2:
                mouse_pan = False
        elif event.type == MOUSEMOTION:
            if mouse_drag or mouse_pan:
                mouse_pos_curr_x, mouse_pos_curr_y = event.pos
                if mouse_drag:
                    camera_rot_x -= mouse_pos_curr_x - mouse_pos_prev_x
                    camera_rot_y -= mouse_pos_curr_y - mouse_pos_prev_y
                else:
                    camera_pos[0] += (mouse_pos_curr_x - mouse_pos_prev_x) * pan_step
                    camera_pos[1] -= (mouse_pos_curr_y - mouse_pos_prev_y) * pan_step
                mouse_pos_prev_x, mouse_pos_prev_y = event.pos

    # (C) VISUALIZATION:
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    # Z axis of camera position controls zoom, this is controlled by glScale
    glTranslatef(camera_pos[0], camera_pos[1], 0)
    glScalef(camera_pos[2], camera_pos[2], 1)
    glRotatef(camera_rot_y, 1, 0, 0)
    glRotatef(camera_rot_x, 0, 0, 1)

    # Sensors
    for pos in sensor_positions:
        draw_sphere(pos, radius=0.012, color=(1,1,0))
    
    # Grid lines
    for x1, y1, x2, y2 in grid_coordinates:
        draw_line(x1, y1, 0, x2, y2, 0)
    draw_line(0, 0, 0, 0, 0, 0.2)
    
    # Plane surface
    draw_box(vertices_precalc[0])
    
    # Stylus tip
    draw_sphere(stylus_pos, radius=0.016, color=(0,0.5,1))

    # Stylus orientation (arrow)
    stylus_dir = get_direction_from_pitch_yaw(pitch, yaw)
    draw_arrow(stylus_pos, stylus_dir, length=0.045, color=(1,0,0))

    
    
    pygame.display.flip()
    
    clock.tick(60)

 


