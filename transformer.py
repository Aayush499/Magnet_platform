import pyswarms as ps
import pygame
import serial
import sys
import time
import math
import csv
import math
import argparse
from enum import Enum

import numpy as np
import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math
import torch
import torch.nn as nn
from scipy.spatial.transform import Rotation as R
from collections import deque

from OpenGL.GLUT import *

# Arguments for running
parser = argparse.ArgumentParser(prog='Stylus Program')
parser.add_argument('--offline', help='Run in offline mode', action='store_true')
args = parser.parse_args()

#--- Visualization parameters
WIN_SIZE = 600
PLATFORM_SIZE = 0.26416  # meters (length of one platform side)
MARGIN = 60
NUM_GRID_LINES = 5 # Number of grid lines, includes left and right
CYLINDER_SECTOR_COUNT = 20 # Number of sectors on cylinder, how good the circle looksMoving_average_window_size = 5  # Number of frames to average for smoothing
Moving_average_window_size = 5  # Number of frames to average for smoothing

SERIAL_PORT = '/dev/ttyACM0' 
BAUD_RATE = 115200
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10
# MAGNET_STRENGTH1 = 1.4
MAGNET_STRENGTH1 = 0.058477513103711194
MAGNET_STRENGTH_RATIO = .5

sensor_mask = [True, True, True, True, True, True, True, False]

WORKING_SENSOR_COUNT = sum(sensor_mask)
TOTAL_SENSOR_COUNT = len(sensor_mask)
surface_points = {
    4: (-.13, .1325, 0),
    5: (0, .1325, 0), 
    8: (.13, .1325, 0),
    9: (-.13, 0, 0),
    12: (0, 0, 0),
    13: (.13, 0, 0),
    16: (-.13, -.1325, 0),
    17: (0, -.1325, 0),
    20: (.13, -.1325, 0)
}

sensor_number_to_aruco_id_map = {
    0: 16,
    1: 17,
    2: 20,
    3: 13,
    4: 8,
    5: 5,
    6: 4,
    7: 9,  # broken sensor
}

#get full sensor positions from aruco ids positions, subtract 2.4 csm from the z axis of each aruco id position
full_sensor_positions = np.array([surface_points[sensor_number_to_aruco_id_map[i]] for i in range(TOTAL_SENSOR_COUNT)])
full_sensor_positions[:, 2] -= 0.024  # subtract 2.4 cm from z axis to get sensor positions

new_sensor_positions = full_sensor_positions[sensor_mask]


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
 
sensor_positions = full_sensor_positions[np.array(sensor_mask)]
# Circumvention for offline testing for now
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except:
    ser = None
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

# measured_B = np.array(...)  # shape (7,3), filled from your Arduino/frame


import torch
import torch.nn as nn

class SensorViTRegressor(nn.Module):
    def __init__(self, 
                 n_tokens=7,        # number of sensors (tokens)
                 in_dim=7,          # features per sensor
                 d_model=128,       # transformer model dim
                 nhead=8,           # attention heads
                 num_layers=4,      # transformer layers
                 out_dim=7,         # regression output [pos, quat]
                 patch_dropout=0.1,
                 mlp_dim=256,
                 use_positional_encoding=True):
        super().__init__()
        
        # Patch embedding: Linear projection from in_dim to d_model
        self.patch_embed = nn.Linear(in_dim, d_model)
        
        # Optional positional embedding
        if use_positional_encoding:
            self.pos_embed = nn.Parameter(torch.zeros(1, n_tokens, d_model))
        else:
            self.pos_embed = None
        
        # Transformer encoder stack
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=mlp_dim,
            dropout=patch_dropout,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Flatten all tokens for regression output
        self.pre_head = nn.Linear(n_tokens * d_model, mlp_dim)
        self.act = nn.ReLU()
        self.head = nn.Linear(mlp_dim, out_dim)
        
    def forward(self, x):
        # x: (batch_size, n_tokens, in_dim)
        x = self.patch_embed(x)  # (batch_size, n_tokens, d_model)
        if self.pos_embed is not None:
            x = x + self.pos_embed  # add positional encoding
        x = self.encoder(x)  # (batch_size, n_tokens, d_model)
        x = x.reshape(x.shape[0], -1)  # flatten all tokens
        x = self.pre_head(x)
        x = self.act(x)
        out = self.head(x)
        return out  # (batch_size, out_dim)




class MagnetTransformer(nn.Module):
    def __init__(self, d_model=64, nhead=4, num_layers=2, n_tokens=7, in_dim=7, out_dim=7):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, d_model)
        encoder_layer = nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.agg = nn.Linear(n_tokens * d_model, d_model)
        self.head = nn.Linear(d_model, out_dim)
    def forward(self, x):
        x = self.input_proj(x)
        x = x.permute(1, 0, 2)
        x = self.encoder(x)
        x = x.permute(1, 0, 2)
        x = x.reshape(x.shape[0], -1)
        x = self.agg(x)
        out = self.head(x)
        return out

# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# model = MagnetTransformer()
# model.load_state_dict(torch.load('magnet_transformer_model.pth', map_location=device))
# model.to(device)
# model.eval()

# Setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = SensorViTRegressor().to(device)

# Dummy input (for 7 sensors, each with 7 features)
batch_size = 10
input_tensor = torch.randn(batch_size, 7, 7).to(device)
output = model(input_tensor)  # shape (batch_size, 7)




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
                #swap 0 and 1 columns
                matrix = [[row[1], row[0], row[2]] for row in matrix]
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


# Display Handling

def init_visualization():
    display = (800,600)
    glutInit()
    glutInitDisplayMode(GLUT_RGBA |  GLUT_DEPTH)
    glutInitWindowSize(*display)
    glutCreateWindow(b'Magnet Stylus Platform')
    glutDisplayFunc(display_screen)
    glutReshapeFunc(glut_reshape_screen)
    glutMotionFunc(glut_mouse_motion)
    glutMouseFunc(glut_mouse_press)
    glutMouseWheelFunc(glut_mouse_scroll)
    glutKeyboardFunc(glut_key_down)
    glutKeyboardUpFunc(glut_key_up)
    glutSetKeyRepeat(GLUT_KEY_REPEAT_OFF)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(10, (display[0]/display[1]), 0.01, 2.0)
    glEnable(GL_DEPTH_TEST)
    global clock
    clock = pygame.time.Clock()
    # camera state if needed
    global camera_pos, camera_rot_x, camera_rot_y
    camera_pos = [0.0, 0.0, 2.0, 2.0 * display[0] / display[1]]
    camera_rot_x, camera_rot_y = 0, 180
    global mouse_drag, mouse_pan, activate_paint
    mouse_drag, mouse_pan, activate_paint = False, False, False
    global paint_color
    paint_color = (1.0, 0.0, 0.0)
    
    # Geometry precalculation
    calculate_box_vertices('plane_surface', (0, 0, -0.01), (0.3, 0.3, 0.02))
    calculate_cylinder_vertices('pencil_body', 0.01, 0.01, 0.1)
    calculate_cone_vertices('pencil_tip', 0.01, 0.045)

def display_screen():
    glClearColor(0.9,0.9,0.9,1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    glLoadIdentity()
    # Z axis of camera position controls zoom, this is controlled by glScale
    glTranslatef(camera_pos[0], camera_pos[1], 0)
    glScalef(camera_pos[2], camera_pos[3], 1)
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
    draw_box(geometry_vertices_precalc['plane_surface'])
    
    # Stylus body
    draw_cylinder(stylus_pos, geometry_vertices_precalc['pencil_body'], (pitch, yaw), color=(1.0, 0.7, 0.0))
    draw_cone(stylus_pos, geometry_vertices_precalc['pencil_tip'], (pitch, yaw), color=(1.0, 0.95, 0.85))

    # Stylus orientation (arrow)
    stylus_dir = get_direction_from_pitch_yaw(pitch, yaw)
    paint_vertex = draw_arrow(stylus_pos, stylus_dir, length=0.045, color=(0,0,0))

    # Line painting management
    for paint_line in paint_vertices_precalc:
        draw_line_multiple(paint_line['vertices'], paint_line['color'])
    if activate_paint:
        paint_vertices_precalc[-1]['vertices'].append(paint_vertex)

    glutSwapBuffers()


# Input Handling

def glut_reshape_screen(w, h):
    glViewport(0, 0, w, h)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(10, (w/h), 0.01, 2.0)
    camera_pos[3] = camera_pos[2] * w / h

def glut_mouse_press(btn, state, x, y):
    global mouse_drag, mouse_pan, mouse_pos_prev_x, mouse_pos_prev_y
    if state == 0:
        match btn:
            case 0:
                mouse_drag = True
                mouse_pos_prev_x, mouse_pos_prev_y = x, y
            case 1:
                mouse_pan = True
                mouse_pos_prev_x, mouse_pos_prev_y = x, y
    elif state == 1:
        match btn:
            case 0:
                mouse_drag = False
            case 1:
                mouse_pan = False

def glut_mouse_motion(x, y):
    global mouse_drag, mouse_pan, mouse_pos_prev_x, mouse_pos_prev_y, camera_pos, camera_rot_x, camera_rot_y
    if mouse_drag or mouse_pan:
        mouse_pos_curr_x, mouse_pos_curr_y = x, y
        if mouse_drag:
            camera_rot_x -= mouse_pos_curr_x - mouse_pos_prev_x
            camera_rot_y -= mouse_pos_curr_y - mouse_pos_prev_y
        else:
            camera_pos[0] += (mouse_pos_curr_x - mouse_pos_prev_x) * pan_step
            camera_pos[1] -= (mouse_pos_curr_y - mouse_pos_prev_y) * pan_step
        mouse_pos_prev_x, mouse_pos_prev_y = x, y

def glut_mouse_scroll(wheel, dir, x, y):
    match dir:
        case 1:
            camera_pos[2] *= zoom_step
            camera_pos[3] *= zoom_step
        case -1:
            camera_pos[2] /= zoom_step
            camera_pos[3] /= zoom_step

def glut_key_down(key, x, y):
    global activate_paint, paint_color
    if key == b' ':
        activate_paint = True
        paint_vertices_precalc.append({
            'color': paint_color,
            'vertices': []
        })
    elif key == b'r':
        paint_color = (1.0, 0.0, 0.0)
    elif key == b'g':
        paint_color = (0.0, 1.0, 0.0)
    elif key == b'b':
        paint_color = (0.0, 0.0, 1.0)

def glut_key_up(key, x, y):
    if key == b' ':
        global activate_paint
        activate_paint = False


# Draw Handling

# box_edges = ((0,1),(2,3),(4,5),(6,7),(0,2),(0,4),(2,6),(4,6),(1,3),(1,5),(3,7),(5,7))
box_surfaces = ((1,3,7,5),(0,1,3,2),(2,3,7,6),(6,7,5,4),(4,5,1,0),(0,2,6,4))
box_colors = ((0.5,0.5,0.5),(1,1,1),(0.5,0.5,0.5),(1,1,1))
cylinder_surfaces = np.concat((
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT),
        np.arange(CYLINDER_SECTOR_COUNT, CYLINDER_SECTOR_COUNT * 2),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT, CYLINDER_SECTOR_COUNT * 2), 1)
    )),
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT), 1),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT, CYLINDER_SECTOR_COUNT * 2), 1)
    )),
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT), 1),
        np.full(CYLINDER_SECTOR_COUNT, -2)
    )),
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT, CYLINDER_SECTOR_COUNT * 2),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT, CYLINDER_SECTOR_COUNT * 2), 1),
        np.full(CYLINDER_SECTOR_COUNT, -1)
    ))
))
cone_surfaces = np.concat((
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT), 1),
        np.full(CYLINDER_SECTOR_COUNT, -2)
    )),
    np.column_stack((
        np.arange(CYLINDER_SECTOR_COUNT),
        np.roll(np.arange(CYLINDER_SECTOR_COUNT), 1),
        np.full(CYLINDER_SECTOR_COUNT, -1)
    ))
))
geometry_vertices_precalc = {}
paint_vertices_precalc = []

def calculate_box_vertices(name, pos, size, isMiddle=True):
    min_coord = [pos[i] - (size[i] / 2) for i in range(3)] if isMiddle else pos
    max_coord = [pos[i] + (size[i] / 2) for i in range(3)] if isMiddle else [pos[i] + size[i] for i in range(3)]
    box_coord = tuple(zip(min_coord, max_coord))
    vertices = []
    for x in box_coord[0]:
        for y in box_coord[1]:
            for z in box_coord[2]:
                vertices.append((x, y, z))
    geometry_vertices_precalc[name] = vertices

def calculate_circle_vertices(pos, r, sectors=CYLINDER_SECTOR_COUNT):
    angles = np.linspace(0, 2 * math.pi, num=sectors + 1)[:-1]
    circle_vertices = np.transpose((np.cos(angles) * r + pos[0], np.sin(angles) * r + pos[1], np.full(sectors, pos[2])))
    return circle_vertices

def calculate_cylinder_vertices(name, rb, rt, h):
    geometry_vertices_precalc[name] = np.concat((
        calculate_circle_vertices((0, 0, 0), rb),
        calculate_circle_vertices((0, 0, h), rt),
        np.array(((0, 0, 0), (0, 0, h)))
    ))

def calculate_cone_vertices(name, r, h=0.04):
    geometry_vertices_precalc[name] = np.concat((
        calculate_circle_vertices((0, 0, 0), r),
        np.array(((0, 0, 0), (0, 0, h)))
    ))

def draw_box(vertices, color=(0.5, 0.5, 0.5)):
    glBegin(GL_QUADS)
    for surface in box_surfaces:
        c = 0
        for v in surface:
            glColor3fv(box_colors[c])
            glVertex3fv(vertices[v])
            c += 1
    glEnd()

def draw_line(x1, y1, z1, x2, y2, z2, color=(0, 0, 0)):
    glColor3f(*color)
    glBegin(GL_LINES)
    glVertex3f(x1, y1, z1)
    glVertex3f(x2, y2, z2)
    glEnd()

def draw_line_multiple(vertices, color=(0.0, 0.0, 1.0)):
    glColor3f(*color)
    glLineWidth(3)
    glBegin(GL_LINE_STRIP)
    for vertex in vertices:
        glVertex3f(*vertex)
    glEnd()
    glLineWidth(1)

def draw_cylinder(pos, vertices, direction, color=(0.5, 0.5, 0.5)):
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    # Yaw first then pitch, else won't be rotated correctly
    glRotatef(180 + direction[1] * 180 / math.pi, 0, 0, 1)
    glRotatef( direction[0] * 180 / math.pi, 0, 1, 0)
    glBegin(GL_TRIANGLES)
    for surface in cylinder_surfaces:
        for v in surface:
            glColor3f(*color)
            glVertex3fv(vertices[v])
    glEnd()
    glPopMatrix()

def draw_cone(pos, vertices, direction, color=(0.5, 0.5, 0.5)):
    glPushMatrix()
    glTranslatef(float(pos[0]), float(pos[1]), float(pos[2]))
    # Yaw first then pitch, else won't be rotated correctly
    glRotatef( direction[1] * 180 / math.pi, 0, 0, 1)
    glRotatef(  direction[0] * 180 / math.pi, 0, 1, 0)
    glBegin(GL_TRIANGLES)
    for surface in cone_surfaces:
        for v in surface:
            glColor3f(*color)
            glVertex3fv(vertices[v])
    glEnd()
    glPopMatrix()

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
    end = start - direction * length
    
    glBegin(GL_LINES)
    glVertex3fv(start)
    glVertex3fv(end)
    glEnd()
    # Simple arrowhead (optional)
    arrow_dir = -direction
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
    return end

def get_direction_from_pitch_yaw(pitch, yaw):
    dx = math.sin(pitch) * math.cos(yaw)
    dy = math.sin(pitch) * math.sin(yaw)
    dz = math.cos(pitch)
    return np.array([dx, dy, dz])

import numpy as np

class SimpleKalman:
    def __init__(self, dim=3, process_var=1e-4, meas_var=1e-2):
        self.x = np.zeros(dim)  # initial state
        self.P = np.eye(dim)    # initial covariance
        self.F = np.eye(dim)    # state transition
        self.H = np.eye(dim)    # measurement function
        self.Q = process_var * np.eye(dim)  # process noise
        self.R = meas_var * np.eye(dim)     # measurement noise

    def update(self, z):
        # Predict
        x_pred = self.F @ self.x
        P_pred = self.F @ self.P @ self.F.T + self.Q
        # Update
        y = z - self.H @ x_pred
        S = self.H @ P_pred @ self.H.T + self.R
        K = P_pred @ self.H.T @ np.linalg.inv(S)
        self.x = x_pred + K @ y
        self.P = (np.eye(len(self.x)) - K @ self.H) @ P_pred
        return self.x
init_visualization()

ambient_field = calibrate_ambient_field(ser)
ambient_field = ambient_field[np.array(sensor_mask)]  # filter by mask
sensor_positions = full_sensor_positions[np.array(sensor_mask)]
 
best_cost = 0
alpha = 0.6  # Between 0 and 1
zoom_step = 1.2
rot_step = 5
pan_step = 0.002
fov = 45 
running = True
kf = SimpleKalman(dim=3)
moving_average_window = []
pos_window = deque(maxlen=Moving_average_window_size)
pitch_window = deque(maxlen=Moving_average_window_size)
yaw_window = deque(maxlen=Moving_average_window_size)

if args.offline:
    debug_iter = 0
    debug_sectors = 60
    debug_vbh = [np.roll(np.c_[calculate_circle_vertices((0, 0, 0.005 * i), 0.13208, debug_sectors), [[math.pi/4, math.pi/4]] * debug_sectors], -i, axis=0) for i in range(debug_sectors)]
    debug_vertices = np.empty((debug_sectors * debug_sectors, 5))
    for i in range(len(debug_vbh)):
        debug_vertices[i::len(debug_vbh)] = debug_vbh[i]

# PSO options
options = {
    'c1': 0.5,
    'c2': 0.3,
    'w': 0.9
}
pso_ext_options = {
    'best_cost_threshold': 1e-5 # PSO will return to default bounds if best cost is below this threshold
}

import joblib
# Load scalers
scaler_pos = joblib.load('scaler_pos.joblib')
scaler_readings = joblib.load('scaler_readings.joblib')
scaler_strength = joblib.load('scaler_strength.joblib')
scaler_y = joblib.load('scaler_y.joblib')

while running:
    measured_B_raw = parse_output_matrix()
    measured_B_raw = measured_B_raw[np.array(sensor_mask)]
    measured_B = measured_B_raw - ambient_field  # shape (7,3)

    # Normalize each token component separately
    sensor_tokens_norm = []
    for i in range(WORKING_SENSOR_COUNT):
        readings = measured_B[i]
        pos = new_sensor_positions[i]
        
        pos_norm = scaler_pos.transform(pos.reshape(1, -1)).flatten()
        read_norm = scaler_readings.transform(readings.reshape(1, -1)).flatten()
        str_norm = scaler_strength.transform([[MAGNET_STRENGTH1]]).flatten()
        
        token = np.concatenate([pos_norm, read_norm, str_norm])
        sensor_tokens_norm.append(token)
    
    # Convert directly to tensor (already normalized!)
    feature_tensor = torch.tensor([sensor_tokens_norm], dtype=torch.float32).to(device)  # shape (1, 7, 7)

    # Model prediction [x, y, z, qx, qy, qz, qw]
    with torch.no_grad():
        out = model(feature_tensor)  # shape (1, 7)
    out_np = out.cpu().numpy().flatten()
    out_phys_units = scaler_y.inverse_transform(out_np.reshape(1, -1)).flatten()

    stylus_pos = out_phys_units[:3]  # Now in meters!
    stylus_pos[2] *= -1  # Flip z if needed
    qx, qy, qz, qw = out_phys_units[3:]
    
    quat = [qx, qy, qz, qw]
    rot = R.from_quat(quat)
    rx, ry, rz = rot.as_euler('xyz', degrees=False)
    pitch, yaw = rx, ry
    
    pos_window.append(stylus_pos)
    pitch_window.append(pitch)
    yaw_window.append(yaw)

    smoothed_pos = np.mean(pos_window, axis=0)
    print(f"Position: {smoothed_pos[0]*100:.2f} cm, {smoothed_pos[1]*100:.2f} cm, {smoothed_pos[2]*100:.2f} cm | Pitch: {pitch*180/math.pi:.2f} deg | Yaw: {yaw*180/math.pi:.2f} deg")
    smoothed_pitch = np.mean(pitch_window)
    smoothed_yaw = np.mean(yaw_window)

    kf_pos = kf.update(smoothed_pos)
    pitch, yaw = smoothed_pitch, smoothed_yaw

    # Visualization:
    glutMainLoopEvent()
    glutPostRedisplay()
    clock.tick(60)


 


