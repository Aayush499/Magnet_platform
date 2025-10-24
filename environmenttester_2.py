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
# import pygame
# from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
from OpenGL.GLUT import *

# Arguments for running
parser = argparse.ArgumentParser(prog='Stylus Program')
parser.add_argument('--offline',  help='Run in offline mode', action='store_true')
args = parser.parse_args()

#--- Visualization parameters
WIN_SIZE = 600
PLATFORM_SIZE = 0.26416  # meters (length of one platform side)
MARGIN = 60
NUM_GRID_LINES = 5 # Number of grid lines, includes left and right
CYLINDER_SECTOR_COUNT = 20 # Number of sectors on cylinder, how good the circle looks

SERIAL_PORT = 'COM3'  # adjust as needed
BAUD_RATE = 115200
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10
MAGNET_STRENGTH1 = 1.1624647114675248 
MAGNET_STRENGTH_RATIO = .5

sensor_mask = [True, True, True, True, True, True, True, False]
WORKING_SENSOR_COUNT = sum(sensor_mask)


def calibrate_ambient_field(ser, n_frames=10):
    """Calibrate by measuring average field at each sensor with no stylus present."""
    frames = []
    print("Calibrating ambient magnetic field...")
    for _ in range(n_frames):
        frame = read_sensor_frame(ser)  # your existing function
        #check if frame[working_indices] is not None
        if frame is not None and all(any(frame[i][j] is not None for j in range(3)) if sensor_mask[i] else True for i in range(len(sensor_mask))):
            frames.append(frame)
    #for all indices i false in working_indices, set frame[i] to (0,0,0)

    frames = np.array(frames)  # shape (n_frames, sensor_count, 3)
    for i in range(SENSOR_COUNT):
        if not sensor_mask[i]:
            frames[:, i] = (0, 0, 0)
    ambient_field = np.mean(frames, axis=0)  # shape (sensor_count, 3)
    #change it back to None for broken sensors
    for i in range(SENSOR_COUNT):
        if not sensor_mask[i]:
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
sensor_positions = full_sensor_positions[np.array(sensor_mask)]
# Circumvention for offline testing for now
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except:
    ser = None

def read_sensor_frame(ser):
    """Reads next complete set of magnetometer readings as (N,3) array."""
    values = [(None, None, None)] * SENSOR_COUNT # only for working sensors
    if not ser:
        # Dummy values
        return np.array([
                [2.704e-05, 3.464e-05, -1.243e-05],
                [3.112e-05, 1.257e-05, 1.011e-05],
                [-1.071e-05, -7.53e-06, 3.497e-05],
                [5.142e-05, -1.6559999999999997e-05, 6.292e-05],
                [4.4e-07, -1.0039999999999999e-05, 2.5989999999999997e-05],
                [1.23e-06, 3.471e-05, -1.448e-05],
                [2.211e-05, -8e-08, 1.997e-05],
                [None, None, None] # Broken
        ])
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


# Display Handling

def init_visualization():
    display = (800,600)
    glutInit()
    glutInitDisplayMode(GLUT_RGBA | GLUT_DEPTH )
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

global rot_aayush
rot_aayush = 0
def display_screen():
    global rot_aayush
    glClearColor(0.9,0.9,0.9,1)
    glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)
    
    glLoadIdentity()
    # Z axis of camera position controls zoom, this is controlled by glScale
    glTranslatef(camera_pos[0], camera_pos[1], 0)
    glScalef(camera_pos[2], camera_pos[3], 1)
    glRotatef(rot_aayush, 1, 0, 0)
    glRotatef(camera_rot_x, 0, 0, 1)
    rot_aayush-=5

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
        if True:
            camera_rot_x -= mouse_pos_curr_x - mouse_pos_prev_x
            # camera_rot_y -= mouse_pos_curr_y - mouse_pos_prev_y
            camera_rot_y +=5
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
    glRotatef(direction[1] * 180 / math.pi, 0, 0, 1)
    glRotatef(180 + direction[0] * 180 / math.pi, 0, 1, 0)
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
    glRotatef(direction[1] * 180 / math.pi, 0, 0, 1)
    glRotatef(direction[0] * 180 / math.pi, 0, 1, 0)
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
    return end

def get_direction_from_pitch_yaw(pitch, yaw):
    dx = math.sin(pitch) * math.cos(yaw)
    dy = math.sin(pitch) * math.sin(yaw)
    dz = math.cos(pitch)
    return np.array([dx, dy, dz])

init_visualization()

ambient_field = calibrate_ambient_field(ser)
ambient_field = ambient_field[np.array(sensor_mask)]  # filter by mask

best_pos = None
best_cost = 0
alpha = 0.6  # Between 0 and 1
zoom_step = 1.2
rot_step = 5
pan_step = 0.002
fov = 45 
running = True
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

while running:
    
    measured_B_raw = read_sensor_frame(ser)  # updated with each "BREAK"
    measured_B_raw = measured_B_raw[np.array(sensor_mask)]  # filter by mask
    measured_B = measured_B_raw - ambient_field  # shape (7,3)
    # print(f"Measured B: {measured_B_raw}")

    # bounds = (
    #     np.array([0, 0, 0, -np.pi/2, -np.pi]),   # min: x,y,z,pitch,yaw
    #     np.array([0.3, 0.3, 0.2, np.pi/2, np.pi]) # max: x,y,z,pitch,yaw
    # )
    if best_pos is None or best_cost < pso_ext_options['best_cost_threshold']:
        bounds = default_bounds
    else:
        center = np.array(best_pos)
        width = np.array([0.03, 0.03, 0.02, np.pi/12, np.pi/12])  # Reasonable max movement per frame
        bounds = (np.maximum(center - width, bounds[0]), np.minimum(center + width, bounds[1]))
        filtered_pos = best_pos

    # Create a PSO optimizer for 5 parameters
    optimizer = ps.single.GlobalBestPSO(
        n_particles=40,
        dimensions=5,
        options=options,
        bounds=bounds,
    )

    # TODO: Find typical bound change and try to form an equation to determine necessary n_particles or iters
    start_time = time.time_ns()
    if not args.offline:
        best_cost, best_pos = optimizer.optimize(cost_pso, iters=100, verbose=False)
    else:
        best_pos = debug_vertices[debug_iter]
        debug_iter = (debug_iter + 1) % len(debug_vertices)
    # print((time.time_ns() - start_time) / 1000000)
    # print(bounds)
    stylus_pos = best_pos[:3]
    pitch, yaw = best_pos[3], best_pos[4]
    
    # (C) VISUALIZATION:
    glutMainLoopEvent()
    glutPostRedisplay()
     


    



    # print(f"Best Cost: {best_cost}")
    # print(f"Estimated Position: {best_pos[:3]}, Pitch: {best_pos[3]}, Yaw: {best_pos[4]}")


