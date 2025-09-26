import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import math

PLATFORM_SIZE = 0.26416  # meters
sensor_positions = np.array([
    (0.0, 0.26416, 0.0),         # Sensor 0
    (0.0, 0.13208, 0.0),        # Sensor 1
    (0.0, 0.0, 0.0),            # Sensor 2
    (0.13208, 0.0, 0.0),        # Sensor 3
    (0.26416, 0.0, 0.0),        # Sensor 4
    (0.26416, 0.13208, 0.0),    # Sensor 5
    (0.26416, 0.26416, 0.0),    # Sensor 6
    (0.13208, 0.26416, 0.0),    # Sensor 7
])

camera_pos = [0.0, 0.0, -0.18]  # [X, Y, Z] position of camera
camera_rot_x = 30  # degrees, pitch
camera_rot_y = 0   # degrees, yaw
zoom_step = 0.02
rot_step = 5
pan_step = 0.02


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

def main():
    pygame.init()
    display = (800,600)
    pygame.display.set_mode(display, DOUBLEBUF|OPENGL)
    gluPerspective(45, (display[0]/display[1]), 0.01, 2.0)
    glEnable(GL_DEPTH_TEST)
    clock = pygame.time.Clock()

    stylus_pos = np.array([0.13, 0.13, 0.03])  # Replace with PSO tracking!
    pitch, yaw = 0.7, 1.2                      # Replace with PSO tracking!
    running = True

    # Camera state
    global camera_pos, camera_rot_x, camera_rot_y

    while running:
        # Handle events (key/mouse)
        for event in pygame.event.get():
            if event.type == QUIT:
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

        glClearColor(0.05,0.05,0.06,1)
        glClear(GL_COLOR_BUFFER_BIT|GL_DEPTH_BUFFER_BIT)

        # Camera transforms
        glLoadIdentity()
        gluPerspective(45, (display[0] / display[1]), 0.01, 2.0)
        glTranslatef(camera_pos[0], camera_pos[1], camera_pos[2])
        glRotatef(camera_rot_x, 1, 0, 0)
        glRotatef(camera_rot_y, 0, 1, 0)

        # Platform boundary
        glColor3f(0.6,0.6,0.65)
        glBegin(GL_LINE_LOOP)
        glVertex3f(0, 0, 0)
        glVertex3f(PLATFORM_SIZE, 0, 0)
        glVertex3f(PLATFORM_SIZE, PLATFORM_SIZE, 0)
        glVertex3f(0, PLATFORM_SIZE, 0)
        glEnd()

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

    pygame.quit()

if __name__ == '__main__':
    main()
