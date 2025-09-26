import pygame
import serial
import sys
import time
import math

SERIAL_PORT = '/dev/ttyACM0'  # adjust as needed
BAUD_RATE = 19200
SENSOR_COUNT = 8
CALIBRATION_SECONDS = 10

layout = [
    (0,0),    # 0: NW
    (0,0.5),  # 1: W
    (0,1),    # 2: SW
    (0.5,1),  # 3: S
    (1,1),    # 4: SE
    (1,0.5),  # 5: E
    (1,0),    # 6: NE
    (0.5,0),  # 7: N
]

# Prepare
pygame.init()
width, height = 500, 500
margin = 70
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Calibrated Magnetometer Vector Field")
font = pygame.font.SysFont("Consolas", 16)
clock = pygame.time.Clock()

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
except Exception as e:
    print("Serial error:", e)
    sys.exit()

sensor_vals = {i: (0.0, 0.0, 0.0) for i in range(SENSOR_COUNT)}

# Calibration state
calibrating = True
cal_start = time.time()
cal_buffer = {i: [] for i in range(SENSOR_COUNT)}
offsets = {i: (0.0, 0.0) for i in range(SENSOR_COUNT)}

def draw_arrow(start, vector, color=(0, 200, 255), scale=2.0):
    # Draw vector arrow
    end = (start[0] + vector[0]*scale, start[1] + vector[1]*scale)
    pygame.draw.line(screen, color, start, end, 3)
    # Arrowhead
    angle = math.atan2(vector[1], vector[0])
    arrow_len = 15
    left = (end[0] - arrow_len * math.cos(angle - math.pi/7),
            end[1] - arrow_len * math.sin(angle - math.pi/7))
    right = (end[0] - arrow_len * math.cos(angle + math.pi/7),
             end[1] - arrow_len * math.sin(angle + math.pi/7))
    pygame.draw.polygon(screen, color, [end, left, right])

while True:
    # ---- Handle quitting
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            ser.close()
            sys.exit()

    # ---- Read serial, process raw lines
    line = ser.readline().decode("utf-8").strip()
    if not line:
        continue

    if line == "BREAK":
        # Update display each frame
        screen.fill((15,15,40))
        for i, pos in enumerate(layout):
            px = margin + pos[0]*(width-2*margin)
            py = margin + pos[1]*(height-2*margin)
            cal_x, cal_y = offsets[i]
            x, y, z = sensor_vals[i]
            
            draw_vec = (x-cal_x, -(y-cal_y)) if not calibrating else (0,0)  # Only draw after calibration
            print(f"Sensor {i}: Raw({x}, {y}, {z}) Calibrated({x-cal_x}, {-(y-cal_y)})")

            # Draw arrow (or calibration point)
            color = (80,180,250) if not calibrating else (200,200,200)
            draw_arrow((px, py), draw_vec, color=color)
            pygame.draw.circle(screen, (255,255,80), (int(px), int(py)), 7)
            label = font.render(f"{i}", True, (255,255,255))
            screen.blit(label, (px+12, py-12))
        # Progress bar
        if calibrating:
            elapsed = time.time() - cal_start
            pct = min(elapsed / CALIBRATION_SECONDS, 1.0)
            bar_w = int(width*pct)
            pygame.draw.rect(screen, (30,180,60), (margin, height-40, bar_w, 24))
            msg = font.render(
                f"Calibrating... {CALIBRATION_SECONDS-int(elapsed)}s left", True, (200,255,200))
            screen.blit(msg, (margin+10, height-38))

        pygame.display.flip()
        clock.tick(20)
        # Calibration finish
        if calibrating and (time.time() - cal_start) > CALIBRATION_SECONDS:
            for idx in range(SENSOR_COUNT):
                buf = cal_buffer[idx]
                if buf:
                    avg_x = sum(x for x, _ in buf)/len(buf)
                    avg_y = sum(y for _, y in buf)/len(buf)
                    offsets[idx] = (avg_x, avg_y)
            calibrating = False
        continue

    parts = line.split(",")
    if len(parts) == 4:
        try:
            idx = int(parts[0])
            x = float(parts[1])
            y = float(parts[2])
            z = float(parts[3])
            sensor_vals[idx] = (x, y, z)
            
            if calibrating:
                cal_buffer[idx].append((x, y))
        except:
            pass
