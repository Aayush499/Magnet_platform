import pygame
import serial
import sys

# CONFIG
serial_port = '/dev/ttyACM0'  # Change this to match your Arduino's port
baud_rate = 115200
sensor_count = 7              # Number of sensors we're reading (channels 0–6)
bar_scale = 0.2               # Pixels per microtesla
max_bar_height = 300

# Setup serial
ser = serial.Serial(serial_port, baud_rate, timeout=1)

# Setup Pygame
pygame.init()
font = pygame.font.SysFont("Consolas", 16)
width, height = 180 * sensor_count, 400
screen = pygame.display.set_mode((width, height))
pygame.display.set_caption("Magnetometer Array Visualization")

clock = pygame.time.Clock()

# Store sensor data
sensor_values = {i: (0.0, 0.0, 0.0) for i in range(sensor_count)}  # channel: (x, y, z)

def draw_sensor_bars():
    screen.fill((30, 30, 30))
    for i in range(sensor_count):
        x, y, z = sensor_values[i]
        base_x = 40 + i * 180
        base_y = height // 2

        # Draw center line
        pygame.draw.line(screen, (90, 90, 90), (base_x - 30, base_y), (base_x + 30, base_y), 1)

        # Draw bars for X, Y, Z
        for idx, val in enumerate((x, y, z)):
            color = [(255, 0, 0), (0, 255, 0), (0, 150, 255)][idx]
            bar_length = int(val * bar_scale)
            bar_x = base_x + (idx - 1) * 20  # offset -20, 0, +20

            if val >= 0:
                pygame.draw.rect(screen, color, (bar_x, base_y - bar_length, 12, bar_length))
            else:
                pygame.draw.rect(screen, color, (bar_x, base_y, 12, -bar_length))

        # Draw label
        label = font.render(f"S{i}", True, (255, 255, 255))
        screen.blit(label, (base_x - 10, base_y + 110))

        # Draw values
        for j, val in enumerate((x, y, z)):
            val_label = font.render(f"{['X', 'Y', 'Z'][j]}: {val:.1f}", True, (200, 200, 200))
            screen.blit(val_label, (base_x - 30, base_y + 130 + j * 18))

while True:
    try:
        line = ser.readline().decode('utf-8').strip()
    except Exception:
        continue

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            ser.close()
            sys.exit()

    if not line:
        continue

    if line == "BREAK":
        draw_sensor_bars()
        pygame.display.flip()
        clock.tick(30)
        continue

    # Parse sensor line
    # parts = line.split(",")
    if len(parts) != 4:
        continue  # skip malformed

    try:
        ch = int(parts[0])
        x = float(parts[1])
        y = float(parts[2])
        z = float(parts[3])
        if ch in sensor_values:
            sensor_values[ch] = (x, y, z)
    except ValueError:
        continue  # parsing error; skip
