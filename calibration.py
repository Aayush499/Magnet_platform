import serial
import re

# Change this to your Arduino port
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

num_samples = 100  # Number of readings to average
x_vals, y_vals, z_vals = [], [], []

pattern = re.compile(r"X:\s*([-+]?\d*\.\d+|\d+)\s*Y:\s*([-+]?\d*\.\d+|\d+)\s*Z:\s*([-+]?\d*\.\d+|\d+)")

print(f"Collecting {num_samples} samples for averaging...")

while len(x_vals) < num_samples:
    line = ser.readline().decode('utf-8').strip()
    match = pattern.search(line)
    if match:
        x, y, z = map(float, match.groups())
        x_vals.append(x)
        y_vals.append(y)
        z_vals.append(z)
        print(f"Sample {len(x_vals)}: X={x}, Y={y}, Z={z}")

if x_vals:
    avg_x = sum(x_vals) / len(x_vals)
    avg_y = sum(y_vals) / len(y_vals)
    avg_z = sum(z_vals) / len(z_vals)
    print("\nAveraged Magnetic Field (uT):")
    print(f"X: {avg_x:.2f}")
    print(f"Y: {avg_y:.2f}")
    print(f"Z: {avg_z:.2f}")
    #make a text file and store the averages
    with open('calibration_results.txt', 'w') as f:
        f.write(f"Averaged Magnetic Field (uT):\n")
        f.write(f"X: {avg_x:.2f}\n")
        f.write(f"Y: {avg_y:.2f}\n")
        f.write(f"Z: {avg_z:.2f}\n")
        print("Calibration results saved to 'calibration_results.txt'.")
# Ensure the serial port is closed when done
ser.close()
 