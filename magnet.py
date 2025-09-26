import serial

# Change 'COM3' to your Arduino port (e.g., 'COM4', '/dev/ttyACM0', or '/dev/ttyUSB0')
ser = serial.Serial('/dev/ttyACM0', 115200, timeout=1)

while True:
    line = ser.readline().decode('utf-8').strip()
    if line:
        print(line)
