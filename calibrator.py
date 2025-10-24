import serial

# Adjust these for your setup
SERIAL_PORT = '/dev/ttyACM0'  # For Linux/Mac, usually /dev/ttyACM0 or /dev/ttyUSB0
# SERIAL_PORT = 'COM3'         # For Windows
BAUD_RATE = 115200            # Set to match your Arduino sketch
TOTAL_SENSOR_COUNT= 8

try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except Exception as e:
    print(f"Could not connect to {SERIAL_PORT}: {e}")
    exit(1)



def read_line(ser):
    """Read a line from the serial port."""
    try:
        line = ser.readline().decode(errors='replace').strip()
        return line
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
                matrix = [values[i:i+3] for i in range(0, len(values), 3)]
                #divide each element by 1e6
                matrix = [[val / 1e6 for val in row] for row in matrix]
                return matrix
            else:
                print(f"Unexpected number of values: {len(values)}")
        except ValueError as ve:
            print(f"Value error: {ve}")
    return None
    


DIST = 4 #inches
#stop and wait for user input

input("Press Enter to continue initiate ambient data collection...")
ambient_values = []
for i in range(100):
    matrix = parse_output_matrix()
    #wait for user input
    
    
    if matrix:
        # print("Received matrix:")
        # for row in matrix:
        #     print(row)
        print(len(ambient_values),": ",matrix[4][2])
        ambient_values.append(matrix[1][2])
    else:
        print("No valid matrix received.")
avg_ambient = sum(ambient_values) / len(ambient_values)
print(f"Average ambient: {avg_ambient}")

input("Press Enter to continue initiate data collection...")
cnt =0
data =[]
while True:
    matrix = parse_output_matrix()
    #wait for user input
    
    
    if matrix:
        # print("Received matrix:")
        # for row in matrix:
        #     print(row)

        print(len(data),": ",matrix[1][2])
        data.append(matrix[1][2])
    else:
        print("No valid matrix received.")
    #find average of the data
    if len(data) > 100:
        avg = sum(data) / len(data) - avg_ambient
        print(f"Average: {avg}")
        #calculate the distance in cm
    
    
        m = ((avg) * (DIST*2.54/100)**3)/(2*1e-6)
        print(f"Calculated m: {m}")
        break
    cnt +=1


# Calculated m: 8.967598442103559e-08
 # Calculated m: -0.15595511887557192


# Calculated m: 0.055712643362743446
# Calculated m: 0.05965506781879959
# Calculated m: 0.05965032397815102
# Calculated m: 0.058477513103711194
