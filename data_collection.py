import serial

import cv2
import numpy as np
import pickle


# Adjust these for your setup
SERIAL_PORT = '/dev/ttyACM0'  # For Linux/Mac, usually /dev/ttyACM0 or /dev/ttyUSB0
# SERIAL_PORT = 'COM3'         # For Windows
BAUD_RATE = 115200            # Set to match your Arduino sketch
TOTAL_SENSOR_COUNT= 8

max_rows = 30000
file = '_data_collected/magnet_data.csv'
try:
    ser = serial.Serial(SERIAL_PORT, BAUD_RATE, timeout=1)
    print(f"Connected to {SERIAL_PORT} at {BAUD_RATE} baud.")
except Exception as e:
    print(f"Could not connect to {SERIAL_PORT}: {e}")
    exit(1)


# sensor_mask = [True, True, True, True, True, True, True, False]
# WORKING_SENSOR_COUNT = sum(sensor_mask)
# TOTAL_SENSOR_COUNT = len(sensor_mask)
    
    
# --- Load calibration ---
with open('output/calibration_data.pkl', 'rb') as f:
    calib_data = pickle.load(f)
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['distortion_coefficients']

#create a csv file in folder _data_collected if it doesn't exist
#the csv will collect x y z and orientation of the magnet and the magnetic field readings from each sensor
import os
if not os.path.exists('_data_collected'):
    os.makedirs('_data_collected')
import csv
csv_file = open(file, 'a', newline='')
csv_writer = csv.writer(csv_file)
#write header if file is empty
if os.stat(file).st_size == 0:
    header = ['x', 'y', 'z', 'rx', 'ry', 'rz']
    for i in range(TOTAL_SENSOR_COUNT):
        header += [f'sensor_{i}_x', f'sensor_{i}_y', f'sensor_{i}_z']
    csv_writer.writerow(header)
    csv_file.flush()

# --- Surface and marker setup ---
marker_length = 0.04     # meters
gap = 0.005              # 5mm
spacing = marker_length + gap
surface_rows, surface_cols = 5, 4
surface_marker_ids = set(range(1, 21))
object_marker_id = 0
origin_marker_ids = (10, 11)
min_surface_markers = 4
calibration_samples = 20

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
     
    print("Ambient field (Earth + background):", ambient_field)
    return ambient_field



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
                # #divide each element by 1e6
                # matrix = [[val / 1e6 for val in row] for row in matrix]
                return matrix
            else:
                print(f"Unexpected number of values: {len(values)}")
        except ValueError as ve:
            print(f"Value error: {ve}")
    return None


def marker_obj_points(marker_length):
    h = marker_length / 2
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

def rvec_tvec_to_mat(rvec, tvec):
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = tvec.flatten()
    return T

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
aruco_params = cv2.aruco.DetectorParameters()
aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)

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
# for idx in range(20):
#     row = idx // surface_cols
#     col = idx % surface_cols
#     surface_points[idx + 1] = np.array([col * spacing, row * spacing, 0.0])



cap = cv2.VideoCapture(0)
 


origin_mean =  (0,0,0)
 


ambient_field = calibrate_ambient_field(ser)
# --- Main measurement loop ---
with open(file, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # ... rest of your code that writes rows ...

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _ = aruco_detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            frame = cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            detected_ids = ids.flatten()
            surface_obj_points, surface_img_points = [], []
            for i, marker_id in enumerate(detected_ids):
                if marker_id in surface_points:
                    surface_obj_points.append(surface_points[marker_id])
                    surface_img_points.append(corners[i][0].mean(axis=0))
            if len(surface_obj_points) >= min_surface_markers:
                surface_obj_points = np.array(surface_obj_points, dtype=np.float32)
                surface_img_points = np.array(surface_img_points, dtype=np.float32)
                success, surface_rvec, surface_tvec = cv2.solvePnP(surface_obj_points, surface_img_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                T_cam_surface = rvec_tvec_to_mat(surface_rvec, surface_tvec)
                T_surface_cam = np.linalg.inv(T_cam_surface)
                obj_idx = np.where(detected_ids == object_marker_id)[0]
                if len(obj_idx) > 0:
                    img_pts = corners[obj_idx[0]][0]
                    obj_pts = marker_obj_points(marker_length)
                    success, obj_rvec, obj_tvec = cv2.solvePnP(obj_pts, img_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
                    T_cam_obj = rvec_tvec_to_mat(obj_rvec, obj_tvec)
                    T_surface_obj = T_surface_cam @ T_cam_obj
                    obj_pos_in_surface = T_surface_obj[:3, 3]
                    obj_pos_relative = obj_pos_in_surface - origin_mean
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, obj_rvec, obj_tvec, marker_length*2)
                    cv2.putText(frame, f'Obj: {obj_pos_relative.round(3)}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                    matrix = parse_output_matrix() or None

                    #add the readings to the csv file
                    #subtract the ambient field from the current readings
                    
                            
                    if matrix:
                        for i in range(len(matrix)):
                                matrix[i] = [matrix[i][j] - ambient_field[i][j] for j in range(3)]
                        row = [obj_pos_relative[0], obj_pos_relative[1], obj_pos_relative[2]]
                        #get rotation in euler angles
                        rmat, _ = cv2.Rodrigues(obj_rvec)
                        sy = np.sqrt(rmat[0,0] * rmat[0,0] +  rmat[1,0] * rmat[1,0])
                        singular = sy < 1e-6
                        if not singular:
                            x_angle = np.arctan2(rmat[2,1] , rmat[2,2])
                            y_angle = np.arctan2(-rmat[2,0], sy)
                            z_angle = np.arctan2(rmat[1,0], rmat[0,0])
                        else:
                            x_angle = np.arctan2(-rmat[1,2], rmat[1,1])
                            y_angle = np.arctan2(-rmat[2,0], sy)
                            z_angle = 0
                        row += [x_angle, y_angle, z_angle]
                        for sensor_reading in matrix:
                            row += sensor_reading
                        csv_writer.writerow(row)

                        csv_file.flush()
                        print(f"Logged data: {row}")
                # Draw surface axes
                axes_3d = np.float32([[0,0,0], [0.05,0,0], [0,0.05,0], [0,0,0.05]])
                imgpts, _ = cv2.projectPoints(axes_3d, surface_rvec, surface_tvec, camera_matrix, dist_coeffs)
                origin_pt = tuple(imgpts[0].ravel().astype(int))
                pt_x = tuple(imgpts[1].ravel().astype(int))
                pt_y = tuple(imgpts[2].ravel().astype(int))
                pt_z = tuple(imgpts[3].ravel().astype(int))
                try:
                    cv2.line(frame, origin_pt, pt_x, (0,0,255), 3)
                    cv2.line(frame, origin_pt, pt_y, (0,255,0), 3)
                    cv2.line(frame, origin_pt, pt_z, (255,0,0), 3)
                    
                    
                except:
                    # in case of error, print the origin point
                    print("Error drawing axes at point:", origin_pt)
        cv2.imshow('Live Pose Estimation', frame)

        
        if cv2.waitKey(1) == 27:
            csv_file.close()
            break
        with open(file, 'r') as f:
            rows = f.readlines()
            if len(rows) > max_rows:
                print(f"Reached {max_rows} rows, stopping data collection.")

                break

        

cap.release()
cv2.destroyAllWindows()


