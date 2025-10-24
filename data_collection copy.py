import math
import serial

import cv2
import numpy as np
import pickle


# Adjust these for your setup
SERIAL_PORT = '/dev/ttyACM0'  # For Linux/Mac, usually /dev/ttyACM0 or /dev/ttyUSB0
# SERIAL_PORT = 'COM3'         # For Windows
BAUD_RATE = 115200            # Set to match your Arduino sketch
TOTAL_SENSOR_COUNT= 8

points = [(-13,13), (0,13), (13,13),
          (-13,0),  (0,0),  (13,0),
          (-13,-13),(0,-13),(13,-13)]
orientations = [   (180-90, -90, 90-180), (0,0,0), (0,0,180), (0,0, -90)]
# add 90 to xangle 180 to zangle
for i in range(len(orientations)):
    x, y, z = orientations[i]
   
    orientations[i] = (x + 90, y, z + 180)
    #convert to between -180 to 180
    #check if any angle greater than 180
    if orientations[i][0] > 180:
        orientations[i] = (orientations[i][0] - 360, orientations[i][1], orientations[i][2])
    if orientations[i][1] > 180:
        orientations[i] = (orientations[i][0], orientations[i][1] - 360, orientations[i][2])
    if orientations[i][2] > 180:
        orientations[i] = (orientations[i][0], orientations[i][1], orientations[i][2] - 360)

max_rows = 5000
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
    header = ['position', 'orientation', 'x', 'y', 'z', 'rx', 'ry', 'rz']
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
object_marker_id = 1
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
 
#rotation matrix

# angle_x = np.deg2rad(90)
# angle_z = np.deg2rad(180)

# Rx_90 = np.array([
#     [1, 0, 0],
#     [0, np.cos(angle_x), -np.sin(angle_x)],
#     [0, np.sin(angle_x),  np.cos(angle_x)]
# ])

# Rz_180 = np.array([
#     [np.cos(angle_z), -np.sin(angle_z), 0],
#     [np.sin(angle_z),  np.cos(angle_z), 0],
#     [0, 0, 1]
# ])
# R_offset = Rz_180 @ Rx_90

def euler_zyx_to_matrix(z, y, x):
    # z, y, x in radians
    cz, sz = np.cos(z), np.sin(z)
    cy, sy = np.cos(y), np.sin(y)
    cx, sx = np.cos(x), np.sin(x)
    Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
    Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
    Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
    return Rz @ Ry @ Rx

def angles_allclose(a, b, atol):
    diffs = (np.array(a) - np.array(b) + np.pi) % (2*np.pi) - np.pi
    return np.all(np.abs(diffs) <= atol)
ambient_field = calibrate_ambient_field(ser)
# --- Main measurement loop ---
brk = False
with open(file, 'a', newline='') as csv_file:
    csv_writer = csv.writer(csv_file)
    # ... rest of your code that writes rows ...
    for o in orientations:
            for p in points:
                print(f"Place magnet at position {p} with orientation {o} degrees.")
                input("Press Enter when ready...")

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
                                #put the rotation angles in euler angles
                                
                                matrix = parse_output_matrix() or None

                                #add the readings to the csv file
                                #subtract the ambient field from the current readings
                                
                                
                                if matrix:
                                    for i in range(len(matrix)):
                                            matrix[i] = [matrix[i][j] - ambient_field[i][j] for j in range(3)]
                                    row = [o, p, obj_pos_relative[0], obj_pos_relative[1], obj_pos_relative[2]]
                                    #get rotation in euler angles with respect to surface
                                    R_surface_obj = T_surface_obj[:3, :3]
                                    R_imaginary_surface_obj =   R_surface_obj
                                    sy = np.sqrt(R_imaginary_surface_obj[0,0]**2 + R_imaginary_surface_obj[1,0]**2)
                                    singular = sy < 1e-6
                                    if not singular:
                                        x_angle = np.arctan2(R_imaginary_surface_obj[2,1], R_imaginary_surface_obj[2,2])
                                        y_angle = np.arctan2(-R_imaginary_surface_obj[2,0], sy)
                                        z_angle = np.arctan2(R_imaginary_surface_obj[1,0], R_imaginary_surface_obj[0,0])
                                    else:
                                        x_angle = np.arctan2(-R_imaginary_surface_obj[1,2], R_imaginary_surface_obj[1,1])
                                        y_angle = np.arctan2(-R_imaginary_surface_obj[2,0], sy)
                                        z_angle = 0
                                    #verify that the point and orientation match the requested ones
                                    cv2.putText(frame, f'Rot: {(x_angle*180/math.pi).round(3)}, {(y_angle*180/math.pi).round(3)}, {(z_angle*180/math.pi).round(3)}', (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2)
                                    requested_position = (p[0]/100, p[1]/100, 0)
                                    requested_orientation = (math.radians(o[0]), math.radians(o[1]), math.radians(o[2]))
                                    angle_tol = 2 * math.pi/180  # 5 degrees in radians
                                    # Convert both sets of angles to matrices
                                    R1 = euler_zyx_to_matrix(z_angle, y_angle, x_angle)
                                    R2 = euler_zyx_to_matrix( requested_orientation[2], requested_orientation[1], requested_orientation[0])
                                    R_rel = R1.T @ R2
                                    angle_tol = 5 * np.pi / 180  # 5 degrees in radians
                                    angle = np.arccos((np.trace(R_rel) - 1) / 2)


                                    if (np.allclose(obj_pos_relative, requested_position, atol=5e-2) and
                                        angle <= angle_tol):
                                        row += [x_angle, y_angle, z_angle]
                                        for sensor_reading in matrix:
                                            row += sensor_reading
                                        csv_writer.writerow(row)


                                        
                                    else:
                                        #red message for position/orientation mismatch
                                        cv2.putText(frame, f'Position/Orientation mismatch. Please adjust.', (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                                        #cv put text the required and measured position/orientation
                                        #round the required orientation to 3 decimal places
                                        requested_orientation_rounded = tuple(round(math.degrees(angle), 3) for angle in requested_orientation)
                                        cv2.putText(frame, f'Required: {requested_position}, {requested_orientation_rounded}', (10,120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,255), 2)
                                         
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
                        brk = True
                        break
                    with open(file, 'r') as f:
                        # rows = f.readlines()
                        # if len(rows) > max_rows:
                        #     print(f"Reached {max_rows} rows, stopping data collection.")
                        #count how many rows are in the csv with orientation = o and position = p
                        row_count = 0
                        for row in csv.reader(f):
                            if len(row) > 0 and row[0] == str(o) and row[1] == str(p):
                                row_count += 1
                        if row_count >= max_rows :
                            print(f"Reached {row_count} rows for position {p} and orientation {o}, moving to next.")

                            break
                if brk:
                    break
            if brk:
                break
            

cap.release()
cv2.destroyAllWindows()


