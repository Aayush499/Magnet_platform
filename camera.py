import cv2
import numpy as np
import pickle

# --- Load calibration ---
with open('output/calibration_data.pkl', 'rb') as f:
    calib_data = pickle.load(f)
camera_matrix = calib_data['camera_matrix']
dist_coeffs = calib_data['distortion_coefficients']

# --- Surface and marker setup ---
marker_length = 0.04     # meters
gap = 0.005              # 5mm
spacing = marker_length + gap
surface_rows, surface_cols = 5, 4
surface_marker_ids = set(range(1, 21))
object_marker_id = 0
origin_marker_ids = (10, 11)
min_surface_markers = 6
calibration_samples = 20

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

surface_points = {}
for idx in range(20):
    row = idx // surface_cols
    col = idx % surface_cols
    surface_points[idx + 1] = np.array([col * spacing, row * spacing, 0.0])

cap = cv2.VideoCapture(0)
origin_samples = []

# --- Calibration phase ---
print('Calibrating origin...')
while len(origin_samples) < calibration_samples:
    ret, frame = cap.read()
    if not ret:
        continue
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, _ = aruco_detector.detectMarkers(gray)
    if ids is not None:
        detected_ids = ids.flatten()
        if all(mid in detected_ids for mid in origin_marker_ids):
            o1 = surface_points[origin_marker_ids[0]]
            o2 = surface_points[origin_marker_ids[1]]
            midpoint = (o1 + o2) / 2
            origin_samples.append(midpoint)
            cv2.putText(frame, "Calibrating...", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
    cv2.imshow('Calibration', frame)
    if cv2.waitKey(1) == 27:  # ESC to quit
        cap.release()
        cv2.destroyAllWindows()
        exit()
print('Origin calibrated.')
origin_mean = np.mean(origin_samples, axis=0)
cv2.destroyAllWindows()

# --- Main measurement loop ---
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
        break
cap.release()
cv2.destroyAllWindows()
