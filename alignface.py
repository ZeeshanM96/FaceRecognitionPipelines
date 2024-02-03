import cv2
import numpy as np
from retinaface import RetinaFace

def align_face(image):
    img_path = image
    img = img_path
    resp = RetinaFace.detect_faces(img_path=img_path)

    if len(resp) > 0:
        face_key = list(resp.keys())[0]  # Process the first detected face
        x1, y1 = resp[face_key]["landmarks"]["right_eye"]
        x2, y2 = resp[face_key]["landmarks"]["left_eye"]

        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Rotate the image
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
    return aligned_image

def align_face_1(image):
    img_path = image
    img = img_path
    resp = RetinaFace.detect_faces(img_path=img_path)
    
    if isinstance(resp, tuple):
        resp = resp[0]

    if len(resp) > 0:
        face_key = list(resp.keys())[0]  # Process the first detected face
        x1, y1 = resp[face_key]["landmarks"]["right_eye"]
        x2, y2 = resp[face_key]["landmarks"]["left_eye"]

        # Calculate angle
        angle = np.degrees(np.arctan2(y2 - y1, x2 - x1))

        # Rotate the image
        center = ((x1 + x2) // 2, (y1 + y2) // 2)
        M = cv2.getRotationMatrix2D(center, angle, 1)
        aligned_image = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        return aligned_image
    else:
        return None  # Return None if no face is detected
