import cv2
import numpy as np

def detect_landmarks(image, faces):
    # Create a copy of the image to draw landmarks on
    annotated_image = image.copy()
    color = (200, 160, 75)  # Color for the landmarks

    first_face_landmarks = None

    for face in faces:
        lmk = face.landmark_2d_106
        lmk_rounded = np.round(lmk).astype(int)

        # Draw each landmark as a circle
        for point in lmk_rounded:
            cv2.circle(annotated_image, tuple(point), 2, color, -1)

        # Save landmarks for the first face
        if first_face_landmarks is None:
            first_face_landmarks = lmk_rounded

    return annotated_image, first_face_landmarks