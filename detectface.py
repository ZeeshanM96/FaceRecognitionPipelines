from insightface.app import FaceAnalysis

def detect_faces(app, image):
    det_sizes = [(640, 640), (256, 256), (64, 64)]  # Different detection sizes to try
    faces = None

    for det_size in det_sizes:
        print(f"Trying detection with size: {det_size}")
        app.prepare(ctx_id=0, det_size=det_size)
        faces = app.get(image)
        if len(faces) == 1:
            print("1 face detected.")
            rimg = app.draw_on(image, faces)
            return faces, rimg
        elif len(faces) == 0:
            print("No faces detected.")
        else:
            print(f"Multiple faces detected: {len(faces)}.")

    return None  # Return None if the correct number of faces is not detected
