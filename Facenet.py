from deepface import DeepFace
import cv2
import os
from landmarks import detect_landmarks
from detectface import detect_faces
from alignface import align_face, align_face_1

def extract_face_embedding(image_path):
    # Load the image
    img = cv2.imread(image_path)
    if img is None:
        print("Image not found or invalid image format.")
        return None
    try:
        # Perform face embedding
        embedding = DeepFace.represent(img, model_name='Facenet', enforce_detection=False)
        return embedding
    except Exception as e:
        print(f"An error occurred: {e}")
        return None
    
def verify_face(image_path1, image_path2):
    # Verify if two images are of the same person using VGG-Face model
    try:
        verification = DeepFace.verify(image_path1, image_path2, model_name='Facenet', enforce_detection=False)
        return verification["verified"]
    except Exception as e:
        print(f"An error occurred during face verification: {e}")
        return None


def extract_embeddings(folder_path, reference_image_path, app):
    embeddings = {}
    verifications = {}

    # Extract embedding for the preprocessed (aligned) reference image
    reference_embedding = DeepFace.represent(reference_image_path, model_name='Facenet', enforce_detection=False)
    if reference_embedding is None:
        print(f"Could not extract embedding for {reference_image_path}")
        return None, None
    embeddings[reference_image_path] = reference_embedding

    # Iterate over each image in the folder
    for image_name in os.listdir(folder_path):
        image_path = os.path.join(folder_path, image_name)
        img = cv2.imread(image_path)
        
        # Step 1: Face Detection
        detection_result = detect_faces(app, img)
        if detection_result is None or len(detection_result[0]) > 1:
            print(f"Skipped {image_name} due to no face or multiple faces detected.")
            continue
        faces, img = detection_result
        
        
        # Step 2: Landmark Detection
        img, first_face_landmarks = detect_landmarks(img, faces)
        if first_face_landmarks is None:
            print(f"No landmarks detected in {image_name}.")
            continue

        # Step 3: Face Alignment
        aligned_face = align_face_1(img)

        # Step 4: Extract Face Embedding
        if aligned_face is not None:
            embedding = DeepFace.represent(aligned_face, model_name='Facenet', enforce_detection=False)
            if embedding is not None:
                embeddings[image_path] = embedding
                is_same_person = verify_face(reference_image_path, image_path)
                verifications[image_path] = is_same_person
                print(f"Verification result for {image_name} and {reference_image_path}: {is_same_person}")
            else:
                print(f"Could not extract embedding for {image_name}")
        else:
            print(f"Could not align face in {image_name}")

    return embeddings, verifications

