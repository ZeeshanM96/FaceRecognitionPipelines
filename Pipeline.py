import cv2
from insightface.app import FaceAnalysis
from Facenet import extract_face_embedding, extract_embeddings
from Cosine_similarities import calculate_similarity, calculate_intra_class_similarity, calculate_inter_class_similarity, extract_labels
from ArcFace import extract_face_embedding_arcface, extract_embeddings_arcface
from landmarks import detect_landmarks
from detectface import detect_faces 
from alignface import align_face
from Vggface import extract_face_embedding_vgg, extract_embeddings_vgg
from Analysis import draw_histogram_intra, draw_histogram_inter
from roc_curve import draw_roc_curve, Dist_similarityscores, draw_confusionMatrix
from Exportfile import create_excel_file, excel_cosine_similarity


def get_user_choice():
    print("From the below list which model you want to use:")
    print("1- DeepFace(facenet)")
    print("2- ArcFace")
    print("3- VGG-Face")
    while True:
        choice = input("Enter your choice between (1,2 and 3): ")
        if choice == "1":
            return choice, "DeepFace(facenet)"
        elif choice == "2":
            return choice, "ArcFace"
        elif choice == "3":
            return choice, "VGG-Face"
        else:
            print("Wrong choice, please try again.")

def main():
    app = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
    # app.prepare(ctx_id=0, det_size=(640, 640))
    # Initialize the FaceAnalysis app
    app = FaceAnalysis(allowed_modules=['detection', 'scrfd_person_2.5g','landmark_2d_106'])
    app.prepare(ctx_id=0, det_size=(256, 256))

    # Load image
    img = cv2.imread('89_PaulAnka_68_m.jpg')

    # Step 1: Face Detection
    detection_result = detect_faces(app, img)
    if detection_result is None or len(detection_result[0]) > 1:
        print(f"Skipped {img} due to no face or multiple faces detected.")
        return
    faces, Image = detection_result

    # Step 2: Landmark Detection and get annotated image
    annotated_image, first_face_landmarks = detect_landmarks(Image, faces)
    
    # Check if landmarks are available
    if first_face_landmarks is not None:
        # Step 3: Face Alignment
        aligned_face = align_face(annotated_image)
        aligned_face_path = 'OutputImage-Pineline.jpg'
        cv2.imwrite(aligned_face_path, aligned_face)

        
        choice, model_name = get_user_choice()
        
        # Step 4: Extract Face Embedding     
        
        if(choice == "1" ): 
            embedding = extract_face_embedding(aligned_face_path)
        elif(choice == "2" ):
            embedding = extract_face_embedding_arcface(aligned_face_path)
        elif(choice == "3" ):
            embedding = extract_face_embedding_vgg(aligned_face_path)
        else:
            print("wrong choice")     
        
        if embedding is not None:
            print("Face Embedding Vector Created! Moving on to Face recognition.")
        else:
            print("Could not extract face embedding.")
    else:
        print("No landmarks detected for the first face.")
        
    print("-------------------------Face Embeddings----------------------------------")
    print(embedding)
        
    # Step 5: Face Recognition
    folder_path = 'Dataset\AgeDB'
    reference_image_path = aligned_face_path
    print("-------------------------Face Verifications----------------------------------")
    if(choice == "1" ):    
        embeddings, verifications = extract_embeddings(folder_path, reference_image_path, app)
    elif(choice == "2" ):  
        embeddings, verifications = extract_embeddings_arcface(folder_path, reference_image_path,app)
    elif(choice == "3" ):
        embeddings, verifications = extract_embeddings_vgg(folder_path, reference_image_path,app)
    if embeddings is None:
        print("Failed to extract embeddings.")
        return

    # Step 6: Calculate cosine 
    print("-------------------------Cosine Similarities----------------------------------")
    similarities = calculate_similarity(embeddings, reference_image_path)
    # Step 7: Print similarities
    for image_path, similarity in similarities.items():
     print(f"Similarity between {reference_image_path} and {image_path}: {similarity}")
    
 
    # Step 9: Intra and Inter similarities 
    labels = extract_labels(similarities, reference_image_path)
    intra_class_similarity_mean, intra_class_similarity = calculate_intra_class_similarity(embeddings, labels)
    inter_class_similarity_mean, inter_class_similarity = calculate_inter_class_similarity(embeddings, labels)
    # import pdb
    # pdb.set_trace()
    # Step 10: draw Histogram
    draw_histogram_intra(intra_class_similarity, intra_class_similarity_mean, model_name)
    draw_histogram_inter(inter_class_similarity, inter_class_similarity_mean, model_name)
    print("-------------------------Mean Intra & Inter Similarity----------------------------------")
    print("Mean Intra-class Similarity:", intra_class_similarity_mean)
    print("Mean Inter-class Similarity:", inter_class_similarity_mean)
    
    # Step 11 : Draw Roc-curve
    true_labels = []
    similarity_scores = []
    for image_path, verified in verifications.items():
        true_labels.append(int(verified))  # Convert True/False to 1/0
        similarity_scores.append(similarities[image_path])
    draw_roc_curve(similarity_scores, true_labels, model_name)
    
    # Step 12: Distribution of Similarity Scores for Matches and Non-Matches
    Dist_similarityscores(similarity_scores, true_labels, model_name)
    
    # Step 13: Confusion Matrix
    draw_confusionMatrix(similarity_scores, true_labels, model_name)
    
    #Step 14: Create Excel File
    excel_cosine_similarity(similarities, model_name, reference_image_path,true_labels, similarity_scores)
    create_excel_file(similarities, model_name, reference_image_path,true_labels, similarity_scores)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
 