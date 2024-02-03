from scipy.spatial.distance import cosine
import numpy as np
import itertools
from numpy import dot
from numpy.linalg import norm


def calculate_similarity(embeddings, reference_image_path):
    reference_embedding = embeddings[reference_image_path][0]['embedding']
    similarities = {}

    for image_path, embedding_list in embeddings.items():
        if image_path != reference_image_path:
            # Again, taking the first embedding from the list
            embedding = embedding_list[0]['embedding']
            similarity = 1 - cosine(reference_embedding, embedding)
            similarities[image_path] = similarity

    return similarities

def extract_labels(similarities, reference_image_path):
    labels = {}
    # Add the reference image label
    reference_label = reference_image_path
    labels[reference_image_path] = reference_label

    for image_path, _ in similarities.items():
        label = image_path.split('\\')[-1].split('_')[1]
        labels[image_path] = label
    return labels

def calculate_intra_class_similarity(embeddings, labels):
    # Dictionary to hold intra-class similarities
    similarities_intra = {}
    # Iterate over each unique label
    for label in set(labels.values()):
        # Get all embeddings for the current label
        label_embeddings = [embed_info[0]['embedding'] for path, embed_info in embeddings.items() if labels[path] == label]
        # Calculate similarity scores for each unique pair of embeddings within the same label
        label_similarities = [1 - cosine(label_embeddings[i], label_embeddings[j]) 
                              for i in range(len(label_embeddings)) 
                              for j in range(i+1, len(label_embeddings))]
        # Store the average similarity for the current label
        similarities_intra[label] = np.mean(label_similarities) if label_similarities else 0
    
    # Calculate the overall mean intra-class similarity across all labels
    overall_mean_intra_similarity = np.mean(list(similarities_intra.values()))
    # print("Overall Mean Intra-Class Similarity:", overall_mean_intra_similarity)
    # print("Intra-Class Similarities by Label:", similarities_intra)
    return overall_mean_intra_similarity, similarities_intra

def calculate_inter_class_similarity(embeddings_dict, labels_dict):
    keys = embeddings_dict.keys() & labels_dict.keys()  # Intersection of keys from both dictionaries
    embeddings = [embeddings_dict[key][0]['embedding'] for key in keys]
    labels = [labels_dict[key] for key in keys]

    # Normalize the embeddings
    normalized_embeddings = [emb / np.linalg.norm(emb) for emb in embeddings if np.linalg.norm(emb) != 0]

    # Calculate inter-class similarities
    similarities_inter = []
    for i in range(len(normalized_embeddings)):
        for j in range(i + 1, len(normalized_embeddings)):
            # Check if the pair is from different classes
            if labels[i] != labels[j]:
                # Calculate cosine for normalized vectors
                similarity_score = 1 - cosine(normalized_embeddings[i], normalized_embeddings[j])
                similarities_inter.append(similarity_score)
    
    overall_mean_inter_similarity = np.mean(similarities_inter) if similarities_inter else 0
    return overall_mean_inter_similarity, similarities_inter

