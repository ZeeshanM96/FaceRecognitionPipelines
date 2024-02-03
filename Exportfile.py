import pandas as pd
from sklearn.metrics import roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import numpy as np


def create_excel_file(similarities, model_name, reference_image_path,true_labels, similarity_scores):
    model_names = ["DeepFace(facenet)", "ArcFace", "VGG-Face"]
    combined_roc(model_names)
    combined_precision_recall(model_names)
    plot_boxplots_of_similarity_scores(model_names)
    plot_histograms_of_similarity_scores(model_names)

def excel_cosine_similarity(similarities, model_name, reference_image_path,true_labels, similarity_scores):
    # Convert the similarities dictionary into a list of dictionaries suitable for DataFrame
    data = []
    if len(similarities) == len(true_labels) == len(similarity_scores):
        # Convert the similarities dictionary into a list of dictionaries for each row
        data = [
            {
                'reference_image_path': reference_image_path,
                'image_path': image_path,
                'similarity': sim_score,
                'true_value': true_label,
                'similarity_score': sim_score_2
            }
            for (image_path, sim_score), true_label, sim_score_2 in zip(similarities.items(), true_labels, similarity_scores)
        ]

        # Convert the list of dictionaries to a DataFrame
        df = pd.DataFrame(data)

        # Specify the filename
        filename = f'DataAnalysis\Cosine_Similarities-{model_name}.xlsx'

        # Write the DataFrame to an Excel file
        df.to_excel(filename, index=False)

        print(f"Excel file '{filename}' has been created.")
    else:
        print("The length of the data lists do not match. Please check your data.")

def combined_roc(model_names):
    # Dictionary to hold model data
    models_data = {}
    # Loop through each model name to load the data from each file
    for model_name in model_names:
        file_path = f"DataAnalysis\\Cosine_Similarities-{model_name}.xlsx"
        data = pd.read_excel(file_path)
        # Assuming 'similarity_score' is the column with the similarity scores
        # and 'true_label' is the column with the true labels
        models_data[model_name] = {
            'true_value': data['true_value'].tolist(),
            'similarity_score': data['similarity_score'].tolist() 
        }

    plt.figure(figsize=(10, 8))

    # Plot ROC curve for each model
    for model_name, data in models_data.items():
        fpr, tpr, _ = roc_curve(data['true_value'], data['similarity_score'])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], 'k--', label='Chance')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves for Multiple Models')
    plt.legend(loc='lower right')
    plt.savefig(f'DataAnalysis\CombinedROCCurve.png')
    
def combined_precision_recall(model_names): 
    models_data = {}
    # Loop through each model name to load the data from each file
    for model_name in model_names:
        file_path = f"DataAnalysis\\Cosine_Similarities-{model_name}.xlsx"
        data = pd.read_excel(file_path)
        # Assuming 'similarity_score' is the column with the similarity scores
        # and 'true_value' is the column with the true labels
        models_data[model_name] = {
            'true_value': data['true_value'].tolist(),
            'similarity_score': data['similarity_score'].tolist() 
        }

    plt.figure(figsize=(10, 8))

    # Plot Precision-Recall curve for each model
    for model_name, data in models_data.items():
        precision, recall, _ = precision_recall_curve(data['true_value'], data['similarity_score'])
        ap_score = average_precision_score(data['true_value'], data['similarity_score'])
        plt.plot(recall, precision, lw=2, label=f'{model_name} (AP = {ap_score:.2f})')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Combined Precision-Recall Curves')
    plt.legend(loc="best")
    plt.grid(True)
    plt.savefig(f'DataAnalysis\CombinedPrecisionRecallCurves.png')
    
def plot_boxplots_of_similarity_scores(model_names):
    similarity_scores = []

    # Read data and prepare for plotting
    for model_name in model_names:
        file_path = f"DataAnalysis\\Cosine_Similarities-{model_name}.xlsx"
        data = pd.read_excel(file_path)
        scores = data['similarity_score'].tolist()
        similarity_scores.append(scores)  # Appending list of scores for each model

    # Plot boxplots for similarity scores
    plt.figure(figsize=(10, 7))
    plt.boxplot(similarity_scores, labels=model_names)
    plt.title('Boxplots of Similarity Scores')
    plt.ylabel('Similarity Score')
    plt.savefig(f'DataAnalysis\BoxplotSimilaritScores.png')

def plot_histograms_of_similarity_scores(model_names):
    similarity_scores = []
    # Read data and prepare for plotting
    for model_name in model_names:
        file_path = f"DataAnalysis\\Cosine_Similarities-{model_name}.xlsx"
        data = pd.read_excel(file_path)
        scores = data['similarity_score'].tolist()
        similarity_scores.append(scores)  # Appending list of scores for each model

    # Plot histograms of similarity scores
    plt.figure(figsize=(10, 7))
    for i, scores in enumerate(similarity_scores):
        plt.hist(scores, bins=20, alpha=0.7, label=model_names[i])

    plt.title('Histograms of Similarity Scores')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.legend(loc="best")
    plt.savefig(f'DataAnalysis\HistogramsSimilarityScores.png')
