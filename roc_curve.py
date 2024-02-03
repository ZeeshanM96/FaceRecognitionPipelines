from sklearn.metrics import roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np

def calculate_tpr_fpr(similarity_scores, true_labels, threshold):
    predictions = [1 if score >= threshold else 0 for score in similarity_scores]
    TP = sum((p == 1 and l == 1) for p, l in zip(predictions, true_labels))
    FP = sum((p == 1 and l == 0) for p, l in zip(predictions, true_labels))
    TN = sum((p == 0 and l == 0) for p, l in zip(predictions, true_labels))
    FN = sum((p == 0 and l == 1) for p, l in zip(predictions, true_labels))

    TPR = TP / (TP + FN) if TP + FN else 0
    FPR = FP / (FP + TN) if FP + TN else 0
    return TPR, FPR

def Dist_similarityscores(similarity_scores, true_labels, model_name):

    match_scores = [score for score, label in zip(similarity_scores, true_labels) if label == 1]
    non_match_scores = [score for score, label in zip(similarity_scores, true_labels) if label == 0]

    # Plot histograms for matches and non-matches
    plt.figure(figsize=(10, 5))

    plt.hist(match_scores, bins=10, alpha=0.7, label='Matches')
    plt.hist(non_match_scores, bins=10, alpha=0.7, label='Non-matches', color='red')

    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of Similarity Scores for Matches and Non-Matches - '+ model_name)
    plt.legend(loc='upper right')
    plt.savefig('DataAnalysis\DistSimilarity' + model_name + '.png')

def draw_roc_curve(similarity_scores, true_labels, model_name):
    min_score, max_score = min(similarity_scores), max(similarity_scores)
    thresholds = np.linspace(min_score, max_score, 50)  # Increase the number to generate more points

    # Calculate TPR and FPR for each threshold
    tpr_list = []
    fpr_list = []
    for threshold in thresholds:
        tpr, fpr = calculate_tpr_fpr(similarity_scores, true_labels, threshold)
        tpr_list.append(tpr)
        fpr_list.append(fpr)

    roc_auc = auc(fpr_list, tpr_list)
    # Now plot the ROC curve with the calculated TPR and FPR values
    plt.figure()
    plt.plot(fpr_list, tpr_list, color='blue', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='orange', lw=2, linestyle='--', label='Chance level')
    plt.xlabel('False Positive Rate')
    plt.ylabel(f'True Positive Rate {model_name}')
    plt.title('ROC Curve - ' +model_name)
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(f'DataAnalysis\ROCCurve_{model_name}.png')  # Save the figure before showing it
    
def draw_confusionMatrix(similarity_scores, true_labels, model_name):
    threshold = 0.5
    predicted_labels = [1 if score >= threshold else 0 for score in similarity_scores]

    # Compute the confusion matrix
    cm = confusion_matrix(true_labels, predicted_labels)

    # Plot the confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix '+model_name)
    plt.savefig(f'DataAnalysis\ConfusionMatrix_{model_name}.png')