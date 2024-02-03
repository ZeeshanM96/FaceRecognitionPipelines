import matplotlib.pyplot as plt



def draw_histogram_intra(intra_class_similarities, intra_class_similarity_mean, model_name):
    
    # Data to plot
    fig, axs = plt.subplots(2, 1, figsize=(20, 12))

    # Data for the histograms
    labels = list(intra_class_similarities.keys())
    labels.remove('OutputImage-Pineline.jpg')

    similarities = [intra_class_similarities[label] for label in labels]

    # Plot for individual intra-class similarities
    axs[0].bar(labels, similarities, color='blue')
    axs[0].set_xlabel('Labels')
    axs[0].set_ylabel('Average Intra-Class Similarity Score')
    axs[0].set_title(f'Intra-Class Similarity Histogram by Label for {model_name}')
    axs[0].set_ylim(0, 1)
    axs[0].tick_params(labelrotation=90)

    # Plot for mean intra-class similarity
    axs[1].bar(['Mean Intra-Class Similarity'], [intra_class_similarity_mean], color='green')
    axs[1].set_ylabel('Mean Similarity Score')
    axs[1].set_title(f'Mean Intra-Class Similarity Histogram for {model_name}')
    axs[1].set_ylim(0, 1)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'DataAnalysis\CombinedIntraSimilarityHistogram_{model_name}.png')
    
def draw_histogram_inter(inter_class_similarities, inter_class_similarity_mean, model_name):
    
    # Data to plot
    fig, axs = plt.subplots(2, 1, figsize=(10, 12))

    num_bins = 20  # Or choose a different number that suits your data

    # Plot histogram for individual inter-class similarities
    axs[0].hist(inter_class_similarities, bins=num_bins, color='blue', range=(0,1))
    axs[0].set_xlabel('Inter-Class Similarity Score')
    axs[0].set_ylabel('Frequency')
    axs[0].set_title(f'Inter-Class Similarity Histogram for {model_name}')
    axs[0].set_ylim(0)

    # Plot for mean inter-class similarity
    axs[1].bar(['Mean Inter-Class Similarity'], [inter_class_similarity_mean], color='green')
    axs[1].set_ylabel('Mean Similarity Score')
    axs[1].set_title(f'Mean Inter-Class Similarity Histogram for {model_name}')
    axs[1].set_ylim(0, 1)

    # Adjust layout
    plt.tight_layout()

    # Save the figure
    plt.savefig(f'DataAnalysis\CombinedInterSimilarityHistogram_{model_name}.png')
    # plt.show()
