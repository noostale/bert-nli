import json
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import load_dataset

def save_plots():
    # Load the original dataset
    dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli")
    
    # Load the adversarial test-set
    adversarial_test_data = load_dataset("iperbole/adversarial_fever_nli")['test']
    
    # Load the augmented train and validation datasets
    augmented_train_dataset_file = "augmented_train_data.jsonl"
    augmented_validation_dataset_file = "augmented_val_data.jsonl"
    augmented_train_labels = []
    augmented_validation_labels = []
    
    # Load the augmented train data
    with open(augmented_train_dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            augmented_train_labels.append(json.loads(line)['label'])
    
    # Load the augmented validation data
    with open(augmented_validation_dataset_file, 'r', encoding='utf-8') as f:
        for line in f:
            augmented_validation_labels.append(json.loads(line)['label'])
    
    # Extract the original dataset splits
    train_labels = dataset["train"]["label"]
    validation_labels = dataset["validation"]["label"]
    test_labels = dataset["test"]["label"]
    
    # Extract the adversarial test-set
    adversarial_test_labels = adversarial_test_data["label"]
    
    # Define the label names
    label_names = ["ENTAILMENT", "CONTRADICTION", "NEUTRAL"]
    
    # Combine original train set and augmented train set labels
    combined_train_labels = train_labels + augmented_train_labels
    combined_validation_labels = validation_labels + augmented_validation_labels
    
    # Create a function to prepare data for plotting
    def prepare_data(labels, label_names):
        counts = pd.Series(labels).value_counts().reindex(label_names, fill_value=0)
        df = pd.DataFrame({'Label': label_names, 'Count': counts})
        return df
    
    # Prepare data for all sets
    train_df = prepare_data(train_labels, label_names)
    validation_df = prepare_data(validation_labels, label_names)
    test_df = prepare_data(test_labels, label_names)
    adversarial_df = prepare_data(adversarial_test_labels, label_names)
    augmented_train_df = prepare_data(augmented_train_labels, label_names)
    augmented_validation_df = prepare_data(augmented_validation_labels, label_names)
    combined_train_df = prepare_data(combined_train_labels, label_names)
    combined_validation_df = prepare_data(combined_validation_labels, label_names)
    
    # Create the plot
    fig, axs = plt.subplots(4, 2, figsize=(15, 20))
    fig.suptitle("Label Distribution in Different Sets")
    
    # Function to plot a single subplot
    def plot_subplot(ax, data, title):
        sns.barplot(x='Label', y='Count', data=data, ax=ax)
        ax.set_title(title)
        ax.set_ylabel('Count')
        ax.set_xlabel('Label')
        
        # Add count
        for i, v in enumerate(data['Count']):
            ax.text(i, v, f"{v}", ha='center', va='bottom')
    
    # Plot all subplots
    plot_subplot(axs[0, 0], train_df, "Original Train Set")
    plot_subplot(axs[0, 1], validation_df, "Original Validation Set")
    plot_subplot(axs[1, 0], test_df, "Original Test Set")
    plot_subplot(axs[1, 1], adversarial_df, "Adversarial Test Set")
    plot_subplot(axs[2, 0], augmented_train_df, "Augmented Train Set")
    plot_subplot(axs[2, 1], augmented_validation_df, "Augmented Validation Set")
    plot_subplot(axs[3, 0], combined_train_df, "Combined Train Set")
    plot_subplot(axs[3, 1], combined_validation_df, "Combined Validation Set")
    
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig('plots.png')
    plt.show()

if __name__ == "__main__":
    save_plots()