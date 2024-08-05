import argparse
import os
import torch
import random
import spacy
import nltk
import seaborn as sns
import gc
import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import subprocess
from datasets import concatenate_datasets
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

from tqdm import tqdm
from cleantext import clean
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from nltk.corpus import wordnet as wn
from nltk.tree import Tree
from torch.utils.data import Dataset
from datasets import load_dataset
from transformers import (
    RobertaTokenizer,
    RobertaModel,
    AutoTokenizer,
    AutoModel,
    RobertaConfig
)
from torch.cuda.amp import GradScaler, autocast


os.environ["TOKENIZERS_PARALLELISM"] = "false"

import warnings
warnings.filterwarnings('ignore')



class FeverDataset(Dataset):
    def __init__(self, dataset, tokenizer):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.labels = {
            "ENTAILMENT": 0,
            "NEUTRAL": 1,
            "CONTRADICTION": 2
        }

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        
        premise = item["premise"].encode('ascii', errors='ignore').decode()
        hypothesis = item["hypothesis"].encode('ascii', errors='ignore').decode()
        
        premise = clean(premise, lower=True)
        hypothesis = clean(hypothesis, lower=True)
        label = item["label"]
        
        # Format the input
        input = f"[CLS] {premise} [SEP] {hypothesis}"
        
        input_ids = self.tokenizer.encode(input, add_special_tokens=False, max_length=512, truncation=True)
        label = self.labels[label] # Convert the label to a number

        return torch.tensor(input_ids), torch.tensor(label)

def collate_fn(batch):
    input_ids = [item[0] for item in batch]
    labels = [item[1] for item in batch]

    # Pad the input_ids
    input_ids = nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=0)
    attention_mask = (input_ids != 0).float()
    return input_ids, torch.stack(labels), attention_mask

class NLIBaselines():
    def __init__(self, dataloader):
        self.dataloader = dataloader
    
    def random(self):
        all_labels = []
        all_predictions = []
        
        for batch in tqdm(self.dataloader):
            labels = batch[1].to(device)
            predictions = torch.randint(0, 3, (len(labels),)).to(device) # Create a tensor of random integers [0,2]
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return accuracy, precision, recall, f1
    
    def majority_class(self):
        label_counts = {0:0, 1:0, 2:0}
        
        # Find the most frequent class
        for batch in tqdm(self.dataloader):
            labels = batch[1].to(device)
            for label in labels:
                label = label.item()
                label_counts[label] += 1

        majority_class_label = max(label_counts, key=label_counts.get)
        
        all_labels = []
        all_predictions = []

        for batch in tqdm(self.dataloader):
            labels = batch[1].to(device)
            predictions = torch.full_like(labels, majority_class_label) # Create a tensor containing only the most frequent class
            
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
                
        accuracy = accuracy_score(all_labels, all_predictions)
        precision = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
        
        return accuracy, precision, recall, f1


class TransformerModel(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super(TransformerModel, self).__init__()
        self.transformer = model_name
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0]
        #cls_output = self.dropout(cls_output) TO USE?
        logits = self.classifier(cls_output)
        return logits

class TransformerModelDropout(nn.Module):
    def __init__(self, model_name, dropout_rate=0.1):
        super(TransformerModelDropout, self).__init__()
        self.transformer = model_name
        self.dropout = nn.Dropout(dropout_rate)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, 3)

    def forward(self, input_ids, attention_mask=None):
        input_ids = input_ids.to(device)
        attention_mask = attention_mask.to(device)
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        cls_output = last_hidden_state[:, 0]
        cls_output = self.dropout(cls_output)
        logits = self.classifier(cls_output)
        return logits


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, num_epochs, criterion, optimizer, patience=5):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.num_epochs = num_epochs
        self.criterion = criterion
        self.optimizer = optimizer

        self.validation_losses = []
        self.validation_accuracies = []
        self.validation_precisions = []
        self.validation_recalls = []
        self.validation_f1s = []
        self.train_losses = []

        self.patience = patience
        self.early_stop = False
        self.best_val_loss = np.Inf
        self.counter = 0
        self.best_model_state = None
        self.current_epoch = -1
        self.best_epoch = -1
        self.best_val_acc = -1

        # Initialize GradScaler for mixed precision training
        self.scaler = GradScaler()

    def train(self):
        torch.cuda.empty_cache()
        gc.collect()

        for epoch in range(self.num_epochs):
            print(f"\nTraining epoch {epoch + 1}")

            self.current_epoch += 1
            if self.early_stop:
                print("Early stopping triggered")
                break

            self.model.train()
            running_loss = 0.0

            for batch in tqdm(self.train_loader):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                attention_mask = batch[2].to(device)

                self.optimizer.zero_grad()

                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()

                running_loss += loss.item() * input_ids.size(0)

            epoch_loss = running_loss / len(self.train_loader.dataset)
            self.train_losses.append(epoch_loss)

            # Compute validation metrics at the end of each epoch
            val_loss, val_acc, val_precision, val_recall, val_f1 = self.evaluate(self.val_loader)

            self.validation_losses.append(val_loss)
            self.validation_accuracies.append(val_acc)
            self.validation_precisions.append(val_precision)
            self.validation_recalls.append(val_recall)
            self.validation_f1s.append(val_f1)

            # Check for early stopping
            self.check_early_stopping(val_acc)

            print(f"Epoch: {epoch + 1}, Best Validation Accuracy so far: {self.best_val_acc}\nValidation Loss: {val_loss}, Train Loss: {epoch_loss}, Validation Accuracy: {val_acc}, Precision: {val_precision}, Recall: {val_recall}, F1-score: {val_f1}")

    def check_early_stopping(self, val_acc):
        if val_acc > self.best_val_acc:
            self.best_val_acc = val_acc
            self.counter = 0

            self.best_epoch = self.current_epoch
            self.best_model_state = self.model.state_dict()
        else:
            self.counter += 1
            print(f"Model has not improved for {self.counter} epochs")
            if self.counter >= self.patience:
                self.early_stop = True
                print(f"Early stopping after {self.counter} epochs without improvement")

    def evaluate(self, data_loader, print_output=False):
        self.model.eval()
        total_loss = 0
        total = 0
        correct = 0
        all_labels = []
        all_preds = []
        
        with torch.no_grad():
            for batch in tqdm(data_loader):
                input_ids = batch[0].to(device)
                labels = batch[1].to(device)
                attention_mask = batch[2].to(device)
        
                with autocast():
                    logits = self.model(input_ids, attention_mask)
                    loss = self.criterion(logits, labels)
        
                total_loss += loss.item()
                total += labels.size(0)
                correct += (logits.argmax(1) == labels).sum().item()
        
                all_labels.extend(labels.cpu().numpy())
                all_preds.extend(logits.argmax(1).cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
        recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
        f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        if print_output:
            self.plot_confusion_matrix(all_labels, all_preds)
        
        return total_loss / len(data_loader), accuracy, precision, recall, f1


    def test(self):
        if self.best_model_state is not None:
            self.model.load_state_dict(self.best_model_state)
            print("Loaded best model from early stopping.")
    
        test_loss, test_acc, test_precision, test_recall, test_f1 = self.evaluate(self.test_loader, print_output=True)
        print(f"Test Loss: {test_loss}, Test Accuracy: {test_acc}, Precision: {test_precision}, Recall: {test_recall}, F1-score: {test_f1}")
        return test_loss, test_acc, test_precision, test_recall, test_f1
    
    

    def plot_confusion_matrix(self, labels, preds):
        cm = confusion_matrix(labels, preds)
        print(cm)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["NEUTRAL", "ENTAILMENT", "CONTRADICTION"])
        disp.plot(cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.show()

    def plot(self):
        plt.figure(figsize=(15, 10))

        best_epoch = self.best_epoch
        best_val_loss = self.validation_losses[best_epoch]

        plt.subplot(2, 2, 1)
        plt.plot(self.validation_losses, label="Validation Loss")
        plt.plot(self.train_losses, label="Train Loss")
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.axhline(y=best_val_loss, color='r', linestyle='--')
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Losses")

        plt.subplot(2, 2, 2)
        plt.plot(self.validation_accuracies, label="Validation Accuracy")
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title("Validation Accuracy")

        plt.subplot(2, 2, 3)
        plt.plot(self.validation_precisions, label="Validation Precision")
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Precision")
        plt.legend()
        plt.title("Validation Precision")

        plt.subplot(2, 2, 4)
        plt.plot(self.validation_recalls, label="Validation Recall")
        plt.axvline(x=best_epoch, color='r', linestyle='--', label='Best Epoch')
        plt.xlabel("Epoch")
        plt.ylabel("Recall")
        plt.legend()
        plt.title("Validation Recall")

        plt.show()
        


def freeze_layers(model, num_layers_to_freeze):
    if isinstance(model, nn.DataParallel):
        model = model.module
    for name, param in model.transformer.named_parameters():
        layer_num = int(name.split('.')[2]) if 'layer' in name else -1
        if layer_num >= 0 and layer_num < num_layers_to_freeze:
            param.requires_grad = False

#################
# MAIN FUNCTION #
#################

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["train", "test", "baselines"], help="Action to perform: train, test or baselines")
    parser.add_argument("--data", choices=["original", "adversarial"], required=False, help="Dataset type: original or adversarial")
    parser.add_argument("--model", choices=["roberta", "roberta_df", "deberta"], required=False, help="Model type: roberta, roberta_df or deberta")
    args = parser.parse_args()
    
    gc.collect()

    if args.model == None:
        args.model = "roberta"

    if args.model == "roberta" or args.model == "roberta_df":
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        config = RobertaConfig.from_pretrained('roberta-base')
        model = RobertaModel.from_pretrained('roberta-base', config=config)
    
    elif args.model == "deberta":
        model = AutoModel.from_pretrained("microsoft/deberta-v3-large")
        tokenizer = AutoTokenizer.from_pretrained('microsoft/deberta-v3-large')
    
    else:
        tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        
    
    
    # DATASET AND DATALOADER CREATION
    
    # Load the complete original dataset
    original_dataset = load_dataset("tommasobonomo/sem_augmented_fever_nli").shuffle(seed=42)
    
    if args.data == "original":
        train_dataset = FeverDataset(original_dataset["train"], tokenizer)
        val_dataset = FeverDataset(original_dataset["validation"], tokenizer)

        # Class labels frequencies
        class_labels = np.array([0, 1, 2])  # 0: ENTAILMENT, 1: NEUTRAL, 2: CONTRADICTION
        y_train = np.concatenate([
            np.full(31128, 0),     # ENTAILMENT
            np.full(7627, 1),      # NEUTRAL
            np.full(12331, 2)      # CONTRADICTION
        ])
        
        # Calculate class weights because we have unbalanced dataset
        class_weights = compute_class_weight(class_weight='balanced', classes=class_labels, y=y_train)
        class_weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

        criterion = nn.CrossEntropyLoss(weight=class_weights_tensor)
    
    elif args.data == "adversarial":
        # Load the augmented data
        augmented_set = load_dataset('json', data_files="augmented_train_data.jsonl").shuffle(seed=42)
        train_augmented_set = concatenate_datasets([augmented_set['train'], original_dataset["train"]])
        train_dataset = FeverDataset(train_augmented_set, tokenizer)
        
        val_dataset = load_dataset('json', data_files="augmented_val_data.jsonl")
        val_dataset = concatenate_datasets([val_dataset['train'], original_dataset["validation"]]) #called train but is validation data
        val_dataset = FeverDataset(val_dataset, tokenizer) 

        criterion = nn.CrossEntropyLoss()
    
    # Original test set for testing
    original_test_dataset = FeverDataset(original_dataset["test"], tokenizer)
    original_test_loader = torch.utils.data.DataLoader(original_test_dataset, batch_size=128, collate_fn=collate_fn, num_workers=4)

    # Adversarial test set for testing
    adversarial_test_dataset = FeverDataset(load_dataset("iperbole/adversarial_fever_nli")['test'], tokenizer)
    adversarial_test_loader = torch.utils.data.DataLoader(adversarial_test_dataset, batch_size=64, collate_fn=collate_fn, num_workers=4)

    if args.action == "baselines":
        print("ORIGINAL TEST-SET")
        print("Computing baselines on the original test-set...")
        baselines = NLIBaselines(original_test_loader)
        random_accuracy, random_precision, random_recall, random_f1 = baselines.random()
        majority_class_accuracy, majority_class_precision, majority_class_recall, majority_class_f1 = baselines.majority_class()
        print("Random baseline -> Accuracy: ", random_accuracy, ", Precision: ", random_precision, ", Recall: ", random_recall, ", F1-score: ", random_f1)
        print("Majority class baseline -> Accuracy: ", majority_class_accuracy, ", Precision: ", majority_class_precision, ", Recall: ", majority_class_recall, ", F1-score: ", majority_class_f1)

        print("ADVERSARIAL TEST-SET")
        print("Computing baselines on the adversarial test-set...")
        baselines = NLIBaselines(adversarial_test_loader)
        random_accuracy, random_precision, random_recall, random_f1 = baselines.random()
        majority_class_accuracy, majority_class_precision, majority_class_recall, majority_class_f1 = baselines.majority_class()
        print("Random baseline -> Accuracy: ", random_accuracy, ", Precision: ", random_precision, ", Recall: ", random_recall, ", F1-score: ", random_f1)
        print("Majority class baseline -> Accuracy: ", majority_class_accuracy, ", Precision: ", majority_class_precision, ", Recall: ", majority_class_recall, ", F1-score: ", majority_class_f1)

        return

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, collate_fn=collate_fn, num_workers=4)
    
    # Define the loss function and optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # 5e-5, 4e-5, 3e-5, and 2e-5
    # [https://github.com/ymcui/Chinese-BERT-wwm/blob/master/README_EN.md#baselines]
    # [https://arxiv.org/pdf/1810.04805]

    
    # Create the TransformerModel instance
    if args.model == "roberta_df":
        model = TransformerModelDropout(model)
        freeze_layers(model, num_layers_to_freeze=3)
    else:
        model = TransformerModel(model)

    # Use both GPUs if available
    model = nn.DataParallel(model).to(device)
    
    # Create the Trainer instance
    if args.model == "roberta" or args.model == "roberta_df":
        trainer = Trainer(model, train_loader, val_loader, None, 2, criterion, optimizer, patience = 3)
    elif args.model == "deberta":
        trainer = Trainer(model, train_loader, val_loader, None, 1, criterion, optimizer, patience = 2)

    # TRAINING
    if args.action == "train":
        if args.data == "original":
            print("\nTraining the model on the original dataset...")
            trainer.train()
            print("\nPlotting the training...")
            trainer.plot()
            if args.model == "roberta":
                torch.save(model.state_dict(), "roberta_base_original_dataset.pth")
            elif args.model == "roberta_df":
                torch.save(model.state_dict(), "roberta_base_df_original_dataset.pth")
            elif args.model == "deberta":
                torch.save(model.state_dict(), "deberta_large_original_dataset.pth")
        
        elif args.data == "adversarial":
            print("\nTraining the model on the augmented dataset...")
            trainer.train()
            print("\nPlotting the training...")
            trainer.plot()
            if args.model == "roberta":
                torch.save(model.state_dict(), "roberta_base_augmented_dataset.pth")
            elif args.model == "roberta_df":
                torch.save(model.state_dict(), "roberta_base_df_augmented_dataset.pth")
            elif args.model == "deberta":
                torch.save(model.state_dict(), "deberta_large_augmented_dataset.pth")

    # TESTING
    elif args.action == "test":
        if args.data == "original":
            
            print("Loading " + args.model + " model trained on the original dataset...")
            if args.model == "roberta":
                try:
                    model.load_state_dict(torch.load("roberta_base_original_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            elif args.model == "roberta_df":
                try:
                    model.load_state_dict(torch.load("roberta_base_df_original_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            elif args.model == "deberta":
                try:
                    model.load_state_dict(torch.load("deberta_large_original_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            
            print("ORIGINAL TEST-SET")
            
            print("Testing the model on the original test-set...")
            trainer.test_loader = original_test_loader
            trainer.test()
            print("\nADVERSARIAL TEST-SET")
            print("Testing the model on the adversarial test-set...")
            trainer.test_loader = adversarial_test_loader
            trainer.test()
            
        elif args.data == "adversarial":
            print("Loading " + args.model + " model trained on the augmented dataset...")
            if args.model == "roberta":
                try:
                    model.load_state_dict(torch.load("roberta_base_augmented_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            elif args.model == "roberta_df":
                try:
                    model.load_state_dict(torch.load("roberta_base_df_augmented_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            elif args.model == "deberta":
                try:
                    model.load_state_dict(torch.load("deberta_large_augmented_dataset.pth"))
                except FileNotFoundError:
                    print("Model not found. Train the model first.")
                    return
            
            print("ORIGINAL TEST-SET")
            print("Testing the model on the original test-set...")
            trainer.test_loader = original_test_loader
            trainer.test()
            
            print("\nADVERSARIAL TEST-SET")
            print("Testing the model on the adversarial test-set...")
            trainer.test_loader = adversarial_test_loader
            trainer.test()
    # BASELINES
    

if __name__ == "__main__":
    print("MODEL TRAINING AND TESTING ENVIRONMENT FOR NLI TASK - FRASCA EMANUELE 1836098\n")
    
    gc.collect()
    torch.cuda.empty_cache()
    
    print("Setting up the environment...")
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    try:
        nlp = spacy.load("en_core_web_sm")
    except IOError:
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
    
    nlp = spacy.load("en_core_web_sm")
    nltk.download('wordnet')

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print("Setting up finished")
    main()

   