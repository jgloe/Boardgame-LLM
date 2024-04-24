import pandas as pd
import numpy as np
import random
from preprocess import train_val_test_split, tokenize
from sklearn.metrics import cohen_kappa_score
import torch
from torch.utils.data import DataLoader
import evaluate
import seaborn as sns
import matplotlib.pyplot as plt

import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def agreement_matrix(y, y_pred):
    mat = np.zeros((11, 11))
    for i in range(len(y)):
        row = int(y[i])
        col = int(y_pred[i])
        mat[row, col] += 1
    return mat

def test(model, loader, device, accuracy):
    predictions, y_true = [], []
    model.eval()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)

        logits = outputs.logits
        batch_predictions = torch.argmax(logits, dim=-1)
        accuracy.add_batch(predictions=batch_predictions, references=batch['labels'])
        predictions = np.append(predictions, batch_predictions.cpu().numpy().astype(int))
        y_true = np.append(y_true, batch['labels'].cpu().numpy().astype(int))
    return predictions, y_true

if __name__ == '__main__':

    # Parameters
    val_size = 0.1
    test_size = 0.1
    num_classes = 11
    test_batch_size = 8
    save_dir = './saved_models'
    data_file = './bgg-19m-reviews.csv'
    checkpoint = 'bert-base-uncased'
    # checkpoint = "jpelhaw/longformer-base-plagiarism-detection"
    results_dir = './results'
    random_seed = 123

    date = '2024-04-23'
    time = '14-33'
    pretrain_path = os.path.join(save_dir, checkpoint, f"{date}_{time}.pth.tar")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    # Load data and clean up dataframe
    reviews = pd.read_csv(data_file)
    reviews = reviews.drop('Unnamed: 0', axis=1)
    reviews = reviews.dropna()
    reviews['rating'] = reviews['rating'].apply(lambda x: round(x))
    reviews.rename(columns={'rating': 'labels'}, inplace=True)

    # For quick testing, restrict to smaller dataset
    # Remove this if analyzing full dataset
    sample_size = 500
    reviews = reviews.sample(sample_size, random_state=random_seed)

    print('Loaded data')

    # Create DatasetDictionary with train/validation/test splits
    tvt = train_val_test_split(reviews, val_size, test_size, random_seed)
    print('Performed train/test split')

    # Tokenize text
    data_collator, tokenized_rating = tokenize(tvt, checkpoint, num_classes)
    print('Tokenized text')

    # Get model and trained weights from fine-tuned training
    fine_tune = torch.load(pretrain_path, map_location=device)
    model = fine_tune['model']
    model.load_state_dict(fine_tune['state_dict'])
    model.to(device)

    # Create evaluation metrics
    accuracy = evaluate.load('accuracy')

    def compute_metrics(eval_pred):
        probs, labels = eval_pred
        predictions = np.argmax(probs, axis=1)
        return accuracy.compute(predictions=predictions, references=labels)
    
    # Create dataloader
    test_dataloader = DataLoader(
        tokenized_rating['test'], shuffle=False, batch_size=test_batch_size, collate_fn=data_collator
    )

    predictions, y_true = test(model, test_dataloader, device, accuracy)

    test_acc = accuracy.compute()['accuracy']
    test_kappa = cohen_kappa_score(predictions, y_true, weights='quadratic')

    print(f"\n Accuracy: {100*test_acc:0.2f}% \n Cohen's kappa: {test_kappa:0.3f}")

    # Plot agreement matrix
    colormap = sns.dark_palette('#69d', as_cmap=True)
    plt.figure(figsize=(10,10))
    mat = agreement_matrix(y_true, predictions)
    sns.set_theme(font_scale=3.0)
    s = sns.heatmap(mat, cmap=colormap, annot=True, cbar=False, annot_kws={'fontsize': 36})
    # s.axhline(y=0, color='k', linewidth=10)
    # s.axhline(y=mat.shape[1], color='k', linewidth=10)
    # s.axvline(x=0, color='k', linewidth=10)
    # s.axvline(x=mat.shape[0], color='k', linewidth=10)
    s.set(xlabel='Predicted Labels', ylabel='True Labels', title=f"\u03BA = {test_kappa:0.3f} (n = {len(test_dataloader.dataset)})")
    plt.tight_layout()
    if not os.path.exists(os.path.join(results_dir, checkpoint)):
        os.makedirs(os.path.join(results_dir, checkpoint))
    plt.savefig(os.path.join(results_dir, checkpoint, f"{date}_{time}.png"))

    plt.show()