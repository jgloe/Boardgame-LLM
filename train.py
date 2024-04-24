import pandas as pd
import numpy as np
import random
import datetime
from preprocess import train_val_test_split, tokenize
from sklearn.metrics import cohen_kappa_score
import torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, LongformerForSequenceClassification, get_scheduler
from torch.optim import AdamW
import evaluate
from tqdm.auto import tqdm

import ssl
import os
ssl._create_default_https_context = ssl._create_unverified_context
os.environ['HF_HUB_DISABLE_SYMLINKS_WARNING'] = '1'

def train(model, loader, device, optimizer, scheduler, progress_bar):
    model.train()
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()} 
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()
        progress_bar.update(1)

def validate(model, loader, device, accuracy):
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
    train_batch_size = 8
    val_batch_size = 8
    learning_rate = 5e-5
    num_epochs = 5
    save_dir = './saved_models'
    data_file = './bgg-19m-reviews.csv'
    checkpoint = 'bert-base-uncased'
    # checkpoint = "jpelhaw/longformer-base-plagiarism-detection"
    pretrain_path = os.path.join(save_dir, checkpoint, '2024-04-23_14-33.pth.tar') # If not using previous checkpoint, pretrain_path = None
    random_seed = 123

    torch.manual_seed(random_seed)
    random.seed(random_seed)
    np.random.seed(random_seed)

    date_time = str(datetime.datetime.now()).split(' ')
    date = date_time[0]
    time_split = date_time[1].split(':')
    time = time_split[0] + '-' + time_split[1]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

    # DataLoaders
    train_dataloader = DataLoader(
        tokenized_rating['train'], shuffle=True, batch_size=train_batch_size, collate_fn=data_collator
    )
    val_dataloader = DataLoader(
        tokenized_rating['valid'], shuffle=False, batch_size=val_batch_size, collate_fn=data_collator
    )

    # Create evaluation metrics
    accuracy = evaluate.load('accuracy')
    
    # Create model
    if not pretrain_path:
        model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, num_labels=num_classes
        )
    else:
        fine_tune = torch.load(pretrain_path, map_location=device)
        model = fine_tune['model']
        model.load_state_dict(fine_tune['state_dict'])
    model.to(device)
    print('Created model')

    # Optimizer and LR Scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        'linear',
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_epochs * len(train_dataloader)
    )

    # Training loop over epochs
    best_metric = 0
    acc_results, kappa_results = [], []

    print('Begin training...')
    progress_bar = tqdm(range(num_epochs * len(train_dataloader)))
    for epoch in range(num_epochs):
        
        # Training
        train(model, train_dataloader, device, optimizer, lr_scheduler, progress_bar)

        # Evaluation
        predictions, y_true = validate(model, val_dataloader, device, accuracy)

        acc_results.append(accuracy.compute()['accuracy'])
        kappa_results.append(cohen_kappa_score(predictions, y_true, weights='quadratic'))
        print(f"\n Accuracy: {100*acc_results[-1]:0.2f}% \n Cohen's kappa: {kappa_results[-1]:0.3f}")

        if acc_results[-1] > best_metric:
            best_metric = acc_results[-1]
            model_save_path = os.path.join(save_dir, checkpoint, f"{date}_{time}.pth.tar")
            if not os.path.exists(os.path.join(save_dir, checkpoint)):
                os.makedirs(os.path.join(save_dir, checkpoint))
            torch.save({'model': model,
                        'state_dict': model.state_dict(),
                        'epoch': epoch,
                        'batch_size': (train_batch_size, val_batch_size),
                        'optimizer': optimizer.state_dict(),
                        'val_accuracy': acc_results,
                        'val_ck': kappa_results}, model_save_path)
            
    print('Training finished')


