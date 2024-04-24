import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict  
from transformers import AutoTokenizer, DataCollatorWithPadding 

def train_val_test_split(reviews, val_size, test_size, seed):
    x, x_test, y, y_test = train_test_split(reviews[['comment']], reviews[['labels']], test_size=test_size, random_state=seed)
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=val_size/(1-test_size), random_state=seed, stratify=y)
    train_df = pd.DataFrame(np.concatenate((x_train, y_train), axis=1), columns=['comment', 'labels'])
    val_df = pd.DataFrame(np.concatenate((x_val, y_val), axis=1), columns=['comment', 'labels'])
    test_df = pd.DataFrame(np.concatenate((x_test, y_test), axis=1), columns=['comment', 'labels'])
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)
    tvt = DatasetDict({
        'train': train_dataset,
        'valid': val_dataset,
        'test': test_dataset
    })
    return tvt

def tokenize(tvt, checkpoint, num_classes):
    tokenizer = AutoTokenizer.from_pretrained(
        checkpoint, 
        num_labels=num_classes
    )

    def preprocess_function(examples):
        return tokenizer(examples['comment'], truncation=True)

    tokenized_rating = tvt.map(preprocess_function, batched=True)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_rating = tokenized_rating.remove_columns(['comment'])
    tokenized_rating.set_format('torch')

    return data_collator, tokenized_rating