import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import (
    BertTokenizer, 
    BertForSequenceClassification, 
    get_linear_schedule_with_warmup
)
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score
)
import re
import logging

# Disable excessive warnings
logging.getLogger("transformers").setLevel(logging.ERROR)

# Check GPU availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Menggunakan perangkat: {device}")

# Text preprocessing functions
def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    text = ' '.join(text.split())
    return text

# Sub-theme categorization
themes = ['komunikasi', 'jaringan', 'peluang bisnis', 'budaya', 'karier']
def categorize_theme(comment):
    keywords = {
       'komunikasi': ['berkomunikasi', 'berbicara', 'berdiskusi'],
       'jaringan': ['jaringan', 'menghubungkan', 'berkolaborasi'],
       'peluang bisnis': ['kesempatan', 'bisnis', 'pasar'],
       'budaya': ['budaya', 'adat istiadat', 'tradisi'],
       'karier': ['karier', 'pekerjaan', 'profesi']
    }
    for theme, keywords_list in keywords.items():
        if any(keyword in comment.lower() for keyword in keywords_list):
            return theme
    return 'general'

# Class Custom Dataset
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = str(self.texts[idx])
        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_token_type_ids=True
        )
        
        return {
            'input_ids': torch.tensor(inputs['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(inputs['attention_mask'], dtype=torch.long),
            'token_type_ids': torch.tensor(inputs['token_type_ids'], dtype=torch.long),
            'labels': torch.tensor(self.labels[idx], dtype=torch.long)
        }

# Training
def train_model(model, train_loader, val_loader, optimizer, scheduler, device, epochs=3):
    model.to(device)
    
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
        
        # Model evaluation
        model.eval()
        val_loss = 0
        predictions, true_labels = [], []
        
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch['token_type_ids'].to(device)
                labels = batch['labels'].to(device)
                
                outputs = model(
                    input_ids, 
                    attention_mask=attention_mask, 
                    token_type_ids=token_type_ids,
                    labels=labels
                )
                
                val_loss += outputs.loss.item()
                
                logits = outputs.logits
                preds = torch.argmax(logits, dim=1)
                
                predictions.extend(preds.cpu().numpy())
                true_labels.extend(labels.cpu().numpy())
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"Train Loss: {total_loss/len(train_loader):.4f}")
        print(f"Validation Loss: {val_loss/len(val_loader):.4f}")
        print(classification_report(true_labels, predictions))
    
    return model

# Main functions
def main():
    # Read dataset
    df = pd.read_excel('dataset.xlsx')
    
    # Preprocessing
    df['Komentar_Cleaned'] = df['Komentar'].apply(preprocess_text)
    
    # Sub-theme categorization
    df['Theme'] = df['Komentar_Cleaned'].apply(categorize_theme)
    
    # Encode label
    df['Label_Encoded'] = df['Sentimen (Label)'].map({'Positif': 1, 'Negatif': 0})
    
    # Split dataset
    X_train, X_val, y_train, y_val = train_test_split(
        df['Komentar_Cleaned'], 
        df['Label_Encoded'], 
        test_size=0.2, 
        random_state=42,
        stratify=df['Label_Encoded']  # Ensure stratified split for balanced classes
    )
    
    # Tokenizer initialization
    model_name = 'indolem/indobert-base-uncased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    
    # Parameter
    MAX_LEN = 128
    BATCH_SIZE = 16
    EPOCHS = 3
    
    # Create a dataset
    train_dataset = SentimentDataset(
        texts=X_train.values, 
        labels=y_train.values, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN
    )
    val_dataset = SentimentDataset(
        texts=X_val.values, 
        labels=y_val.values, 
        tokenizer=tokenizer, 
        max_len=MAX_LEN
    )
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)
    
    # Model initialization
    model = BertForSequenceClassification.from_pretrained(
        model_name, 
        num_labels=2
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=2e-5)
    total_steps = len(train_loader) * EPOCHS
    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=0, 
        num_training_steps=total_steps
    )
    
    # Train the model
    trained_model = train_model(
        model, 
        train_loader, 
        val_loader, 
        optimizer, 
        scheduler, 
        device, 
        epochs=EPOCHS
    )
    
    # Final evaluation
    model.eval()
    predictions, true_labels = [], []
    
    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(
                input_ids, 
                attention_mask=attention_mask, 
                token_type_ids=token_type_ids
            )
            
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1)
            
            predictions.extend(preds.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())
    
    # Visualization of Results
    plt.figure(figsize=(12, 8))
    
    # Confusion Matrix
    cm = confusion_matrix(true_labels, predictions)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Confusion Matrix')
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    
    # Report Classification
    report = classification_report(true_labels, predictions, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='YlGnBu')
    plt.title('Klasifikasi Metrik')
    
    plt.tight_layout()
    plt.show()
    
    # Print detailed classification report
    print("\nLaporan Klasifikasi:")
    print(classification_report(true_labels, predictions))
    
    # Theme visualization
    theme_counts = df['Theme'].value_counts()
    plt.figure(figsize=(10, 6))
    plt.bar(theme_counts.index, theme_counts.values)
    plt.title('Distribution of Themes')
    plt.xlabel('Theme')
    plt.ylabel('Count')
    plt.show()
    
    # Word Cloud for positive and negative sentiment
    from wordcloud import WordCloud
    positive_comments = df[df['Label_Encoded'] == 1]['Komentar_Cleaned']
    negative_comments = df[df['Label_Encoded'] == 0]['Komentar_Cleaned']
    positive_wordcloud = WordCloud(width=800, height=400).generate(' '.join(positive_comments))
    negative_wordcloud = WordCloud(width=800, height=400).generate(' '.join(negative_comments))
    plt.figure(figsize=(10, 6))
    plt.imshow(positive_wordcloud, interpolation='bilinear')
    plt.title('Positive Sentiment Word Cloud')
    plt.axis('off')
    plt.show()
    plt.figure(figsize=(10, 6))
    plt.imshow(negative_wordcloud, interpolation='bilinear')
    plt.title('Negative Sentiment Word Cloud')
    plt.axis('off')
    plt.show()
    
    # Correlation analysis of themes and sentiments
    sentiment_distribution = df.groupby('Theme')['Label_Encoded'].value_counts().unstack().fillna(0)
    plt.figure(figsize=(10, 6))
    sentiment_distribution.plot(kind='bar')
    plt.title('Sentiment Distribution per Theme')
    plt.xlabel('Theme')
    plt.ylabel('Count')
    plt.show()


if __name__ == "__main__":
    main()
