import os
import json
import random
import re
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, multilabel_confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import spacy
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import nltk
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
import joblib

nltk.download('stopwords', quiet=True)
nltk.download('wordnet', quiet=True)

class SalesCallClassifier:
    def __init__(self, domain_knowledge_path='domain_knowledge.json', dataset_path='calls_dataset.csv', model_path='classifier.pkl', vectorizer_path='vectorizer.pkl'):
        # Ensure domain knowledge file exists
        self._create_domain_knowledge_file(domain_knowledge_path)
        
        # Load domain knowledge
        with open(domain_knowledge_path, 'r') as f:
            self.domain_knowledge = json.load(f)
        
        # Initialize text processing tools
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))
        
        # Load SpaCy model
        self.nlp = spacy.load('en_core_web_sm')
        
        # Paths for model and vectorizer
        self.dataset_path = dataset_path
        self.model_path = model_path
        self.vectorizer_path = vectorizer_path

        # Load model and vectorizer if available
        if os.path.exists(self.model_path) and os.path.exists(self.vectorizer_path):
            self.model = joblib.load(self.model_path)
            self.vectorizer = joblib.load(self.vectorizer_path)
        else:
            self.model = None
            self.vectorizer = None

    def _create_domain_knowledge_file(self, path):
        """Create domain knowledge file if it doesn't exist"""
        if not os.path.exists(path):
            default_knowledge = {
                "competitors": ["CompetitorX", "CompetitorY", "CompetitorZ"],
                "features": ["analytics", "AI engine", "data pipeline"],
                "pricing_keywords": ["discount", "renewal cost", "budget", "pricing model"]
            }
            with open(path, 'w') as f:
                json.dump(default_knowledge, f, indent=4)

    def generate_synthetic_dataset(self, num_samples=150, output_csv='calls_dataset.csv'):
        """Generate synthetic sales call dataset and save to CSV."""
        label_options = ['Objection', 'Pricing Discussion', 'Security', 'Competition']

        data = []
        for i in range(num_samples):
            num_labels = random.randint(1, 3)
            selected_labels = random.sample(label_options, num_labels)

            text_templates = [
                f"We are considering {random.choice(self.domain_knowledge['competitors'])} due to their pricing concerns.",
                f"Can you provide a {random.choice(self.domain_knowledge['pricing_keywords'])} for your product features?",
                f"Our team is worried about {random.choice(['data security', 'compliance', 'integration'])}.",
                f"The {random.choice(self.domain_knowledge['features'])} is impressive, but how does it compare to competitors?",
                f"What discounts do you offer for renewals?"
            ]

            text_snippet = random.choice(text_templates)

            data.append({
                'id': i + 1,
                'text_snippet': text_snippet,
                'labels': ', '.join(selected_labels)
            })

        df = pd.DataFrame(data)
        df.to_csv(output_csv, index=False)
        print(f"Synthetic dataset saved to {output_csv}")

    def preprocess_text(self, text):
        """Advanced text preprocessing"""
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        tokens = text.split()
        cleaned_tokens = [
            self.lemmatizer.lemmatize(token) 
            for token in tokens 
            if token not in self.stop_words
        ]
        return ' '.join(cleaned_tokens)

    def extract_entities(self, text):
        """Extract entities using dictionary lookup and SpaCy NER"""
        # Dictionary-based extraction
        dict_entities = {category: [] for category in self.domain_knowledge}
        for category, keywords in self.domain_knowledge.items():
            for keyword in keywords:
                if keyword.lower() in text.lower():
                    dict_entities[category].append(keyword)

        # NER-based extraction using SpaCy
        doc = self.nlp(text)
        ner_entities = {ent.label_: [] for ent in doc.ents}
        for ent in doc.ents:
            ner_entities[ent.label_].append(ent.text)

        # Combine results
        return {
            'dictionary_entities': dict_entities,
            'ner_entities': ner_entities
        }

    def evaluate_model(self, y_test, y_pred, label_names):
        """Evaluate model performance and visualize confusion matrix"""
        print("Classification Report:\n")
        print(classification_report(y_test, y_pred, target_names=label_names))

        # Generate confusion matrices for each label
        cm = multilabel_confusion_matrix(y_test, y_pred)

        # Plot confusion matrix heatmaps
        plt.figure(figsize=(12, 8))
        for i, label in enumerate(label_names):
            plt.subplot(2, 2, i + 1)
            sns.heatmap(cm[i], annot=True, fmt='d', cmap='Blues', 
                        xticklabels=['Negative', 'Positive'],
                        yticklabels=['Negative', 'Positive'])
            plt.title(f'Confusion Matrix - {label}')

        plt.tight_layout()
        plt.savefig('confusion_matrix.png')
        print("Confusion matrix heatmaps saved as confusion_matrix.png")

    def train(self):
        """Train the model, evaluate it, and save it"""
        df = pd.read_csv(self.dataset_path)
        df['processed_text'] = df['text_snippet'].apply(self.preprocess_text)

        # Transform text into TF-IDF features
        self.vectorizer = TfidfVectorizer(max_features=500)
        X = self.vectorizer.fit_transform(df['processed_text'])
        y = df['labels'].str.get_dummies(sep=', ')

        # Split the dataset
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train Multi-Label Classifier
        self.model = MultiOutputClassifier(LogisticRegression())
        self.model.fit(X_train, y_train)

        # Make predictions
        y_pred = self.model.predict(X_test)

        # Evaluate the model
        self.evaluate_model(y_test, y_pred, y.columns)

        # Save the model and vectorizer
        joblib.dump(self.model, self.model_path)
        joblib.dump(self.vectorizer, self.vectorizer_path)
        print(f"Model saved to {self.model_path}")
        print(f"Vectorizer saved to {self.vectorizer_path}")

    def predict(self, text):
        """Predict labels for a given text"""
        if not self.model or not self.vectorizer:
            raise ValueError("Model and vectorizer are not loaded or trained.")
        
        processed_text = self.preprocess_text(text)
        text_features = self.vectorizer.transform([processed_text])
        prediction = self.model.predict(text_features)
        return prediction

# Define FastAPI application
app = FastAPI()

# Pydantic model for input
class TextSnippet(BaseModel):
    text_snippet: str

@app.post("/predict")
async def predict_labels_and_entities(snippet: TextSnippet):
    try:
        # Initialize classifier and ensure the model and vectorizer are ready
        classifier = SalesCallClassifier()
        
        if classifier.model is None or classifier.vectorizer is None:
            # Train the model if it's not already trained
            classifier.train()

        # Preprocess the text
        processed_text = classifier.preprocess_text(snippet.text_snippet)

        # Predict labels
        label_names = ['Objection', 'Pricing Discussion', 'Security', 'Competition']
        predicted_label_values = classifier.predict(snippet.text_snippet)[0]  # Single prediction
        predicted_label_values = [int(val) for val in predicted_label_values]  # Ensure Python int
        predicted_labels = dict(zip(label_names, predicted_label_values))

        # Extract entities
        extracted_entities = classifier.extract_entities(snippet.text_snippet)

        # Return JSON-compatible response
        return {
            "processed_text": processed_text,
            "predicted_labels": predicted_labels,
            "extracted_entities": extracted_entities
        }

    except Exception as e:
        print(f"Error: {str(e)}")  # Log error details for debugging
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

if __name__ == "__main__":
    # Generate synthetic dataset and train the model
    classifier = SalesCallClassifier()
    classifier.generate_synthetic_dataset()  # Generate the CSV file
    classifier.train()  # Train the model
    uvicorn.run(app, host="0.0.0.0", port=8000)
