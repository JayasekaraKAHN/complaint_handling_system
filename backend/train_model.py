#!/usr/bin/env python3
"""
Clean Model Training Script for Sri Lankan Telecom Complaint System
Uses only datafinal.csv to train the complaint classifier and solution matcher.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ComplaintModelTrainer:
    def __init__(self, data_file="data/datafinal.csv"):
        self.data_file = data_file
        self.model = None
        self.categories = []
        self.solution_patterns = {}
        
    def load_data(self):
        """Load and clean data from datafinal.csv"""
        logger.info(f"Loading data from {self.data_file}...")
        
        try:
            # Try different encodings
            try:
                df = pd.read_csv(self.data_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.data_file, encoding='latin1')
            
            logger.info(f"Loaded {len(df)} records")
            
            # Keep only essential columns for training
            required_columns = ['Issue_Description', 'Solution']
            optional_columns = ['Device_type_settings_VPN_APN', 'Site_KPI_Alarm', 'Signal_Strength']
            
            # Check which columns exist
            available_columns = [col for col in required_columns + optional_columns if col in df.columns]
            df = df[available_columns]
            
            # Remove rows with missing Issue_Description or Solution
            df = df.dropna(subset=['Issue_Description', 'Solution'])
            
            logger.info(f"After cleaning: {len(df)} valid records")
            logger.info(f"Available columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def preprocess_data(self, df):
        """Preprocess data for training"""
        logger.info("Preprocessing data...")
        
        features = []
        labels = []
        solutions = []
        
        for idx, row in df.iterrows():
            try:
                # Main complaint text
                issue_text = str(row['Issue_Description']).strip()
                
                # Additional context if available
                context_parts = [issue_text]
                
                if 'Device_type_settings_VPN_APN' in df.columns:
                    device_info = str(row['Device_type_settings_VPN_APN']).strip()
                    if device_info and device_info != 'nan':
                        context_parts.append(device_info)
                
                if 'Site_KPI_Alarm' in df.columns:
                    site_info = str(row['Site_KPI_Alarm']).strip()
                    if site_info and site_info != 'nan':
                        context_parts.append(site_info)
                
                if 'Signal_Strength' in df.columns:
                    signal_info = str(row['Signal_Strength']).strip()
                    if signal_info and signal_info != 'nan':
                        context_parts.append(signal_info)
                
                # Combine all context
                full_context = ' '.join(context_parts)
                
                # Solution
                solution = str(row['Solution']).strip()
                
                if full_context and solution and solution != 'nan':
                    features.append(full_context)
                    solutions.append(solution)
                    
                    # Create category label
                    category = self.categorize_complaint(full_context)
                    labels.append(category)
                
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                continue
        
        logger.info(f"Preprocessed {len(features)} training samples")
        return features, labels, solutions
    
    def categorize_complaint(self, text):
        """Categorize complaints based on content"""
        text_lower = text.lower()
        
        # Voice and calling issues
        if any(keyword in text_lower for keyword in 
               ['voice call', 'call drop', 'call issue', 'volte', 'vowifi', 'calling']):
            return 'voice_call'
        
        # Data and internet issues
        if any(keyword in text_lower for keyword in 
               ['data', 'speed', 'internet', 'slow', 'browsing', '4g', '3g', '2g']):
            return 'data_internet'
        
        # Coverage and signal issues
        if any(keyword in text_lower for keyword in 
               ['coverage', 'signal', 'weak signal', 'no signal', 'poor coverage']):
            return 'coverage_signal'
        
        # Site and infrastructure issues
        if any(keyword in text_lower for keyword in 
               ['site', 'alarm', 'unavailability', 'site down', 'bts', 'cell']):
            return 'site_infrastructure'
        
        # Device and hardware issues
        if any(keyword in text_lower for keyword in 
               ['device', 'router', 'dongle', 'sim', 'hardware', 'phone']):
            return 'device_hardware'
        
        # Network configuration issues
        if any(keyword in text_lower for keyword in 
               ['configuration', 'settings', 'apn', 'vpn', 'network mode']):
            return 'network_config'
        
        # Default category
        return 'general_technical'
    
    def build_solution_patterns(self, features, solutions):
        """Build solution patterns for quick matching"""
        logger.info("Building solution patterns...")
        
        # Create exact match patterns
        for feature, solution in zip(features, solutions):
            feature_key = feature.lower().strip()
            self.solution_patterns[feature_key] = solution
        
        # Create keyword-based patterns
        keyword_solutions = {}
        for feature, solution in zip(features, solutions):
            # Extract key terms from complaints
            words = feature.lower().split()
            for word in words:
                if len(word) > 4:  # Skip short words
                    if word not in keyword_solutions:
                        keyword_solutions[word] = []
                    keyword_solutions[word].append(solution)
        
        # Keep most common solution for each keyword
        for keyword, sols in keyword_solutions.items():
            if len(sols) > 1:  # Only if keyword appears multiple times
                most_common = max(set(sols), key=sols.count)
                self.solution_patterns[f"keyword_{keyword}"] = most_common
        
        logger.info(f"Built {len(self.solution_patterns)} solution patterns")
    
    def train_classifier(self, features, labels):
        """Train the complaint classifier"""
        logger.info("Training classifier...")
        
        self.categories = sorted(list(set(labels)))
        logger.info(f"Categories: {self.categories}")
        
        # Check class distribution
        from collections import Counter
        label_counts = Counter(labels)
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Prepare data for training
        if len(features) < 10:
            logger.warning("Limited training data - using all for training")
            X_train, X_test = features, features[:3] if len(features) > 3 else features
            y_train, y_test = labels, labels[:3] if len(labels) > 3 else labels
        else:
            # Split data
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.25, random_state=42, stratify=labels
                )
            except ValueError:
                # Fallback if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.25, random_state=42
                )
        
        # Create and train pipeline
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=3000,
                ngram_range=(1, 2),
                stop_words='english',
                lowercase=True,
                min_df=1,
                max_df=0.95
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train
        self.model.fit(X_train, y_train)
        
        # Evaluate
        if len(X_test) > 0:
            y_pred = self.model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            
            logger.info(f"Training completed - Accuracy: {accuracy:.3f}")
            logger.info("Classification Report:")
            print(classification_report(y_test, y_pred, zero_division=0))
        else:
            accuracy = 1.0
            logger.info("Training completed on full dataset")
        
        return accuracy
    
    def save_models(self):
        """Save trained models and patterns"""
        os.makedirs("models", exist_ok=True)
        
        # Save classifier
        classifier_path = "models/complaint_classifier.pkl"
        with open(classifier_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save solution patterns
        patterns_path = "models/solution_patterns.pkl"
        with open(patterns_path, 'wb') as f:
            pickle.dump(self.solution_patterns, f)
        
        # Save metadata
        metadata = {
            'categories': self.categories,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.solution_patterns),
            'model_version': '1.0'
        }
        
        metadata_path = "models/model_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Models saved:")
        logger.info(f"  - Classifier: {classifier_path}")
        logger.info(f"  - Patterns: {patterns_path}")
        logger.info(f"  - Metadata: {metadata_path}")
    
    def train_complete_model(self):
        """Complete training workflow"""
        logger.info("🚀 Starting model training...")
        
        # Load data
        df = self.load_data()
        
        # Preprocess
        features, labels, solutions = self.preprocess_data(df)
        
        if len(features) == 0:
            raise ValueError("No valid training data found!")
        
        # Build solution patterns
        self.build_solution_patterns(features, solutions)
        
        # Train classifier
        accuracy = self.train_classifier(features, labels)
        
        # Save models
        self.save_models()
        
        results = {
            'accuracy': accuracy,
            'samples': len(features),
            'categories': len(self.categories),
            'patterns': len(self.solution_patterns)
        }
        
        logger.info("✅ Training completed successfully!")
        return results

def main():
    """Main training function"""
    print("🔧 Complaint Handling System - Model Training")
    print("=" * 50)
    
    try:
        trainer = ComplaintModelTrainer()
        results = trainer.train_complete_model()
        
        print("\n📊 Training Results:")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Training Samples: {results['samples']}")
        print(f"   Categories: {results['categories']}")
        print(f"   Solution Patterns: {results['patterns']}")
        print("\n🎯 Models ready for use!")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
