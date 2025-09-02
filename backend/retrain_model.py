#!/usr/bin/env python3
"""
Comprehensive Model Retraining Script for Sri Lankan Telecom Complaint System
Uses datafinal.csv to retrain the complaint classifier and solution matcher.
"""

import pandas as pd
import numpy as np
import os
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
import json
from datetime import datetime
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ComplaintModelRetrainer:
    def __init__(self, data_file="data/datafinal.csv"):
        self.data_file = data_file
        self.model = None
        self.vectorizer = None
        self.categories = []
        self.solution_mappings = {}
        self.device_patterns = {}
        self.site_patterns = {}
        
    def load_and_preprocess_data(self):
        """Load and preprocess the datafinal.csv data"""
        logger.info("Loading datafinal.csv data...")
        
        try:
            # Try different encodings
            try:
                df = pd.read_csv(self.data_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.data_file, encoding='latin1')
            
            logger.info(f"Loaded {len(df)} records from datafinal.csv")
            
            # Clean and prepare the data
            df = df.dropna(subset=['Issue_Description', 'Solution'])
            
            # Extract key features for training
            features = []
            labels = []
            solutions = []
            
            for idx, row in df.iterrows():
                try:
                    # Combine multiple text fields for richer feature extraction
                    issue_text = str(row['Issue_Description'])
                    device_info = str(row.get('Device_type_settings_VPN_APN', ''))
                    site_info = str(row.get('Site_KPI_Alarm', ''))
                    signal_info = str(row.get('Signal_Strength', ''))
                    
                    # Create comprehensive feature text
                    feature_text = f"{issue_text} {device_info} {site_info} {signal_info}".strip()
                    
                    # Extract solution
                    solution = str(row['Solution']).strip()
                    
                    if feature_text and solution and solution != 'nan':
                        features.append(feature_text)
                        solutions.append(solution)
                        
                        # Create category labels based on issue types
                        category = self.categorize_issue(issue_text, device_info, site_info)
                        labels.append(category)
                        
                except Exception as e:
                    logger.warning(f"Skipping row {idx}: {e}")
                    continue
            
            logger.info(f"Processed {len(features)} valid training samples")
            
            return features, labels, solutions
            
        except Exception as e:
            logger.error(f"Error loading data: {e}")
            raise
    
    def categorize_issue(self, issue_text, device_info, site_info):
        """Categorize issues based on content analysis"""
        issue_lower = issue_text.lower()
        device_lower = device_info.lower()
        site_lower = site_info.lower()
        
        # Site and infrastructure issues
        if any(keyword in site_lower for keyword in ['alarm', 'unavailability', 'site down', 'abnormal kpi']):
            return 'site_infrastructure'
        
        # Voice call issues
        if any(keyword in issue_lower for keyword in ['voice call', 'call drop', 'call issue', 'volte', 'vowifi']):
            return 'voice_call'
        
        # Data and coverage issues
        if any(keyword in issue_lower for keyword in ['data', 'speed', 'internet', 'coverage', 'signal']):
            if 'indoor' in issue_lower or 'building' in issue_lower:
                return 'indoor_coverage'
            else:
                return 'data_coverage'
        
        # Device specific issues
        if any(keyword in device_lower for keyword in ['router', 'dongle', 'sim']):
            return 'device_hardware'
        
        # Default category
        return 'general_technical'
    
    def extract_patterns(self, features, labels, solutions):
        """Extract patterns for exact matching and rules"""
        logger.info("Extracting patterns from data...")
        
        # Create solution mappings
        for feature, solution in zip(features, solutions):
            feature_key = feature.lower().strip()
            if feature_key not in self.solution_mappings:
                self.solution_mappings[feature_key] = solution
        
        # Extract device patterns
        device_solutions = {}
        for feature, solution in zip(features, solutions):
            # Extract device type mentions
            for device in ['s10 router', 'huawei router', 'mobile device', 'dongle', 'iphone', 'samsung']:
                if device in feature.lower():
                    if device not in device_solutions:
                        device_solutions[device] = []
                    device_solutions[device].append(solution)
        
        # Store most common solutions for each device
        for device, sols in device_solutions.items():
            self.device_patterns[device] = max(set(sols), key=sols.count)
        
        # Extract site patterns
        site_solutions = {}
        for feature, solution in zip(features, solutions):
            # Extract site mentions
            if 'klpor5' in feature.lower():
                site_solutions['klpor5'] = solution
            elif 'klpet1' in feature.lower():
                site_solutions['klpet1'] = solution
        
        self.site_patterns = site_solutions
        
        logger.info(f"Extracted {len(self.solution_mappings)} solution mappings")
        logger.info(f"Extracted {len(self.device_patterns)} device patterns")
        logger.info(f"Extracted {len(self.site_patterns)} site patterns")
    
    def train_classifier(self, features, labels):
        """Train the complaint classifier"""
        logger.info("Training complaint classifier...")
        
        # Get unique categories
        self.categories = list(set(labels))
        logger.info(f"Training on {len(self.categories)} categories: {self.categories}")
        
        # Check class distribution
        from collections import Counter
        label_counts = Counter(labels)
        logger.info(f"Label distribution: {label_counts}")
        
        # Filter out categories with only 1 sample for stratified split
        min_samples_per_class = 2
        valid_samples = []
        valid_labels = []
        
        for feature, label in zip(features, labels):
            if label_counts[label] >= min_samples_per_class:
                valid_samples.append(feature)
                valid_labels.append(label)
        
        logger.info(f"Using {len(valid_samples)} samples with sufficient class representation")
        
        # If we have too few samples for splitting, use all for training
        if len(valid_samples) < 10:
            logger.warning("Limited samples - using all data for training")
            X_train, X_test = valid_samples, valid_samples[:5] if len(valid_samples) > 5 else valid_samples
            y_train, y_test = valid_labels, valid_labels[:5] if len(valid_labels) > 5 else valid_labels
        else:
            # Split data with stratification if possible
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    valid_samples, valid_labels, test_size=0.2, random_state=42, stratify=valid_labels
                )
            except ValueError:
                # Fall back to random split if stratification fails
                X_train, X_test, y_train, y_test = train_test_split(
                    valid_samples, valid_labels, test_size=0.2, random_state=42
                )
        
        # Create pipeline with TF-IDF and Naive Bayes
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True
            )),
            ('classifier', MultinomialNB(alpha=0.1))
        ])
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        logger.info(f"Model training completed with accuracy: {accuracy:.3f}")
        logger.info("\\nClassification Report:")
        print(classification_report(y_test, y_pred, zero_division=0))
        
        return accuracy
    
    def save_model(self, model_dir="models"):
        """Save the trained model and patterns"""
        os.makedirs(model_dir, exist_ok=True)
        
        # Save the classifier model
        model_path = os.path.join(model_dir, "complaint_classifier.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save patterns and mappings
        patterns = {
            'categories': self.categories,
            'solution_mappings': self.solution_mappings,
            'device_patterns': self.device_patterns,
            'site_patterns': self.site_patterns,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.solution_mappings)
        }
        
        patterns_path = os.path.join(model_dir, "patterns.json")
        with open(patterns_path, 'w') as f:
            json.dump(patterns, f, indent=2)
        
        logger.info(f"Model saved to {model_path}")
        logger.info(f"Patterns saved to {patterns_path}")
    
    def update_prompts_file(self):
        """Update prompts.py with new patterns from training"""
        logger.info("Updating prompts.py with new training patterns...")
        
        # Create new device rules
        device_rules = []
        for device, solution in self.device_patterns.items():
            device_rules.append(f"- {device.title()} -> {solution}")
        
        # Create new site rules
        site_rules = []
        for site, solution in self.site_patterns.items():
            site_rules.append(f'- "{site.upper()}" issues -> {solution}')
        
        # Create solution frequency analysis
        solution_freq = {}
        for solution in self.solution_mappings.values():
            solution_freq[solution] = solution_freq.get(solution, 0) + 1
        
        # Sort by frequency
        top_solutions = sorted(solution_freq.items(), key=lambda x: x[1], reverse=True)
        
        logger.info("Top 10 most frequent solutions:")
        for i, (solution, freq) in enumerate(top_solutions[:10]):
            logger.info(f"{i+1}. {solution} ({freq} cases)")
        
        # Generate training summary
        summary = f"""
# Model Retraining Summary - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Training Statistics:
- Total samples: {len(self.solution_mappings)}
- Categories: {len(self.categories)}
- Device patterns: {len(self.device_patterns)}
- Site patterns: {len(self.site_patterns)}

## Top Solutions by Frequency:
{chr(10).join([f"{i+1}. {sol} ({freq} cases)" for i, (sol, freq) in enumerate(top_solutions[:10])])}

## Device Patterns Discovered:
{chr(10).join(device_rules)}

## Site Patterns Discovered:
{chr(10).join(site_rules)}
"""
        
        # Save summary
        with open("RETRAINING_SUMMARY.md", 'w', encoding='utf-8') as f:
            f.write(summary)
        
        logger.info("Training summary saved to RETRAINING_SUMMARY.md")
    
    def retrain_complete_system(self):
        """Complete retraining workflow"""
        logger.info("Starting complete system retraining...")
        
        # Load and preprocess data
        features, labels, solutions = self.load_and_preprocess_data()
        
        # Extract patterns
        self.extract_patterns(features, labels, solutions)
        
        # Train classifier
        accuracy = self.train_classifier(features, labels)
        
        # Save model and patterns
        self.save_model()
        
        # Update prompts
        self.update_prompts_file()
        
        logger.info(f"Retraining completed successfully with {accuracy:.3f} accuracy!")
        
        return {
            'accuracy': accuracy,
            'samples': len(features),
            'categories': len(self.categories),
            'patterns': len(self.solution_mappings)
        }

def main():
    """Main retraining function"""
    print("🔄 Sri Lankan Telecom Complaint System - Model Retraining")
    print("=" * 60)
    
    # Initialize retrainer
    retrainer = ComplaintModelRetrainer()
    
    # Run complete retraining
    results = retrainer.retrain_complete_system()
    
    print("\\n✅ Retraining Results:")
    print(f"   📊 Accuracy: {results['accuracy']:.3f}")
    print(f"   📝 Training Samples: {results['samples']}")
    print(f"   🏷️  Categories: {results['categories']}")
    print(f"   🔍 Patterns: {results['patterns']}")
    print("\\n🎯 Model ready for deployment!")

if __name__ == "__main__":
    main()
