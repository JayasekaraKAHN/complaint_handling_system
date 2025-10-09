import pandas as pd
import os
import pickle
import json
from datetime import datetime

class UnifiedModelTrainer:
    def __init__(self):
        self.complaint_data = None
        self.location_data = None
        self.model_path = "models/unified_complaint_model.pkl"
    
    def clean_dataframe_for_json(self, df):
        """
        Clean DataFrame by replacing NaN, inf, and other non-JSON-compliant values
        """
        if df is None:
            return None
        
        # Create a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Replace NaN values with None (which becomes null in JSON)
        cleaned_df = cleaned_df.where(pd.notna(cleaned_df), None)
        
        # Replace infinite values with None
        import numpy as np
        cleaned_df = cleaned_df.replace([np.inf, -np.inf], None)
        
        return cleaned_df
        
    def load_data(self):
        print("Loading data from Excel files...")
        
        try:
            # Load complaint data
            complaint_file = "data/Information_Gathering_form.xlsx"
            if os.path.exists(complaint_file):
                self.complaint_data = pd.read_excel(complaint_file)
                print(f"Loaded {len(self.complaint_data)} complaint records")
                
                # Clean complaint data
                self.complaint_data = self.complaint_data.dropna(subset=['Issue Description', 'Solution'])
                self.complaint_data.columns = self.complaint_data.columns.str.strip()
                
                print(f"After cleaning: {len(self.complaint_data)} valid complaint records")
            else:
                print(f"Complaint file {complaint_file} not found!")
                return False
            
            # Load location data
            location_file = "data/location_data_mapping.xlsx"
            if os.path.exists(location_file):
                self.location_data = pd.read_excel(location_file)
                print(f"Loaded {len(self.location_data)} location records")
                
                # Clean location data
                self.location_data = self.location_data.dropna(subset=['Site Name'])
                self.location_data.columns = self.location_data.columns.str.strip()
                
                print(f"After cleaning: {len(self.location_data)} valid location records")
            else:
                print(f"Location file {location_file} not found!")
                return False
            
            return True
            
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def save_unified_model(self):
        print("Saving unified model...")

        # Create models directory
        os.makedirs("models", exist_ok=True)

        # Enhance complaint data with VoLTE status and usage analytics if available
        if self.complaint_data is not None:
            enhanced_complaint_data = self.complaint_data.copy()
            if not enhanced_complaint_data.empty:
                from app.msisdn_dashboard import get_usage_data, check_msisdn_volte_provisioning
                # Add columns for VoLTE status and usage analytics
                if 'MSISDN' in enhanced_complaint_data.columns:
                    enhanced_complaint_data['VoLTE_Status'] = enhanced_complaint_data['MSISDN'].apply(
                        lambda msisdn: check_msisdn_volte_provisioning(msisdn) if pd.notna(msisdn) else {}
                    )
                    enhanced_complaint_data['Usage_Analytics'] = enhanced_complaint_data['MSISDN'].apply(
                        lambda msisdn: get_usage_data(msisdn) if pd.notna(msisdn) else {}
                    )
                
                # Clean NaN values to avoid JSON serialization issues
                enhanced_complaint_data = self.clean_dataframe_for_json(enhanced_complaint_data)
        else:
            enhanced_complaint_data = self.complaint_data
        
        # Clean location data as well
        cleaned_location_data = self.clean_dataframe_for_json(self.location_data) if self.location_data is not None else self.location_data

        # Create unified model data
        model_data = {
            'complaint_data': enhanced_complaint_data,
            'location_data': cleaned_location_data,
            'metadata': {
                'created_date': datetime.now().isoformat(),
                'complaint_records': len(enhanced_complaint_data) if enhanced_complaint_data is not None else 0,
                'location_records': len(cleaned_location_data) if cleaned_location_data is not None else 0,
                'model_type': 'unified_simple_model',
                'version': '1.1',
                'includes_volte_status': True,
                'includes_usage_analytics': True
            }
        }

        # Save the unified model
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)

        print(f"Unified model saved to {self.model_path}")
        return True
    
    def train_nlp_model(self):
        """
        Enhance model using TF-IDF and cosine similarity for more specified answers.
        Trains on complaint data Issue Description and Solution.
        """
        print("Training NLP model for specified answers...")
        try:
            from sklearn.feature_extraction.text import TfidfVectorizer
            from sklearn.metrics.pairwise import cosine_similarity
        except ImportError:
            print("scikit-learn not installed. Please install it to use NLP enhancements.")
            return False

        if self.complaint_data is None or self.complaint_data.empty:
            print("Complaint data not loaded.")
            return False

        # Prepare data
        issues = self.complaint_data['Issue Description'].astype(str).tolist()
        solutions = self.complaint_data['Solution'].astype(str).tolist()

        # Train TF-IDF vectorizer
        vectorizer = TfidfVectorizer(stop_words='english')
        X = vectorizer.fit_transform(issues)

        # Save NLP model and vectorizer
        nlp_model = {
            'vectorizer': vectorizer,
            'tfidf_matrix': X,
            'issues': issues,
            'solutions': solutions
        }
        with open('models/nlp_complaint_model.pkl', 'wb') as f:
            pickle.dump(nlp_model, f)
        print("NLP model saved to models/nlp_complaint_model.pkl")
        return True

    def train(self):
        print("Starting unified model training...")
        
        # Load data
        if not self.load_data():
            print("Failed to load data")
            return False
        
        # Save unified model
        if not self.save_unified_model():
            print("Failed to save model")
            return False
        
        # Train NLP model for enhanced answers
        self.train_nlp_model()
        
        print("Unified model training completed successfully!")
        
        # Print summary
        print("\n" + "="*50)
        print("TRAINING SUMMARY")
        print("="*50)
        if self.complaint_data is not None:
            print(f"Complaint Records: {len(self.complaint_data)}")
        if self.location_data is not None:
            print(f"Location Records: {len(self.location_data)}")
        print(f"Model saved to: {self.model_path}")
        print("="*50)
        
        return True

def main():
    trainer = UnifiedModelTrainer()
    success = trainer.train()
    
    if success:
        print("\nModel training completed successfully!")
        print("You can now run main.py to start the complaint handling system.")
    else:
        print("\nModel training failed!")
        print("Please check the logs and ensure data files are available.")

if __name__ == "__main__":
    main()