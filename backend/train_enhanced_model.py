#!/usr/bin/env python3
"""
Enhanced Model Training Script for Sri Lankan Telecom Complaint System
Incorporates location data (lon/lat), site information, and comprehensive pattern analysis.
"""

import pandas as pd
import numpy as np
import os
import pickle
import json
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from sklearn.pipeline import Pipeline
from datetime import datetime
import logging
from math import radians, cos, sin, asin, sqrt

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class EnhancedComplaintModelTrainer:
    def __init__(self, data_file="data/datafinal.csv"):
        self.data_file = data_file
        self.model = None
        self.categories = []
        self.solution_patterns = {}
        self.location_patterns = {}
        self.site_patterns = {}
        self.device_patterns = {}
        self.kpi_patterns = {}
        
    def parse_location(self, location_str):
        """Parse location string - handles coordinates, names, or tab-separated values"""
        if pd.isna(location_str) or not str(location_str).strip():
            return None, None, None
        
        location_str = str(location_str).strip()
        
        # Try to parse coordinates (lat/lon or lon/lat)
        coord_patterns = [
            r'([0-9.-]+)[\s\t,]+([0-9.-]+)',  # Space, tab, or comma separated
            r'lat[:\s]*([0-9.-]+)[\s,]*lon[:\s]*([0-9.-]+)',  # lat: x lon: y
            r'([0-9]{1,2}\.[0-9]+)[\s\t]+([0-9]{2,3}\.[0-9]+)'  # Typical lat lon format
        ]
        
        for pattern in coord_patterns:
            match = re.search(pattern, location_str, re.IGNORECASE)
            if match:
                coord1, coord2 = float(match.group(1)), float(match.group(2))
                # Determine lat/lon based on typical Sri Lankan coordinates
                if 5.0 <= coord1 <= 10.0 and 79.0 <= coord2 <= 82.0:
                    return coord1, coord2, "coordinates"  # lat, lon
                elif 5.0 <= coord2 <= 10.0 and 79.0 <= coord1 <= 82.0:
                    return coord2, coord1, "coordinates"  # lon, lat swapped
        
        # If no coordinates found, treat as location name
        return None, None, location_str.lower().strip()
    
    def categorize_by_location(self, lat, lon, location_name):
        """Categorize location for pattern analysis"""
        if lat and lon:
            # Major city regions based on coordinates
            if 6.8 <= lat <= 7.0 and 79.8 <= lon <= 80.2:
                return "colombo_metro"
            elif 7.2 <= lat <= 7.4 and 80.6 <= lon <= 80.8:
                return "kandy_region"
            elif 6.0 <= lat <= 6.2 and 80.2 <= lon <= 80.4:
                return "galle_south"
            else:
                return "other_region"
        
        if location_name:
            location_lower = location_name.lower()
            if any(city in location_lower for city in ['colombo', 'dehiwala', 'mount lavinia']):
                return "colombo_metro"
            elif any(city in location_lower for city in ['kandy', 'peradeniya']):
                return "kandy_region"
            elif any(city in location_lower for city in ['galle', 'matara', 'hambantota']):
                return "galle_south"
            elif any(city in location_lower for city in ['jaffna', 'vavuniya']):
                return "northern_region"
            else:
                return "other_region"
        
        return "unknown_location"
    
    def extract_site_info(self, site_text):
        """Extract site codes and patterns from site/alarm information"""
        if pd.isna(site_text) or not str(site_text).strip():
            return {}
        
        site_text = str(site_text).upper()
        site_info = {}
        
        # Extract site codes (pattern: 2-3 letters + 3-4 alphanumeric)
        site_codes = re.findall(r'[A-Z]{2,3}[A-Z0-9]{3,4}', site_text)
        if site_codes:
            site_info['site_codes'] = site_codes
            site_info['primary_site'] = site_codes[0]
        
        # Extract alarm types
        alarm_keywords = ['KPI', 'ALARM', 'UNAVAILABILITY', 'DOWN', 'ABNORMAL']
        found_alarms = [keyword for keyword in alarm_keywords if keyword in site_text]
        if found_alarms:
            site_info['alarm_types'] = found_alarms
        
        return site_info
    
    def load_and_analyze_data(self):
        """Load data and perform comprehensive analysis"""
        logger.info(f"Loading and analyzing data from {self.data_file}...")
        
        try:
            df = pd.read_csv(self.data_file, encoding='utf-8')
        except UnicodeDecodeError:
            df = pd.read_csv(self.data_file, encoding='latin1')
        
        logger.info(f"Loaded {len(df)} records with {len(df.columns)} columns")
        
        # Clean essential columns
        df = df.dropna(subset=['Issue_Description', 'Solution'])
        logger.info(f"After cleaning: {len(df)} valid records")
        
        features = []
        labels = []
        solutions = []
        location_data = []
        
        for idx, row in df.iterrows():
            try:
                # Core complaint text
                issue_text = str(row['Issue_Description']).strip()
                solution = str(row['Solution']).strip()
                
                if not issue_text or not solution or solution == 'nan':
                    continue
                
                # Build comprehensive feature text
                feature_parts = [issue_text]
                
                # Add device information
                if 'Device_type_settings_VPN_APN' in df.columns:
                    device_info = str(row['Device_type_settings_VPN_APN']).strip()
                    if device_info and device_info != 'nan':
                        feature_parts.append(f"Device: {device_info}")
                
                # Add site/alarm information
                site_info_text = ""
                if 'Site_KPI_Alarm' in df.columns:
                    site_alarm = str(row['Site_KPI_Alarm']).strip()
                    if site_alarm and site_alarm != 'nan':
                        feature_parts.append(f"Site: {site_alarm}")
                        site_info_text = site_alarm
                
                # Add signal strength
                if 'Signal_Strength' in df.columns:
                    signal = str(row['Signal_Strength']).strip()
                    if signal and signal != 'nan':
                        feature_parts.append(f"Signal: {signal}")
                
                # Parse location data
                location_text = ""
                if 'Lon' in df.columns and 'Lat' in df.columns:
                    lon_val = row.get('Lon', '')
                    lat_val = row.get('Lat', '')
                    if pd.notna(lon_val) and pd.notna(lat_val):
                        location_text = f"{lat_val}\t{lon_val}"
                elif 'Location' in df.columns:
                    location_text = str(row['Location']).strip()
                
                # Add district/area information
                if 'DISTRICT' in df.columns:
                    district = str(row['DISTRICT']).strip()
                    if district and district != 'nan':
                        feature_parts.append(f"District: {district}")
                
                # Add site name if available
                if 'SITE_NAME' in df.columns:
                    site_name = str(row['SITE_NAME']).strip()
                    if site_name and site_name != 'nan':
                        feature_parts.append(f"Site: {site_name}")
                
                # Combine all features
                full_feature = ' '.join(feature_parts)
                
                # Parse location for categorization
                lat, lon, location_name = self.parse_location(location_text)
                location_category = self.categorize_by_location(lat, lon, location_name)
                
                # Extract site information
                site_info = self.extract_site_info(site_info_text)
                
                # Store data
                features.append(full_feature)
                solutions.append(solution)
                location_data.append({
                    'lat': lat,
                    'lon': lon,
                    'location_name': location_name,
                    'location_category': location_category,
                    'site_info': site_info
                })
                
                # Enhanced categorization
                category = self.enhanced_categorize_complaint(
                    full_feature, location_category, site_info
                )
                labels.append(category)
                
            except Exception as e:
                logger.warning(f"Skipping row {idx}: {e}")
                continue
        
        logger.info(f"Processed {len(features)} training samples")
        return features, labels, solutions, location_data
    
    def enhanced_categorize_complaint(self, text, location_category, site_info):
        """Enhanced categorization considering location and site information"""
        text_lower = text.lower()
        
        # Site-specific issues (highest priority)
        if site_info.get('alarm_types'):
            alarm_types = site_info['alarm_types']
            if 'KPI' in alarm_types or 'ABNORMAL' in alarm_types:
                return 'site_kpi_issue'
            elif 'UNAVAILABILITY' in alarm_types or 'DOWN' in alarm_types:
                return 'site_down'
            else:
                return 'site_infrastructure'
        
        # Voice call issues
        if any(keyword in text_lower for keyword in 
               ['voice call', 'call drop', 'call issue', 'volte', 'vowifi', 'calling', 'voice quality']):
            return 'voice_call'
        
        # Data speed issues (location-aware)
        if any(keyword in text_lower for keyword in 
               ['data', 'speed', 'slow', 'internet', 'browsing', 'download', 'upload']):
            if location_category == 'colombo_metro':
                return 'data_speed_urban'
            else:
                return 'data_speed_rural'
        
        # Coverage issues (location-specific)
        if any(keyword in text_lower for keyword in 
               ['coverage', 'signal', 'weak signal', 'no signal', 'poor coverage']):
            if 'indoor' in text_lower or 'building' in text_lower:
                return 'indoor_coverage'
            else:
                return 'outdoor_coverage'
        
        # Device and hardware
        if any(keyword in text_lower for keyword in 
               ['device', 'router', 'dongle', 'sim', 'hardware', 'phone', 'mobile']):
            return 'device_hardware'
        
        # Network configuration
        if any(keyword in text_lower for keyword in 
               ['configuration', 'settings', 'apn', 'vpn', 'network mode']):
            return 'network_config'
        
        return 'general_technical'
    
    def build_enhanced_patterns(self, features, solutions, labels, location_data):
        """Build comprehensive pattern database"""
        logger.info("Building enhanced pattern database...")
        
        # Exact solution mapping
        for feature, solution in zip(features, solutions):
            feature_key = feature.lower().strip()
            self.solution_patterns[feature_key] = solution
        
        # Location-based patterns
        for location_info, solution in zip(location_data, solutions):
            loc_category = location_info['location_category']
            if loc_category not in self.location_patterns:
                self.location_patterns[loc_category] = []
            self.location_patterns[loc_category].append(solution)
        
        # Site-based patterns
        for location_info, solution in zip(location_data, solutions):
            site_info = location_info['site_info']
            if 'primary_site' in site_info:
                site_code = site_info['primary_site']
                self.site_patterns[site_code] = solution
        
        # Device patterns
        device_solutions = {}
        for feature, solution in zip(features, solutions):
            # Extract device mentions
            device_keywords = ['router', 'dongle', 'mobile', 'huawei', 'phone', 'sim']
            for device in device_keywords:
                if device in feature.lower():
                    if device not in device_solutions:
                        device_solutions[device] = []
                    device_solutions[device].append(solution)
        
        # Store most common solution for each device
        for device, sols in device_solutions.items():
            if len(sols) > 0:
                self.device_patterns[device] = max(set(sols), key=sols.count)
        
        # KPI patterns
        kpi_keywords = ['kpi', 'speed', 'slow', 'poor', 'weak']
        for feature, solution in zip(features, solutions):
            for kpi in kpi_keywords:
                if kpi in feature.lower():
                    if kpi not in self.kpi_patterns:
                        self.kpi_patterns[kpi] = []
                    self.kpi_patterns[kpi].append(solution)
        
        # Store most common KPI solutions
        for kpi, sols in self.kpi_patterns.items():
            if len(sols) > 0:
                self.kpi_patterns[kpi] = max(set(sols), key=sols.count)
        
        logger.info(f"Built patterns:")
        logger.info(f"  - Solution patterns: {len(self.solution_patterns)}")
        logger.info(f"  - Location patterns: {len(self.location_patterns)}")
        logger.info(f"  - Site patterns: {len(self.site_patterns)}")
        logger.info(f"  - Device patterns: {len(self.device_patterns)}")
        logger.info(f"  - KPI patterns: {len(self.kpi_patterns)}")
    
    def train_enhanced_classifier(self, features, labels):
        """Train enhanced classifier with location and site awareness"""
        logger.info("Training enhanced classifier...")
        
        self.categories = sorted(list(set(labels)))
        logger.info(f"Categories: {self.categories}")
        
        from collections import Counter
        label_counts = Counter(labels)
        logger.info(f"Label distribution: {dict(label_counts)}")
        
        # Prepare training data
        if len(features) < 10:
            X_train, X_test = features, features[:3] if len(features) > 3 else features
            y_train, y_test = labels, labels[:3] if len(labels) > 3 else labels
        else:
            try:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42, stratify=labels
                )
            except ValueError:
                X_train, X_test, y_train, y_test = train_test_split(
                    features, labels, test_size=0.2, random_state=42
                )
        
        # Enhanced pipeline with better parameters
        self.model = Pipeline([
            ('tfidf', TfidfVectorizer(
                max_features=5000,
                ngram_range=(1, 3),
                stop_words='english',
                lowercase=True,
                min_df=1,
                max_df=0.9,
                analyzer='word'
            )),
            ('classifier', MultinomialNB(alpha=0.05))
        ])
        
        self.model.fit(X_train, y_train)
        
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
    
    def save_enhanced_models(self):
        """Save all models and pattern data"""
        os.makedirs("models", exist_ok=True)
        
        # Save classifier
        with open("models/complaint_classifier.pkl", 'wb') as f:
            pickle.dump(self.model, f)
        
        # Save all patterns
        patterns_data = {
            'solution_patterns': self.solution_patterns,
            'location_patterns': self.location_patterns,
            'site_patterns': self.site_patterns,
            'device_patterns': self.device_patterns,
            'kpi_patterns': self.kpi_patterns
        }
        
        with open("models/enhanced_patterns.pkl", 'wb') as f:
            pickle.dump(patterns_data, f)
        
        # Save metadata
        metadata = {
            'categories': self.categories,
            'training_date': datetime.now().isoformat(),
            'training_samples': len(self.solution_patterns),
            'model_version': '2.0_enhanced',
            'features': [
                'location_aware_categorization',
                'site_pattern_matching',
                'device_specific_patterns',
                'kpi_based_analysis',
                'coordinate_parsing'
            ]
        }
        
        with open("models/model_metadata.json", 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info("Enhanced models saved:")
        logger.info("  - Classifier: models/complaint_classifier.pkl")
        logger.info("  - Patterns: models/enhanced_patterns.pkl")
        logger.info("  - Metadata: models/model_metadata.json")
    
    def train_complete_enhanced_model(self):
        """Complete enhanced training workflow"""
        logger.info("🚀 Starting enhanced model training...")
        
        # Load and analyze data
        features, labels, solutions, location_data = self.load_and_analyze_data()
        
        if len(features) == 0:
            raise ValueError("No valid training data found!")
        
        # Build enhanced patterns
        self.build_enhanced_patterns(features, solutions, labels, location_data)
        
        # Train classifier
        accuracy = self.train_enhanced_classifier(features, labels)
        
        # Save models
        self.save_enhanced_models()
        
        results = {
            'accuracy': accuracy,
            'samples': len(features),
            'categories': len(self.categories),
            'solution_patterns': len(self.solution_patterns),
            'location_patterns': len(self.location_patterns),
            'site_patterns': len(self.site_patterns),
            'device_patterns': len(self.device_patterns)
        }
        
        logger.info("✅ Enhanced training completed successfully!")
        return results

def main():
    """Main enhanced training function"""
    print("🔧 Enhanced Complaint Handling System - Model Training")
    print("=" * 60)
    
    try:
        trainer = EnhancedComplaintModelTrainer()
        results = trainer.train_complete_enhanced_model()
        
        print("\n📊 Enhanced Training Results:")
        print(f"   Accuracy: {results['accuracy']:.3f}")
        print(f"   Training Samples: {results['samples']}")
        print(f"   Categories: {results['categories']}")
        print(f"   Solution Patterns: {results['solution_patterns']}")
        print(f"   Location Patterns: {results['location_patterns']}")
        print(f"   Site Patterns: {results['site_patterns']}")
        print(f"   Device Patterns: {results['device_patterns']}")
        print("\n🎯 Enhanced models ready for deployment!")
        
    except Exception as e:
        logger.error(f"Enhanced training failed: {e}")
        print(f"\n❌ Training failed: {e}")

if __name__ == "__main__":
    main()
