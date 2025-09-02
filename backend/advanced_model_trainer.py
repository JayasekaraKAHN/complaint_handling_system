#!/usr/bin/env python3
"""
Advanced Telecom Complaint Analysis and Model Training System
This script analyzes the telecom complaint dataset and trains models for:
1. Complaint Classification
2. Solution Recommendation
3. Company-specific Knowledge Base Generation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics.pairwise import cosine_similarity
import pickle
import json
from datetime import datetime
from typing import Optional, Dict, List, Any
import warnings
warnings.filterwarnings('ignore')

class TelecomComplaintAnalyzer:
    def __init__(self, data_path):
        """Initialize the analyzer with data path"""
        self.data_path = data_path
        self.df: Optional[pd.DataFrame] = None
        self.vectorizer = None
        self.classifier = None
        self.solution_matcher = None
        self.company_knowledge = {}
        
    def load_and_analyze_data(self):
        """Load and perform comprehensive analysis of the complaint data"""
        print("🔍 Loading and analyzing telecom complaint data...")
        
        try:
            # Load the dataset
            self.df = pd.read_csv(self.data_path)
            print(f"📊 Dataset loaded: {len(self.df)} records")
            
            # Basic statistics
            print(f"📈 Data shape: {self.df.shape}")
            print(f"🏢 Columns: {self.df.columns.tolist()}")
            
            # Key analysis columns
            key_columns = ['Issue_Description', 'Solution', 'Device_type_settings_VPN_APN', 
                          'Signal_Strength', 'Site_KPI_Alarm', 'DISTRICT']
            
            available_columns = [col for col in key_columns if col in self.df.columns]
            print(f"✅ Available key columns: {available_columns}")
            
            # Clean and prepare data
            self.df['Issue_Description'] = self.df['Issue_Description'].fillna('Unknown issue')
            self.df['Solution'] = self.df['Solution'].fillna('Standard troubleshooting required')
            self.df['Device_type_settings_VPN_APN'] = self.df['Device_type_settings_VPN_APN'].fillna('Unknown device')
            
            return self.df
            
        except FileNotFoundError:
            print(f"❌ Error: Data file not found at {self.data_path}")
            raise
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            raise
    
    def analyze_complaint_patterns(self):
        """Analyze complaint patterns and generate insights"""
        print("\n🔬 Analyzing complaint patterns...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_analyze_data() first.")
        
        analysis_report = {
            'total_complaints': len(self.df),
            'unique_issues': self.df['Issue_Description'].nunique(),
            'device_types': {},
            'common_issues': {},
            'signal_issues': {},
            'district_analysis': {},
            'solution_patterns': {}
        }
        
        # Device type analysis
        if 'Device_type_settings_VPN_APN' in self.df.columns:
            device_counts = self.df['Device_type_settings_VPN_APN'].value_counts()
            analysis_report['device_types'] = {k: int(v) for k, v in device_counts.to_dict().items()}
            print(f"📱 Device types: {dict(device_counts.head())}")
        
        # Issue pattern analysis
        issue_patterns = self.df['Issue_Description'].str.lower().str.extract(
            r'(voice|call|coverage|data|network|signal|connection)'
        )[0].value_counts()
        analysis_report['common_issues'] = {k: int(v) for k, v in issue_patterns.to_dict().items()}
        print(f"🚨 Common issue types: {dict(issue_patterns.head())}")
        
        # Signal strength analysis
        if 'Signal_Strength' in self.df.columns:
            signal_issues = self.df[self.df['Signal_Strength'].notna()]
            weak_signal_count = len(signal_issues[signal_issues['Signal_Strength'].str.contains('weak|poor|low', case=False, na=False)])
            analysis_report['signal_issues']['weak_signal_complaints'] = weak_signal_count
            print(f"📶 Weak signal complaints: {weak_signal_count}")
        
        # District analysis
        if 'DISTRICT' in self.df.columns:
            district_counts = self.df['DISTRICT'].value_counts()
            analysis_report['district_analysis'] = {k: int(v) for k, v in district_counts.to_dict().items()}
            print(f"🏘️ Top affected districts: {dict(district_counts.head())}")
        
        # Solution effectiveness analysis
        solution_counts = self.df['Solution'].value_counts()
        analysis_report['solution_patterns'] = {k: int(v) for k, v in solution_counts.to_dict().items()}
        print(f"🔧 Common solutions: {dict(solution_counts.head(3))}")
        
        # Save analysis report
        with open('data/complaint_analysis_report.json', 'w') as f:
            json.dump(analysis_report, f, indent=2, default=str)
        
        return analysis_report
    
    def create_complaint_categories(self):
        """Create complaint categories for classification"""
        print("\n🏷️ Creating complaint categories...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_analyze_data() first.")
        
        def categorize_complaint(description):
            description = str(description).lower()
            if any(word in description for word in ['voice', 'call', 'calling']):
                return 'voice_issues'
            elif any(word in description for word in ['coverage', 'signal', 'network']):
                return 'coverage_issues'
            elif any(word in description for word in ['data', 'internet', 'browsing']):
                return 'data_issues'
            elif any(word in description for word in ['device', 'phone', 'handset']):
                return 'device_issues'
            else:
                return 'other_issues'
        
        self.df['complaint_category'] = self.df['Issue_Description'].apply(categorize_complaint)
        
        category_counts = self.df['complaint_category'].value_counts()
        print(f"📊 Complaint categories: {dict(category_counts)}")
        
        return category_counts
    
    def train_complaint_classifier(self):
        """Train a comprehensive model to classify complaint types and predict solutions"""
        print("\n🤖 Training advanced complaint classification model...")
        
        if self.df is None or 'complaint_category' not in self.df.columns:
            raise ValueError("Data not prepared. Call create_complaint_categories() first.")
        
        # Prepare features for training
        features = []
        
        # Text features from issue description
        text_features = self.df['Issue_Description'].fillna('').astype(str)
        
        # Add device information if available
        if 'Device_type_settings_VPN_APN' in self.df.columns:
            device_features = self.df['Device_type_settings_VPN_APN'].fillna('').astype(str)
            text_features = text_features + ' ' + device_features
        
        # Add signal strength information
        if 'Signal_Strength' in self.df.columns:
            signal_features = self.df['Signal_Strength'].fillna('').astype(str)
            text_features = text_features + ' ' + signal_features
        
        # Add site alarm information
        if 'Site_KPI_Alarm' in self.df.columns:
            site_features = self.df['Site_KPI_Alarm'].fillna('').astype(str)
            text_features = text_features + ' ' + site_features
        
        # Vectorize text features
        self.vectorizer = TfidfVectorizer(max_features=1000, ngram_range=(1, 2), stop_words='english')
        X = self.vectorizer.fit_transform(text_features)
        y = self.df['complaint_category']
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Train classifier with multiple algorithms
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.svm import SVC
        from sklearn.naive_bayes import MultinomialNB
        
        models = {
            'random_forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'gradient_boosting': GradientBoostingClassifier(random_state=42),
            'svm': SVC(probability=True, random_state=42),
            'naive_bayes': MultinomialNB()
        }
        
        best_model = None
        best_score = 0
        model_results = {}
        
        for name, model in models.items():
            model.fit(X_train, y_train)
            score = model.score(X_test, y_test)
            model_results[name] = score
            
            if score > best_score:
                best_score = score
                best_model = model
                self.classifier = model
            
            print(f"🎯 {name.replace('_', ' ').title()}: {score:.3f} accuracy")
        
        print(f"🏆 Best model: {best_score:.3f} accuracy")
        
        # Detailed classification report
        if self.classifier is not None:
            y_pred = self.classifier.predict(X_test)
            print("\n📊 Detailed Classification Report:")
            print(classification_report(y_test, y_pred))
        
        return model_results
    
    def create_solution_recommendation_system(self):
        """Create an intelligent solution recommendation system"""
        print("\n💡 Creating solution recommendation system...")
        
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        # Create solution patterns based on complaint-solution pairs
        solution_patterns = {}
        
        for _, row in self.df.iterrows():
            issue = str(row['Issue_Description']).lower()
            solution = str(row['Solution'])
            category = row.get('complaint_category', 'other_issues')
            
            # Extract key features
            device = str(row.get('Device_type_settings_VPN_APN', '')).lower()
            signal = str(row.get('Signal_Strength', '')).lower()
            district = str(row.get('DISTRICT', '')).upper()
            site_alarm = str(row.get('Site_KPI_Alarm', '')).lower()
            
            # Create pattern key
            pattern_key = f"{category}_{device}_{signal}_{district}"
            
            if pattern_key not in solution_patterns:
                solution_patterns[pattern_key] = {
                    'solutions': [],
                    'frequency': 0,
                    'success_indicators': [],
                    'related_issues': [],
                    'technical_details': {
                        'device_type': device,
                        'signal_conditions': signal,
                        'location': district,
                        'site_status': site_alarm
                    }
                }
            
            solution_patterns[pattern_key]['solutions'].append(solution)
            solution_patterns[pattern_key]['frequency'] += 1
            solution_patterns[pattern_key]['related_issues'].append(issue)
        
        # Process and refine patterns
        for pattern_key, pattern_data in solution_patterns.items():
            # Get most common solution for this pattern
            solution_counts = {}
            for sol in pattern_data['solutions']:
                solution_counts[sol] = solution_counts.get(sol, 0) + 1
            
            if solution_counts:
                most_common_solution = max(solution_counts.keys(), key=lambda k: solution_counts[k])
                pattern_data['recommended_solution'] = most_common_solution
                pattern_data['confidence'] = solution_counts[most_common_solution] / len(pattern_data['solutions'])
            else:
                pattern_data['recommended_solution'] = "Standard troubleshooting required"
                pattern_data['confidence'] = 0.0
            
            # Analyze success patterns
            if pattern_data['confidence'] > 0.7:
                pattern_data['effectiveness'] = 'high'
            elif pattern_data['confidence'] > 0.5:
                pattern_data['effectiveness'] = 'medium'
            else:
                pattern_data['effectiveness'] = 'low'
        
        self.solution_matcher = solution_patterns
        
        # Save solution patterns
        with open('data/solution_patterns.json', 'w') as f:
            json.dump(solution_patterns, f, indent=2, default=str)
        
        print(f"✅ Created {len(solution_patterns)} solution patterns")
        return solution_patterns
    
    def generate_company_specific_knowledge(self):
        """Generate company-specific knowledge base from the data"""
        print("\n🏢 Generating company-specific knowledge base...")
        
        if self.df is None:
            raise ValueError("Data not loaded.")
        
        knowledge_base = {
            'company_profile': {
                'total_complaints_analyzed': len(self.df),
                'analysis_date': datetime.now().isoformat(),
                'data_source': 'datafinal.csv'
            },
            'network_infrastructure': {},
            'device_compatibility': {},
            'coverage_patterns': {},
            'service_quality_metrics': {},
            'resolution_strategies': {}
        }
        
        # Network infrastructure analysis
        if 'Site_KPI_Alarm' in self.df.columns:
            site_issues = self.df['Site_KPI_Alarm'].value_counts()
            knowledge_base['network_infrastructure'] = {
                'common_site_issues': dict(site_issues.head(10)),
                'critical_sites': list(site_issues.index[:5])
            }
        
        # Device compatibility patterns
        if 'Device_type_settings_VPN_APN' in self.df.columns:
            device_issues = self.df.groupby('Device_type_settings_VPN_APN')['Issue_Description'].apply(list)
            device_patterns = {}
            
            for device, issues in device_issues.items():
                issue_types = []
                for issue in issues:
                    issue_str = str(issue).lower()
                    if 'voice' in issue_str or 'call' in issue_str:
                        issue_types.append('voice_issues')
                    elif 'data' in issue_str or 'internet' in issue_str:
                        issue_types.append('data_issues')
                    elif 'coverage' in issue_str or 'signal' in issue_str:
                        issue_types.append('coverage_issues')
                
                from collections import Counter
                issue_counter = Counter(issue_types)
                device_patterns[device] = dict(issue_counter)
            
            knowledge_base['device_compatibility'] = device_patterns
        
        # Coverage analysis by district
        if 'DISTRICT' in self.df.columns:
            coverage_analysis = self.df.groupby('DISTRICT').agg({
                'Issue_Description': 'count',
                'Signal_Strength': lambda x: x.str.contains('weak|poor|low', case=False, na=False).sum()
            }).to_dict()
            
            knowledge_base['coverage_patterns'] = {
                'complaints_by_district': coverage_analysis.get('Issue_Description', {}),
                'weak_signal_by_district': coverage_analysis.get('Signal_Strength', {})
            }
        
        # Service quality metrics
        if 'Signal_Strength' in self.df.columns:
            signal_quality = self.df['Signal_Strength'].value_counts()
            knowledge_base['service_quality_metrics'] = {
                'signal_strength_distribution': dict(signal_quality),
                'poor_signal_percentage': len(self.df[self.df['Signal_Strength'].str.contains('weak|poor|low', case=False, na=False)]) / len(self.df) * 100
            }
        
        # Resolution strategies
        solution_effectiveness = {}
        for category in self.df['complaint_category'].unique():
            category_data = self.df[self.df['complaint_category'] == category]
            solutions = category_data['Solution'].value_counts()
            solution_effectiveness[category] = {
                'primary_solutions': dict(solutions.head(3)),
                'resolution_rate': len(category_data) / len(self.df) * 100
            }
        
        knowledge_base['resolution_strategies'] = solution_effectiveness
        
        self.company_knowledge = knowledge_base
        
        # Save knowledge base
        with open('data/company_knowledge_base.json', 'w') as f:
            json.dump(knowledge_base, f, indent=2, default=str)
        
        print("✅ Company-specific knowledge base created")
        return knowledge_base
    
    def generate_intelligent_solution(self, complaint_text: str, device_info: str = "", 
                                   signal_strength: str = "", location: str = "", 
                                   site_alarm: str = "") -> Dict[str, Any]:
        """Generate intelligent solution based on trained models and patterns"""
        
        if self.vectorizer is None or self.classifier is None:
            raise ValueError("Models not trained. Call train_complaint_classifier() first.")
        
        # Prepare input for classification
        combined_text = f"{complaint_text} {device_info} {signal_strength} {site_alarm}"
        text_vector = self.vectorizer.transform([combined_text])
        
        # Classify complaint
        predicted_category = self.classifier.predict(text_vector)[0]
        confidence_scores = self.classifier.predict_proba(text_vector)[0]
        category_confidence = max(confidence_scores)
        
        # Find matching solution pattern
        pattern_key = f"{predicted_category}_{device_info.lower()}_{signal_strength.lower()}_{location.upper()}"
        
        solution_data = {
            'complaint_analysis': {
                'original_complaint': complaint_text,
                'predicted_category': predicted_category,
                'confidence': float(category_confidence),
                'category_description': self._get_category_description(predicted_category)
            },
            'root_cause_analysis': self._analyze_root_cause(complaint_text, device_info, signal_strength, site_alarm),
            'recommended_solution': None,
            'alternative_solutions': [],
            'technical_details': {},
            'pattern_matching': {},
            'company_specific_context': {}
        }
        
        # Get solution from patterns
        if self.solution_matcher and pattern_key in self.solution_matcher:
            pattern = self.solution_matcher[pattern_key]
            solution_data['recommended_solution'] = pattern['recommended_solution']
            solution_data['pattern_matching'] = {
                'pattern_found': True,
                'pattern_confidence': pattern['confidence'],
                'pattern_effectiveness': pattern['effectiveness'],
                'similar_cases': pattern['frequency']
            }
        else:
            # Find similar patterns
            similar_patterns = self._find_similar_patterns(predicted_category, device_info, signal_strength, location)
            if similar_patterns:
                best_pattern = similar_patterns[0]
                solution_data['recommended_solution'] = best_pattern['recommended_solution']
                solution_data['pattern_matching'] = {
                    'pattern_found': False,
                    'similar_pattern_used': True,
                    'similarity_score': best_pattern.get('similarity', 0.8),
                    'pattern_confidence': best_pattern['confidence']
                }
            else:
                # Fallback to category-based solution
                category_solutions = self._get_category_solutions(predicted_category)
                solution_data['recommended_solution'] = category_solutions[0] if category_solutions else "Standard troubleshooting required"
                solution_data['pattern_matching'] = {
                    'pattern_found': False,
                    'fallback_used': True,
                    'category_based': True
                }
        
        # Add technical analysis
        solution_data['technical_details'] = {
            'device_analysis': self._analyze_device_compatibility(device_info),
            'signal_analysis': self._analyze_signal_strength(signal_strength),
            'location_analysis': self._analyze_location_factors(location),
            'infrastructure_status': self._analyze_infrastructure(site_alarm)
        }
        
        # Add company-specific context
        if self.company_knowledge:
            solution_data['company_specific_context'] = {
                'similar_cases_in_area': self._get_area_cases(location),
                'device_known_issues': self._get_device_issues(device_info),
                'escalation_required': self._determine_escalation(predicted_category, category_confidence),
                'estimated_resolution_time': self._estimate_resolution_time(predicted_category, site_alarm)
            }
        
        return solution_data
    
    def _get_category_description(self, category: str) -> str:
        """Get description for complaint category"""
        descriptions = {
            'voice_issues': 'Voice call related problems including call drops, poor quality, and connection failures',
            'coverage_issues': 'Network coverage and signal strength related problems',
            'data_issues': 'Data connectivity, speed, and internet access problems',
            'device_issues': 'Device-specific configuration and compatibility problems',
            'other_issues': 'General service issues not fitting other categories'
        }
        return descriptions.get(category, 'Unclassified issue type')
    
    def _analyze_root_cause(self, complaint: str, device: str, signal: str, site_alarm: str) -> Dict[str, Any]:
        """Analyze potential root causes"""
        root_causes = []
        primary_cause = "Unknown"
        
        complaint_lower = complaint.lower()
        
        if 'call drop' in complaint_lower or 'voice' in complaint_lower:
            if 'weak' in signal.lower() or 'poor' in signal.lower():
                primary_cause = "Poor signal strength affecting voice quality"
                root_causes.append("Signal attenuation")
            elif site_alarm:
                primary_cause = "Network infrastructure issues"
                root_causes.append("Site equipment malfunction")
            else:
                primary_cause = "Voice service configuration issue"
                root_causes.append("VoLTE/VoWiFi configuration")
        
        elif 'data' in complaint_lower or 'internet' in complaint_lower:
            if 'slow' in complaint_lower:
                primary_cause = "Data throughput limitation"
                root_causes.append("Network congestion or data bundle limitation")
            else:
                primary_cause = "Data connectivity issue"
                root_causes.append("APN configuration or network routing")
        
        elif 'coverage' in complaint_lower or 'signal' in complaint_lower:
            primary_cause = "Coverage gap or signal attenuation"
            root_causes.append("Distance from cell tower")
            root_causes.append("Physical obstructions")
            if 'indoor' in complaint_lower:
                root_causes.append("Indoor signal penetration")
        
        return {
            'primary_cause': primary_cause,
            'contributing_factors': root_causes,
            'confidence': 0.8,
            'analysis_method': 'pattern_based'
        }
    
    def _find_similar_patterns(self, category: str, device: str, signal: str, location: str) -> List[Dict]:
        """Find similar solution patterns"""
        if not self.solution_matcher:
            return []
        
        similar_patterns = []
        
        for pattern_key, pattern_data in self.solution_matcher.items():
            pattern_parts = pattern_key.split('_')
            if len(pattern_parts) >= 4:
                pat_category, pat_device, pat_signal, pat_location = pattern_parts[0], pattern_parts[1], pattern_parts[2], pattern_parts[3]
                
                similarity = 0
                if pat_category == category:
                    similarity += 0.4
                if device.lower() in pat_device or pat_device in device.lower():
                    similarity += 0.3
                if signal.lower() in pat_signal or pat_signal in signal.lower():
                    similarity += 0.2
                if location.upper() == pat_location:
                    similarity += 0.1
                
                if similarity > 0.5:
                    pattern_data['similarity'] = similarity
                    similar_patterns.append(pattern_data)
        
        return sorted(similar_patterns, key=lambda x: x.get('similarity', 0), reverse=True)
    
    def _get_category_solutions(self, category: str) -> List[str]:
        """Get solutions for category"""
        if self.df is None:
            return []
        
        category_data = self.df[self.df['complaint_category'] == category]
        solutions = category_data['Solution'].value_counts()
        return list(solutions.index[:3])
    
    def _analyze_device_compatibility(self, device_info: str) -> Dict[str, Any]:
        """Analyze device compatibility"""
        return {
            'device_type': device_info,
            'known_issues': 'None identified' if not device_info else f"Checking compatibility for {device_info}",
            'recommended_settings': 'Standard APN configuration'
        }
    
    def _analyze_signal_strength(self, signal_strength: str) -> Dict[str, Any]:
        """Analyze signal strength factors"""
        if not signal_strength:
            return {'status': 'Unknown', 'recommendation': 'Check signal strength'}
        
        signal_lower = signal_strength.lower()
        if 'weak' in signal_lower or 'poor' in signal_lower:
            return {
                'status': 'Poor signal detected',
                'recommendation': 'Consider signal amplifier or relocate device',
                'technical_action': 'Check for physical obstructions'
            }
        elif 'good' in signal_lower or 'strong' in signal_lower:
            return {
                'status': 'Good signal strength',
                'recommendation': 'Signal is adequate, check other factors',
                'technical_action': 'Focus on configuration or device issues'
            }
        else:
            return {
                'status': 'Signal status unclear',
                'recommendation': 'Perform signal strength measurement',
                'technical_action': 'Request detailed signal report'
            }
    
    def _analyze_location_factors(self, location: str) -> Dict[str, Any]:
        """Analyze location-specific factors"""
        return {
            'area': location or 'Unknown',
            'coverage_status': 'Checking coverage maps',
            'nearby_sites': 'Identifying nearest cell towers'
        }
    
    def _analyze_infrastructure(self, site_alarm: str) -> Dict[str, Any]:
        """Analyze infrastructure status"""
        if not site_alarm:
            return {'status': 'No alarms reported', 'action': 'Normal operations'}
        
        alarm_lower = site_alarm.lower()
        if 'down' in alarm_lower or 'outage' in alarm_lower:
            return {
                'status': 'Site outage detected',
                'priority': 'High',
                'action': 'Immediate technical intervention required'
            }
        elif 'alarm' in alarm_lower:
            return {
                'status': 'Site alarm active',
                'priority': 'Medium',
                'action': 'Check and clear alarms'
            }
        else:
            return {
                'status': 'Infrastructure monitoring required',
                'priority': 'Low',
                'action': 'Routine maintenance check'
            }
    
    def _get_area_cases(self, location: str) -> Dict[str, Any]:
        """Get similar cases in the area"""
        if not self.df is None and 'DISTRICT' in self.df.columns and location:
            area_cases = self.df[self.df['DISTRICT'].str.contains(location, case=False, na=False)]
            return {
                'total_cases': len(area_cases),
                'common_issues': list(area_cases['Issue_Description'].value_counts().head(3).index)
            }
        return {'total_cases': 0, 'common_issues': []}
    
    def _get_device_issues(self, device_info: str) -> List[str]:
        """Get known issues for device"""
        if not self.df is None and 'Device_type_settings_VPN_APN' in self.df.columns and device_info:
            device_cases = self.df[self.df['Device_type_settings_VPN_APN'].str.contains(device_info, case=False, na=False)]
            return list(device_cases['Issue_Description'].value_counts().head(3).index)
        return []
    
    def _determine_escalation(self, category: str, confidence: float) -> bool:
        """Determine if escalation is needed"""
        high_priority_categories = ['coverage_issues', 'voice_issues']
        return category in high_priority_categories or confidence < 0.6
    
    def _estimate_resolution_time(self, category: str, site_alarm: str) -> str:
        """Estimate resolution time"""
        if site_alarm and ('down' in site_alarm.lower() or 'outage' in site_alarm.lower()):
            return "2-4 hours (infrastructure repair required)"
        elif category == 'device_issues':
            return "30 minutes - 2 hours (configuration change)"
        elif category == 'coverage_issues':
            return "1-7 days (infrastructure enhancement may be needed)"
        else:
            return "1-24 hours (standard resolution time)"
    
    def save_models(self):
        """Save trained models and patterns"""
        print("\n💾 Saving trained models...")
        
        if self.classifier is not None:
            with open('models/complaint_classifier.pkl', 'wb') as f:
                pickle.dump(self.classifier, f)
        
        if self.vectorizer is not None:
            with open('models/text_vectorizer.pkl', 'wb') as f:
                pickle.dump(self.vectorizer, f)
        
        if self.solution_matcher:
            with open('models/solution_patterns.pkl', 'wb') as f:
                pickle.dump(self.solution_matcher, f)
        
        if self.company_knowledge:
            with open('models/company_knowledge.pkl', 'wb') as f:
                pickle.dump(self.company_knowledge, f)
        
        print("✅ Models saved successfully")
    
    def load_models(self):
        """Load pre-trained models"""
        print("\n📥 Loading pre-trained models...")
        
        try:
            with open('models/complaint_classifier.pkl', 'rb') as f:
                self.classifier = pickle.load(f)
            
            with open('models/text_vectorizer.pkl', 'rb') as f:
                self.vectorizer = pickle.load(f)
            
            with open('models/solution_patterns.pkl', 'rb') as f:
                self.solution_matcher = pickle.load(f)
            
            with open('models/company_knowledge.pkl', 'rb') as f:
                self.company_knowledge = pickle.load(f)
            
            print("✅ Models loaded successfully")
            return True
            
        except FileNotFoundError as e:
            print(f"❌ Model files not found: {e}")
            return False
        print("\n🤖 Training complaint classification model...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_analyze_data() first.")
        
        # Prepare features and target
        X = self.df['Issue_Description'].fillna('')
        y = self.df['complaint_category']
        
        # Vectorize text
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        X_vectorized = self.vectorizer.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X_vectorized, y, test_size=0.2, random_state=42, stratify=y
        )
        
        # Train classifier
        self.classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        self.classifier.fit(X_train, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"✅ Classification accuracy: {accuracy:.2%}")
        print(f"📊 Classification report:")
        print(classification_report(y_test, y_pred))
        
        return accuracy
    
    def train_solution_matcher(self):
        """Train a solution recommendation system"""
        print("\n🎯 Training solution recommendation system...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_analyze_data() first.")
        
        # Create solution-issue pairs
        solution_data = self.df[['Issue_Description', 'Solution']].dropna()
        
        # Vectorize issues and solutions
        issue_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        solution_vectorizer = TfidfVectorizer(max_features=500, stop_words='english')
        
        issue_vectors = issue_vectorizer.fit_transform(solution_data['Issue_Description'])
        solution_vectors = solution_vectorizer.fit_transform(solution_data['Solution'])
        
        self.solution_matcher = {
            'issue_vectorizer': issue_vectorizer,
            'solution_vectorizer': solution_vectorizer,
            'issue_vectors': issue_vectors,
            'solution_vectors': solution_vectors,
            'solutions': solution_data['Solution'].tolist(),
            'issues': solution_data['Issue_Description'].tolist()
        }
        
        print(f"✅ Solution matcher trained with {len(solution_data)} issue-solution pairs")
        return len(solution_data)
    
    def generate_company_knowledge_base(self):
        """Generate company-specific knowledge base"""
        print("\n📚 Generating company-specific knowledge base...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_and_analyze_data() first.")
        
        knowledge_base = {}
        
        # Group by complaint category and extract solutions
        for category in self.df['complaint_category'].unique():
            category_data = self.df[self.df['complaint_category'] == category]
            
            # Get most common solutions for this category
            solutions = category_data['Solution'].value_counts().head(3)
            
            # Get typical issues
            issues = category_data['Issue_Description'].value_counts().head(3)
            
            # Get device patterns
            devices = category_data['Device_type_settings_VPN_APN'].value_counts().head(3)
            
            knowledge_base[category] = {
                'description': self._get_category_description(category),
                'common_solutions': {k: int(v) for k, v in solutions.to_dict().items()},
                'typical_issues': {k: int(v) for k, v in issues.to_dict().items()},
                'affected_devices': {k: int(v) for k, v in devices.to_dict().items()},
                'resolution_tips': self._get_resolution_tips(category),
                'prevention_measures': self._get_prevention_measures(category)
            }
        
        # General company policies
        knowledge_base['company_policies'] = {
            'sla_response_time': '24 hours for critical issues',
            'escalation_procedure': 'L1 -> L2 -> L3 -> Network Operations',
            'customer_notification': 'SMS and email notifications for outages',
            'compensation_policy': 'Service credits for extended outages'
        }
        
        # Network infrastructure insights
        if 'Site_KPI_Alarm' in self.df.columns:
            site_issues = self.df[self.df['Site_KPI_Alarm'].notna()]
            knowledge_base['network_insights'] = {
                'total_sites_monitored': int(site_issues['Site_KPI_Alarm'].nunique()),
                'common_alarms': {k: int(v) for k, v in site_issues['Site_KPI_Alarm'].value_counts().head(5).to_dict().items()}
            }
        
        self.company_knowledge = knowledge_base
        
        # Save knowledge base
        with open('data/company_knowledge_base.json', 'w') as f:
            json.dump(knowledge_base, f, indent=2, default=str)
        
        print(f"✅ Knowledge base created with {len(knowledge_base)} categories")
        return knowledge_base
    
    def _get_resolution_tips(self, category):
        """Get resolution tips for each category"""
        tips = {
            'voice_issues': [
                'Check network signal strength in the area',
                'Restart device and test call quality',
                'Update device software and carrier settings',
                'Check for network maintenance in the area'
            ],
            'coverage_issues': [
                'Verify nearest tower location and status',
                'Check for physical obstructions',
                'Use Wi-Fi calling when available',
                'Consider signal booster installation'
            ],
            'data_issues': [
                'Reset network settings on device',
                'Check APN configuration',
                'Verify data plan allowances',
                'Test connection in different locations'
            ],
            'device_issues': [
                'Update device firmware',
                'Check device compatibility',
                'Reset network settings',
                'Contact device manufacturer if needed'
            ],
            'other_issues': [
                'Perform comprehensive diagnostics',
                'Check service status dashboard',
                'Contact technical support for assistance',
                'Document issue for trend analysis'
            ]
        }
        return tips.get(category, ['Contact customer service for assistance'])
    
    def _get_prevention_measures(self, category):
        """Get prevention measures for each category"""
        measures = {
            'voice_issues': [
                'Regular network quality monitoring',
                'Proactive maintenance scheduling',
                'Customer device update notifications'
            ],
            'coverage_issues': [
                'Network capacity planning',
                'Site optimization programs',
                'Customer coverage mapping'
            ],
            'data_issues': [
                'Network capacity upgrades',
                'APN configuration guidance',
                'Data usage monitoring tools'
            ],
            'device_issues': [
                'Device compatibility testing',
                'Regular firmware update campaigns',
                'Customer education programs'
            ],
            'other_issues': [
                'Comprehensive monitoring systems',
                'Regular system health checks',
                'Customer communication programs'
            ]
        }
        return measures.get(category, ['Regular system monitoring and maintenance'])

def main():
    """Main function to run the analysis"""
    # Initialize analyzer
    analyzer = TelecomComplaintAnalyzer('data/datafinal.csv')
    
    try:
        # Load or train models
        if not analyzer.load_models():
            print("🚀 Starting comprehensive telecom complaint analysis...")
            print("=" * 60)
            
            # Load and analyze data
            analyzer.load_and_analyze_data()
            
            # Analyze patterns
            analysis_report = analyzer.analyze_complaint_patterns()
            
            # Create categories
            category_counts = analyzer.create_complaint_categories()
            
            # Train models
            model_results = analyzer.train_complaint_classifier()
            solution_patterns = analyzer.create_solution_recommendation_system()
            
            # Generate knowledge base
            knowledge_base = analyzer.generate_company_specific_knowledge()
            
            # Save everything
            analyzer.save_models()
            
            # Final summary
            print("\n" + "=" * 60)
            print("📊 ANALYSIS COMPLETE - SUMMARY REPORT")
            print("=" * 60)
            if analyzer.df is not None:
                print(f"📈 Total complaints analyzed: {len(analyzer.df)}")
            else:
                print("📈 Total complaints analyzed: 0")
            print(f"🏷️ Complaint categories: {len(category_counts)}")
            print(f"🎯 Best model accuracy: {max(model_results.values()):.3f}")
            print(f"💡 Solution patterns created: {len(solution_patterns)}")
            print(f"📚 Knowledge base created successfully")
            print(f"✅ All models and data saved successfully")
        
        # Test the trained models
        print("\n🧪 Testing intelligent solution generation...")
        
        test_cases = [
            {
                'complaint': "Customer experiencing call drops frequently",
                'device': "Samsung smartphone",
                'signal': "weak signal strength",
                'location': "Colombo",
                'site_alarm': ""
            },
            {
                'complaint': "No network coverage in residential area", 
                'device': "iPhone",
                'signal': "no coverage",
                'location': "Kandy",
                'site_alarm': "Site down alarm"
            },
            {
                'complaint': "Internet browsing is very slow",
                'device': "Huawei router",
                'signal': "good signal",
                'location': "Kalutara", 
                'site_alarm': ""
            }
        ]
        
        print("\n🔬 Sample Solution Generation:")
        for i, test_case in enumerate(test_cases, 1):
            print(f"\n--- Test Case {i} ---")
            
            solution = analyzer.generate_intelligent_solution(
                complaint_text=test_case['complaint'],
                device_info=test_case['device'],
                signal_strength=test_case['signal'],
                location=test_case['location'],
                site_alarm=test_case['site_alarm']
            )
            
            print(f"Complaint: {test_case['complaint']}")
            print(f"Category: {solution['complaint_analysis']['predicted_category']}")
            print(f"Confidence: {solution['complaint_analysis']['confidence']:.2%}")
            print(f"Root Cause: {solution['root_cause_analysis']['primary_cause']}")
            print(f"Solution: {solution['recommended_solution']}")
            print(f"Pattern Match: {solution['pattern_matching']}")
            
        print("\n🎉 Enhanced model analysis complete!")
        print("✅ The system can now provide intelligent solutions based on:")
        print("   • Complaint classification and categorization")
        print("   • Pattern matching from historical data")
        print("   • Root cause analysis")
        print("   • Company-specific knowledge base")
        print("   • Device, location, and signal analysis")
        
    except Exception as e:
        print(f"❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
