"""
Advanced Solution Generator with Enhanced Keyword Mapping
Considers all user inputs and datafinal.csv patterns
"""
import pickle
import os
import json
import pandas as pd
import re
from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Optional

class AdvancedSolutionGenerator:
    def __init__(self):
        self.classifier_model = None
        self.solution_patterns = {}
        self.model_metadata = {}
        self.data_patterns = {}
        self.keyword_solutions = {}
        self.field_patterns = {}
        self.load_models()
        self.analyze_training_data()
    
    def load_models(self):
        """Load trained models and enhanced patterns"""
        try:
            with open('models/complaint_classifier.pkl', 'rb') as f:
                self.classifier_model = pickle.load(f)
            
            # Try to load enhanced patterns first, fallback to basic patterns
            try:
                with open('models/enhanced_patterns.pkl', 'rb') as f:
                    enhanced_patterns = pickle.load(f)
                self.solution_patterns = enhanced_patterns.get('solution_patterns', {})
                self.location_patterns = enhanced_patterns.get('location_patterns', {})
                self.site_patterns = enhanced_patterns.get('site_patterns', {})
                self.device_patterns = enhanced_patterns.get('device_patterns', {})
                self.kpi_patterns = enhanced_patterns.get('kpi_patterns', {})
                print("✅ Enhanced patterns loaded")
            except FileNotFoundError:
                with open('models/solution_patterns.pkl', 'rb') as f:
                    self.solution_patterns = pickle.load(f)
                self.location_patterns = {}
                self.site_patterns = {}
                self.device_patterns = {}
                self.kpi_patterns = {}
                print("⚠️ Using basic patterns only")
            
            with open('models/model_metadata.json', 'r') as f:
                self.model_metadata = json.load(f)
            
            self.model_loaded = True
            print("✅ Advanced models loaded successfully")
            
        except Exception as e:
            print(f"❌ Error loading models: {e}")
            self.model_loaded = False
    
    def analyze_training_data(self):
        """Analyze datafinal.csv to extract advanced patterns"""
        try:
            df = pd.read_csv('data/datafinal.csv')
            print(f"📊 Analyzing {len(df)} training records...")
            
            # Extract keyword-solution mappings from each field
            self.field_patterns = {
                'device_patterns': self._extract_device_patterns(df),
                'site_patterns': self._extract_site_patterns(df),
                'signal_patterns': self._extract_signal_patterns(df),
                'location_patterns': self._extract_location_patterns(df),
                'issue_patterns': self._extract_issue_patterns(df)
            }
            
            # Build comprehensive keyword mappings
            self.keyword_solutions = self._build_keyword_solutions(df)
            
            print(f"🔍 Extracted {len(self.keyword_solutions)} keyword-solution mappings")
            
        except Exception as e:
            print(f"⚠️ Could not analyze training data: {e}")
            self.field_patterns = {}
            self.keyword_solutions = {}
    
    def parse_location_coordinates(self, location_str: str) -> Tuple[Optional[float], Optional[float], Optional[str]]:
        """Parse location string to extract coordinates or location name"""
        if not location_str or pd.isna(location_str):
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
    
    def categorize_location(self, lat: Optional[float], lon: Optional[float], location_name: Optional[str]) -> str:
        """Categorize location for pattern matching"""
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
    
    def extract_site_codes(self, site_text: str) -> List[str]:
        """Extract site codes from site/alarm text"""
        if not site_text or pd.isna(site_text):
            return []
        
        site_text = str(site_text).upper()
        # Extract site codes (pattern: 2-3 letters + 3-4 alphanumeric)
        site_codes = re.findall(r'[A-Z]{2,3}[A-Z0-9]{3,4}', site_text)
        return site_codes
    
    def _categorize_device(self, device_text: str) -> str:
        """Categorize device type for pattern matching"""
        if not device_text:
            return "unknown"
        
        device_lower = device_text.lower()
        if any(keyword in device_lower for keyword in ['mobile', 'phone', 'smartphone']):
            return "mobile_device"
        elif any(keyword in device_lower for keyword in ['router', 'huawei router']):
            return "router"
        elif any(keyword in device_lower for keyword in ['dongle', 'usb']):
            return "dongle"
        elif 'sim' in device_lower:
            return "sim_related"
        else:
            return "other_device"
    
    def _categorize_signal(self, signal_strength: str | None, signal_quality: str | None) -> str:
        """Categorize signal issues for pattern matching"""
        strength_text = (signal_strength or '').lower()
        quality_text = (signal_quality or '').lower()
        
        if any(keyword in strength_text for keyword in ['weak', 'poor', 'low']):
            return "weak_signal"
        elif any(keyword in strength_text for keyword in ['no signal', 'no coverage']):
            return "no_signal"
        elif any(keyword in quality_text for keyword in ['poor', 'bad', 'intermittent']):
            return "poor_quality"
        elif strength_text or quality_text:
            return "signal_issue"
        else:
            return "unknown"
    
    def _extract_device_patterns(self, df) -> Dict:
        """Extract device-specific patterns"""
        device_patterns = {}
        
        if 'Device_type_settings_VPN_APN' in df.columns:
            device_data = df.dropna(subset=['Device_type_settings_VPN_APN', 'Solution'])
            
            for _, row in device_data.iterrows():
                device_info = str(row['Device_type_settings_VPN_APN']).lower()
                solution = str(row['Solution'])
                
                # Extract device types
                device_keywords = ['mobile', 'router', 'huawei', 'dongle', 'sim', 'phone', 'iphone', 'samsung']
                for keyword in device_keywords:
                    if keyword in device_info:
                        if keyword not in device_patterns:
                            device_patterns[keyword] = []
                        device_patterns[keyword].append(solution)
        
        # Keep most common solution for each device type
        for device, solutions in device_patterns.items():
            device_patterns[device] = max(set(solutions), key=solutions.count)
        
        return device_patterns
    
    def _extract_site_patterns(self, df) -> Dict:
        """Extract site and alarm patterns"""
        site_patterns = {}
        
        if 'Site_KPI_Alarm' in df.columns:
            site_data = df.dropna(subset=['Site_KPI_Alarm', 'Solution'])
            
            for _, row in site_data.iterrows():
                site_info = str(row['Site_KPI_Alarm']).lower()
                solution = str(row['Solution'])
                
                # Extract site-related keywords
                site_keywords = ['alarm', 'unavailability', 'site down', 'kpi', 'abnormal', 'outage', 'maintenance']
                for keyword in site_keywords:
                    if keyword in site_info:
                        if keyword not in site_patterns:
                            site_patterns[keyword] = []
                        site_patterns[keyword].append(solution)
                
                # Extract specific site codes (e.g., KLPOR5, KLPET1)
                site_codes = re.findall(r'[A-Z]{2}[A-Z0-9]{3,4}', site_info.upper())
                for code in site_codes:
                    if code not in site_patterns:
                        site_patterns[code] = []
                    site_patterns[code].append(solution)
        
        # Keep most common solution for each pattern
        for pattern, solutions in site_patterns.items():
            if len(solutions) > 0:
                site_patterns[pattern] = max(set(solutions), key=solutions.count)
        
        return site_patterns
    
    def _extract_signal_patterns(self, df) -> Dict:
        """Extract signal strength patterns"""
        signal_patterns = {}
        
        if 'Signal_Strength' in df.columns:
            signal_data = df.dropna(subset=['Signal_Strength', 'Solution'])
            
            for _, row in signal_data.iterrows():
                signal_info = str(row['Signal_Strength']).lower()
                solution = str(row['Solution'])
                
                # Extract signal-related keywords
                signal_keywords = ['weak', 'poor', 'low', 'no signal', 'coverage', 'indoor', 'outdoor']
                for keyword in signal_keywords:
                    if keyword in signal_info:
                        if keyword not in signal_patterns:
                            signal_patterns[keyword] = []
                        signal_patterns[keyword].append(solution)
        
        # Keep most common solution for each pattern
        for pattern, solutions in signal_patterns.items():
            if len(solutions) > 0:
                signal_patterns[pattern] = max(set(solutions), key=solutions.count)
        
        return signal_patterns
    
    def _extract_location_patterns(self, df) -> Dict:
        """Extract location-specific patterns"""
        location_patterns = {}
        
        # Check for location-related columns
        location_columns = ['Location', 'DISTRICT', 'district', 'province']
        for col in location_columns:
            if col in df.columns:
                location_data = df.dropna(subset=[col, 'Solution'])
                
                for _, row in location_data.iterrows():
                    location = str(row[col]).lower()
                    solution = str(row['Solution'])
                    
                    if location != 'nan' and len(location) > 2:
                        if location not in location_patterns:
                            location_patterns[location] = []
                        location_patterns[location].append(solution)
        
        # Keep most common solution for each location
        for location, solutions in location_patterns.items():
            if len(solutions) > 0:
                location_patterns[location] = max(set(solutions), key=solutions.count)
        
        return location_patterns
    
    def _extract_issue_patterns(self, df) -> Dict:
        """Extract issue description patterns"""
        issue_patterns = {}
        
        if 'Issue_Description' in df.columns:
            issue_data = df.dropna(subset=['Issue_Description', 'Solution'])
            
            for _, row in issue_data.iterrows():
                issue = str(row['Issue_Description']).lower()
                solution = str(row['Solution'])
                
                # Extract key phrases and words
                words = re.findall(r'\b\w{4,}\b', issue)  # Words with 4+ characters
                for word in words:
                    if word not in issue_patterns:
                        issue_patterns[word] = []
                    issue_patterns[word].append(solution)
        
        # Keep patterns that appear multiple times
        filtered_patterns = {}
        for word, solutions in issue_patterns.items():
            if len(solutions) >= 2:  # Only keep if appears 2+ times
                filtered_patterns[word] = max(set(solutions), key=solutions.count)
        
        return filtered_patterns
    
    def _build_keyword_solutions(self, df) -> Dict:
        """Build comprehensive keyword-solution mapping"""
        keyword_solutions = {}
        
        # Combine all patterns
        all_patterns = {}
        for pattern_type, patterns in self.field_patterns.items():
            all_patterns.update(patterns)
        
        # Add weighted scoring for better matching
        for keyword, solution in all_patterns.items():
            keyword_solutions[keyword.lower()] = {
                'solution': solution,
                'confidence': 0.8,  # Base confidence
                'source': 'data_analysis'
            }
        
        return keyword_solutions
    
    def analyze_user_input(self, msisdn: str, complaint: str, 
                          device_type_settings_vpn_apn: str = "", 
                          signal_strength: str = "",
                          quality_of_signal: str = "",
                          site_kpi_alarm: str = "", 
                          past_data_analysis: str = "",
                          indoor_outdoor_coverage_issue: str = "") -> Dict:
        """Enhanced analysis of all user inputs with comprehensive field matching"""
        analysis = {
            'matched_patterns': [],
            'confidence_scores': [],
            'field_matches': {},
            'category_prediction': None,
            'best_solution': None,
            'device_info': {},
            'signal_info': {},
            'site_info': {},
            'coverage_info': {}
        }
        
        if not self.model_loaded:
            return analysis
        
        # Analyze device information
        analysis['device_info'] = {
            'device_type': device_type_settings_vpn_apn or '',
            'has_vpn': 'vpn' in (device_type_settings_vpn_apn or '').lower(),
            'device_category': self._categorize_device(device_type_settings_vpn_apn or '')
        }
        
        # Analyze signal information
        analysis['signal_info'] = {
            'signal_strength': signal_strength or '',
            'signal_quality': quality_of_signal or '',
            'signal_category': self._categorize_signal(signal_strength, quality_of_signal)
        }
        
        # Extract site information
        site_codes = self.extract_site_codes(site_kpi_alarm or '')
        analysis['site_info'] = {
            'site_codes': site_codes,
            'primary_site': site_codes[0] if site_codes else None,
            'kpi_alarm_text': site_kpi_alarm or ''
        }
        
        # Analyze coverage information
        analysis['coverage_info'] = {
            'coverage_issue': indoor_outdoor_coverage_issue or '',
            'past_analysis': past_data_analysis or ''
        }
        
        # Combine all input text for ML prediction
        full_text = f"{complaint} {device_type_settings_vpn_apn} {signal_strength} {quality_of_signal} {site_kpi_alarm} {past_data_analysis} {indoor_outdoor_coverage_issue}".strip()
        
        # Get ML prediction
        try:
            if self.classifier_model is not None:
                predicted_category = self.classifier_model.predict([full_text])[0]
                confidence_scores = self.classifier_model.predict_proba([full_text])[0]
                analysis['category_prediction'] = predicted_category
                analysis['ml_confidence'] = max(confidence_scores)
        except Exception as e:
            print(f"ML prediction error: {e}")
        
        # Enhanced pattern matching with location and site awareness
        matched_solutions = []
        total_confidence = 0
        
        # 1. Check enhanced site patterns if available
        if hasattr(self, 'site_patterns') and site_codes:
            for site_code in site_codes:
                if site_code in self.site_patterns:
                    matched_solutions.append({
                        'solution': self.site_patterns[site_code],
                        'confidence': 0.9,
                        'source': 'site_pattern',
                        'match_type': f'site_code_{site_code}'
                    })
                    total_confidence += 0.9
        
        # 2. Check enhanced device patterns based on device analysis
        if hasattr(self, 'device_patterns') and analysis['device_info']['device_category'] != 'unknown':
            device_category = analysis['device_info']['device_category']
            if device_category in self.field_patterns.get('device_patterns', {}):
                device_solution = self.field_patterns['device_patterns'][device_category]
                matched_solutions.append({
                    'solution': device_solution,
                    'confidence': 0.8,
                    'source': 'device_pattern',
                    'match_type': f'device_{device_category}'
                })
                total_confidence += 0.8
        
        # 3. Check signal patterns
        if hasattr(self, 'signal_patterns') and analysis['signal_info']['signal_category'] != 'unknown':
            signal_category = analysis['signal_info']['signal_category']
            if signal_category in self.field_patterns.get('signal_patterns', {}):
                signal_solution = self.field_patterns['signal_patterns'][signal_category]
                matched_solutions.append({
                    'solution': signal_solution,
                    'confidence': 0.7,
                    'source': 'signal_pattern',
                    'match_type': f'signal_{signal_category}'
                })
                total_confidence += 0.7
        
        # 4. Check coverage issue patterns
        if indoor_outdoor_coverage_issue:
            coverage_lower = indoor_outdoor_coverage_issue.lower()
            if 'indoor' in coverage_lower:
                matched_solutions.append({
                    'solution': "Install indoor coverage booster or check indoor signal repeater configuration.",
                    'confidence': 0.75,
                    'source': 'coverage_pattern',
                    'match_type': 'indoor_coverage'
                })
                total_confidence += 0.75
        
        # 5. Fallback to basic solution patterns
        full_text_key = full_text.lower().strip()
        if full_text_key in self.solution_patterns:
            matched_solutions.append({
                'solution': self.solution_patterns[full_text_key],
                'confidence': 0.85,
                'source': 'exact_match',
                'match_type': 'full_text'
            })
            total_confidence += 0.85
        
        # Select best solution
        if matched_solutions:
            # Sort by confidence and select highest
            best_match = max(matched_solutions, key=lambda x: x['confidence'])
            analysis['best_solution'] = best_match['solution']
            analysis['best_match_info'] = best_match
            analysis['all_matches'] = matched_solutions
        
        analysis['total_confidence'] = total_confidence
        return analysis
        
        # Find best solution based on frequency and confidence
        if matched_solutions:
            solution_counter = Counter(matched_solutions)
            analysis['best_solution'] = solution_counter.most_common(1)[0][0]
            analysis['pattern_confidence'] = total_confidence / len(matched_solutions)
        
        return analysis
    
    def _match_field_patterns(self, field_value: str, field_name: str) -> List[Dict]:
        """Match patterns for a specific field"""
        matches = []
        field_value_lower = field_value.lower()
        
        # Check keyword solutions
        for keyword, solution_data in self.keyword_solutions.items():
            if keyword in field_value_lower:
                matches.append({
                    'keyword': keyword,
                    'solution': solution_data['solution'],
                    'confidence': solution_data['confidence'],
                    'field': field_name
                })
        
        # Check field-specific patterns
        if field_name == 'site_alarm' and 'site_patterns' in self.field_patterns:
            for pattern, solution in self.field_patterns['site_patterns'].items():
                if pattern in field_value_lower:
                    matches.append({
                        'keyword': pattern,
                        'solution': solution,
                        'confidence': 0.9,
                        'field': field_name
                    })
        
        return matches
    
    def generate_advanced_solution(self, msisdn: str, complaint: str, 
                                 device_type_settings_vpn_apn: str = "", 
                                 signal_strength: str = "",
                                 quality_of_signal: str = "",
                                 site_kpi_alarm: str = "", 
                                 past_data_analysis: str = "",
                                 indoor_outdoor_coverage_issue: str = "") -> str:
        """Generate solution using advanced analysis"""
        
        if not self.model_loaded:
            return "System unavailable - models not loaded properly."
        
        # Analyze all inputs
        analysis = self.analyze_user_input(
            msisdn, complaint, device_type_settings_vpn_apn, signal_strength, 
            quality_of_signal, site_kpi_alarm, past_data_analysis, indoor_outdoor_coverage_issue
        )
        
        # Generate solution based on analysis
        if analysis['best_solution']:
            # Pattern-based solution found
            solution_text = analysis['best_solution']
            
            return f"""
SOLUTION FOR MSISDN: {msisdn}
COMPLAINT CATEGORY: {analysis.get('category_prediction', 'Unknown')}

RECOMMENDED SOLUTION:
{solution_text}
            """.strip()
        
        # Fallback to category-based solution
        elif analysis['category_prediction']:
            category_solutions = {
                'voice_call': """
1. Enable VoLTE on your device (Settings > Network > VoLTE)
2. Check network mode is set to 4G/LTE
3. Restart device and test call quality
4. If using iPhone, check VoWiFi settings
5. Verify you're in a good coverage area""",
                
                'data_internet': """
2. Verify data balance and plan activation
3. Test data speed using speed test app
4. Try switching between 4G/3G modes
5. Clear browser cache and restart device""",
                
                'coverage_signal': """
1. Check if there are planned maintenance activities in your area
2. Move to different location and test signal
3. Check if other users in area have same issue
4. Verify device supports local frequency bands
5. Consider Wi-Fi calling if available""",
                
                'site_infrastructure': """
1. Report site issue to technical operations center
2. Check for maintenance notifications on Mobitel website
3. Verify if issue affects multiple customers
4. Document exact time and location of issue""",
                
                'device_hardware': """
1. Check SIM card is properly inserted and clean
2. Verify device is not blacklisted or stolen
3. Test SIM in different compatible device
4. Check if device software needs updating
5. Ensure device is unlocked for local networks"""
            }
            
            solution = category_solutions.get(analysis['category_prediction'], 
                                            "Contact technical support for specialized assistance")
            
            return f"""
SOLUTION FOR MSISDN: {msisdn}
COMPLAINT CATEGORY: {analysis['category_prediction']} (Confidence: {analysis.get('ml_confidence', 0):.1%})

RECOMMENDED SOLUTION:
{solution}
            """.strip()
        
        # Ultimate fallback
        else:
            return f"""
SOLUTION FOR MSISDN: {msisdn}
STATUS: Unable to automatically determine specific solution

CONTEXT ANALYSIS:
• Complaint: {complaint}
• Device Info: {device_type_settings_vpn_apn if device_type_settings_vpn_apn else 'Not specified'}
• Signal Issues: {signal_strength or quality_of_signal or 'None reported'}
• Site/KPI Info: {site_kpi_alarm if site_kpi_alarm else 'None provided'}
• Coverage Issue: {indoor_outdoor_coverage_issue if indoor_outdoor_coverage_issue else 'Not specified'}

For urgent issues, please call technical support immediately.
            """.strip()

# Global instance
advanced_generator = AdvancedSolutionGenerator()

def generate_solution(msisdn: str, complaint_text: str, 
                     device_type_settings_vpn_apn: str | None = "", 
                     signal_strength: str | None = "",
                     quality_of_signal: str | None = "",
                     site_kpi_alarm: str | None = "", 
                     past_data_analysis: str | None = "",
                     indoor_outdoor_coverage_issue: str | None = "") -> str:
    """Main solution generation function with advanced analysis"""
    return advanced_generator.generate_advanced_solution(
        msisdn, complaint_text, 
        device_type_settings_vpn_apn or "", 
        signal_strength or "", 
        quality_of_signal or "",
        site_kpi_alarm or "", 
        past_data_analysis or "",
        indoor_outdoor_coverage_issue or ""
    )
