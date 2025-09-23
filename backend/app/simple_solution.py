"""
Simplified complaint handling solution system
"""

import pandas as pd
import pickle
import hashlib
import numpy as np
import os
import requests
import json
from typing import Dict, List, Optional, Tuple
import sys

# Add parent directory to path to import prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prompts import create_complaint_solution_prompt, create_pattern_analysis_prompt, create_new_complaint_prompt
except ImportError:
    print("Warning: Could not import prompts module. Using fallback prompts.")
    
    def create_complaint_solution_prompt(complaint_details, similar_cases=None, location_info=None):
        return f"Generate a solution for: {complaint_details.get('complaint', 'N/A')}"
    
    def create_pattern_analysis_prompt(complaint_text, historical_data):
        return f"Analyze patterns for: {complaint_text}"
    
    def create_new_complaint_prompt(complaint_details, location_context=None):
        return f"Generate new solution for: {complaint_details.get('complaint', 'N/A')}"

def limit_to_sentences(text: str, max_sentences: int = 5) -> str:
    """
    Limit text to maximum number of sentences
    """
    if not text or not text.strip():
        return text
    
    # Split by common sentence endings
    import re
    sentences = re.split(r'[.!?]+', text.strip())
    
    # Remove empty sentences and clean up
    sentences = [s.strip() for s in sentences if s.strip()]
    
    # Take only the first max_sentences
    if len(sentences) > max_sentences:
        sentences = sentences[:max_sentences]
    
    # Rejoin with periods
    result = '. '.join(sentences)
    if result and not result.endswith('.'):
        result += '.'
    
    return result

class SimpleComplaintHandler:
    def __init__(self, model_path: str = "models/unified_complaint_model.pkl"):
        self.model_path = model_path
        self.complaint_data = None
        self.location_data = None
        self.ollama_url = "http://localhost:11434"
        
        # Load the unified model
        self.load_model()
    
    def load_model(self):
        """Load the unified model containing all complaint and location data"""
        try:
            if os.path.exists(self.model_path):
                with open(self.model_path, 'rb') as f:
                    model_data = pickle.load(f)
                self.complaint_data = model_data.get('complaint_data')
                self.location_data = model_data.get('location_data')
                print(f"Loaded unified model with {len(self.complaint_data)} complaints and {len(self.location_data)} locations")
            else:
                print(f"Model file {self.model_path} not found. Please train the model first.")
                self.complaint_data = pd.DataFrame()
                self.location_data = pd.DataFrame()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.complaint_data = pd.DataFrame()
            self.location_data = pd.DataFrame()
    
    def create_complaint_signature(self, complaint_text: str, conditions: Dict | None = None) -> str:
        """Create a unique signature for a complaint based on text and ONLY filled conditions"""
        # Normalize complaint text
        normalized_complaint = complaint_text.lower().strip()
        
        # Include ONLY filled relevant conditions in signature for exact matching
        condition_str = ""
        if conditions:
            # All columns from Information_Gathering_form.xlsx that affect the solution
            all_conditions = [
                'device_type_settings_vpn_apn', 'signal_strength', 'quality_of_signal', 
                'site_kpi_alarm', 'past_data_analysis', 'indoor_outdoor_coverage_issue',
                'location', 'longitude', 'latitude'
            ]
            condition_values = []
            for key in all_conditions:
                value = conditions.get(key)
                # Only include non-empty, meaningful values
                if value is not None and str(value).strip() not in ['', 'n/a', 'none', 'null', 'undefined']:
                    # For coordinates, round to reasonable precision and ensure they're valid
                    if key in ['longitude', 'latitude'] and isinstance(value, (int, float)):
                        if abs(float(value)) > 0.001:  # Only include non-zero coordinates
                            condition_values.append(f"{key}:{round(float(value), 6)}")
                    else:
                        clean_value = str(value).lower().strip()
                        if clean_value:  # Only add if not empty after cleaning
                            condition_values.append(f"{key}:{clean_value}")
            condition_str = "|".join(sorted(condition_values))  # Sort for consistency
        
        # Create hash signature
        signature_input = f"{normalized_complaint}|{condition_str}"
        return hashlib.md5(signature_input.encode()).hexdigest()
    
    def find_exact_match(self, complaint_text: str, conditions: Dict | None = None) -> Optional[str]:
        """Find exact match for the same complaint under same conditions"""
        if self.complaint_data is None or self.complaint_data.empty:
            return None
        
        current_signature = self.create_complaint_signature(complaint_text, conditions)
        
        # Check each row in complaint data
        for _, row in self.complaint_data.iterrows():
            # Create signature for historical complaint using ONLY filled columns
            hist_conditions = {}
            
            # Only add fields that have meaningful values in the historical data
            if pd.notna(row.get('Device type/settings/VPN/APN')) and str(row.get('Device type/settings/VPN/APN', '')).strip():
                hist_conditions['device_type_settings_vpn_apn'] = str(row.get('Device type/settings/VPN/APN', ''))
            
            if pd.notna(row.get('Signal Strength')) and str(row.get('Signal Strength', '')).strip():
                hist_conditions['signal_strength'] = str(row.get('Signal Strength', ''))
            
            if pd.notna(row.get('Qulity of Signal')) and str(row.get('Qulity of Signal', '')).strip():
                hist_conditions['quality_of_signal'] = str(row.get('Qulity of Signal', ''))
            
            if pd.notna(row.get('Site KPI/Alarm')) and str(row.get('Site KPI/Alarm', '')).strip():
                hist_conditions['site_kpi_alarm'] = str(row.get('Site KPI/Alarm', ''))
            
            if pd.notna(row.get('Past Data analysis')) and str(row.get('Past Data analysis', '')).strip():
                hist_conditions['past_data_analysis'] = str(row.get('Past Data analysis', ''))
            
            if pd.notna(row.get('Indoor/Outdoor coverage issue')) and str(row.get('Indoor/Outdoor coverage issue', '')).strip():
                hist_conditions['indoor_outdoor_coverage_issue'] = str(row.get('Indoor/Outdoor coverage issue', ''))
            
            # For coordinates, only include if they are valid numbers and not zero
            try:
                lon_val = row.get('Lon')
                if pd.notna(lon_val):
                    lon_float = float(lon_val)
                    if abs(lon_float) > 0.001:
                        hist_conditions['longitude'] = lon_float
            except (ValueError, TypeError):
                # Skip invalid coordinate values
                pass
            
            try:
                lat_val = row.get('Lat')
                if pd.notna(lat_val):
                    lat_float = float(lat_val)
                    if abs(lat_float) > 0.001:
                        hist_conditions['latitude'] = lat_float
            except (ValueError, TypeError):
                # Skip invalid coordinate values
                pass
            
            hist_signature = self.create_complaint_signature(
                str(row.get('Issue Description', '')), 
                hist_conditions
            )
            
            if current_signature == hist_signature:
                return str(row.get('Solution', 'No solution found'))
        
        return None
    
    def get_location_context(self, complaint_details: Dict) -> Optional[Dict]:
        """Get location context from coordinates or location name"""
        if self.location_data is None or self.location_data.empty:
            return None
        
        # Try to find location context using coordinates
        longitude = complaint_details.get('longitude')
        latitude = complaint_details.get('latitude')
        location_name = complaint_details.get('location')
        
        if longitude is not None and latitude is not None:
            # Find closest location data point
            min_distance = float('inf')
            closest_location = None
            
            for _, row in self.location_data.iterrows():
                try:
                    row_lon = float(row.get('Lon', 0))
                    row_lat = float(row.get('Lat', 0))
                    
                    # Simple distance calculation
                    distance = ((longitude - row_lon) ** 2 + (latitude - row_lat) ** 2) ** 0.5
                    
                    if distance < min_distance and distance < 0.01:  # Within ~1km
                        min_distance = distance
                        closest_location = row.to_dict()
                except (ValueError, TypeError):
                    continue
            
            if closest_location:
                return {
                    'site_name': closest_location.get('Site Name', 'Unknown'),
                    'rsrp_range_1': closest_location.get('RSRP >-105dBm (%)', 'N/A'),
                    'rsrp_range_2': closest_location.get('RSRP -105~-110dBm (%)', 'N/A'),
                    'rsrp_range_3': closest_location.get('RSRP -110~-115dBm (%)', 'N/A'),
                    'rsrp_weak': closest_location.get('RSRP <-115dBm (%)', 'N/A'),
                    'coverage_quality': 'Good' if float(str(closest_location.get('RSRP >-105dBm (%)', '0')).replace('%', '')) > 70 else 'Poor'
                }
        
        # Try to find by location name if coordinates not available
        if location_name:
            for _, row in self.location_data.iterrows():
                site_name = str(row.get('Site Name', '')).lower()
                if location_name.lower() in site_name or site_name in location_name.lower():
                    return {
                        'site_name': row.get('Site Name', 'Unknown'),
                        'rsrp_range_1': row.get('RSRP >-105dBm (%)', 'N/A'),
                        'rsrp_range_2': row.get('RSRP -105~-110dBm (%)', 'N/A'),
                        'rsrp_range_3': row.get('RSRP -110~-115dBm (%)', 'N/A'),
                        'rsrp_weak': row.get('RSRP <-115dBm (%)', 'N/A'),
                        'coverage_quality': 'Good' if float(str(row.get('RSRP >-105dBm (%)', '0')).replace('%', '')) > 70 else 'Poor'
                    }
        
        return None
    
    def find_similar_complaints(self, complaint_text: str, top_n: int = 5) -> List[Dict]:
        """Find similar complaints with different conditions"""
        if self.complaint_data is None or self.complaint_data.empty:
            return []
        
        # Simple text similarity based on common words
        complaint_words = set(complaint_text.lower().split())
        similarities = []
        
        for _, row in self.complaint_data.iterrows():
            issue_desc = str(row.get('Issue Description', ''))
            issue_words = set(issue_desc.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(complaint_words & issue_words)
            union = len(complaint_words | issue_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.1:  # Only include if some similarity
                similarities.append({
                    'similarity': similarity,
                    'data': row.to_dict()
                })
        
        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['data'] for item in similarities[:top_n]]
        """Find similar complaints with different conditions"""
        if self.complaint_data is None or self.complaint_data.empty:
            return []
        
        # Simple text similarity based on common words
        complaint_words = set(complaint_text.lower().split())
        similarities = []
        
        for _, row in self.complaint_data.iterrows():
            issue_desc = str(row.get('Issue Description', ''))
            issue_words = set(issue_desc.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(complaint_words & issue_words)
            union = len(complaint_words | issue_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.1:  # Only include if some similarity
                similarities.append({
                    'similarity': similarity,
                    'data': row.to_dict()
                })
        
        # Sort by similarity and return top_n
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        return [item['data'] for item in similarities[:top_n]]
    
    def call_ollama_llm(self, prompt: str, model: str = "llama3.2:1b") -> str:
        """Call Ollama LLM with the given prompt"""
        try:
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": 0.7,
                        "max_tokens": 150,  # Reduced for more concise responses
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', 'Unable to generate solution')
            else:
                return f"LLM Error: HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to Ollama. Please ensure Ollama is running with llama3.2:1b model."
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def generate_solution(self, complaint_details: Dict) -> Tuple[str, str]:
        """
        Main method to generate solution following the specified logic:
        1. Same complaint + same conditions -> same solution
        2. Similar complaint + different conditions -> pattern analysis
        3. New complaint -> LLM generation with context
        
        Returns: (solution_text, solution_type)
        """
        complaint_text = complaint_details.get('complaint', '')
        
        # Step 1: Check for exact match (same complaint + same conditions)
        exact_solution = self.find_exact_match(complaint_text, complaint_details)
        if exact_solution:
            return exact_solution, "exact_match"
        
        # Get location context for LLM prompts
        location_context = self.get_location_context(complaint_details)
        
        # Step 2: Check for similar complaints (different conditions)
        similar_complaints = self.find_similar_complaints(complaint_text)
        if similar_complaints:
            # If we have location context and multiple similar cases, use comprehensive prompt
            if location_context and len(similar_complaints) >= 3:
                prompt = create_complaint_solution_prompt(complaint_details, similar_complaints, location_context)
                solution = self.call_ollama_llm(prompt)
                return solution, "comprehensive_analysis"
            else:
                # Use pattern analysis prompt for simpler cases
                prompt = create_pattern_analysis_prompt(complaint_text, similar_complaints)
                solution = self.call_ollama_llm(prompt)
                return solution, "pattern_analysis"
        
        # Step 3: New complaint - use LLM with location context
        prompt = create_new_complaint_prompt(complaint_details, location_context)
        solution = self.call_ollama_llm(prompt)
        return solution, "new_complaint"

# Global handler instance
complaint_handler = None

def get_complaint_handler():
    """Get or create the global complaint handler instance"""
    global complaint_handler
    if complaint_handler is None:
        complaint_handler = SimpleComplaintHandler()
    return complaint_handler

def generate_solution(msisdn: str, complaint_text: str, **kwargs) -> str:
    """
    Simplified interface for generating solutions - considers ALL columns for exact matching
    Returns simplified answers (3-5 sentences). Exact matches return the same solution from dataset.
    """
    handler = get_complaint_handler()
    
    complaint_details = {
        'msisdn': msisdn,
        'complaint': complaint_text,
        'device_type_settings_vpn_apn': kwargs.get('device_type_settings_vpn_apn'),
        'signal_strength': kwargs.get('signal_strength'),
        'quality_of_signal': kwargs.get('quality_of_signal'),
        'site_kpi_alarm': kwargs.get('site_kpi_alarm'),
        'past_data_analysis': kwargs.get('past_data_analysis'),
        'indoor_outdoor_coverage_issue': kwargs.get('indoor_outdoor_coverage_issue'),
        'location': kwargs.get('location'),
        'longitude': kwargs.get('longitude'),
        'latitude': kwargs.get('latitude'),
    }
    
    solution, solution_type = handler.generate_solution(complaint_details)
    
    # For exact matches, return the solution from dataset but limit to max 5 sentences
    if solution_type == "exact_match":
        return limit_to_sentences(solution.strip(), max_sentences=5)
    
    # For other types, add brief metadata and ensure solution is concise
    return f"{solution.strip()}\n\n[Solution type: {solution_type}]"