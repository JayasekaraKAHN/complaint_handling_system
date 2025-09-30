"""
Simplified complaint handling solution system - Paragraph Format (No Specific Locations)
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
import re

# Add parent directory to path to import prompts
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from prompts import create_complaint_solution_prompt, create_pattern_analysis_prompt, create_new_complaint_prompt
except ImportError:
    print("Warning: Could not import prompts module. Using fallback prompts.")
    
    def create_complaint_solution_prompt(complaint_details, similar_cases=None, location_info=None):
        return f"Generate paragraph solutions for: {complaint_details.get('complaint', 'N/A')}"
    
    def create_pattern_analysis_prompt(complaint_text, historical_data):
        return f"Analyze patterns for: {complaint_text} in paragraph format"
    
    def create_new_complaint_prompt(complaint_details, location_context=None):
        return f"Generate new paragraph solutions for: {complaint_details.get('complaint', 'N/A')}"

def remove_specific_locations(text: str) -> str:
    """
    Remove specific location names like COL313I and numerical values from text
    """
    if not text or not text.strip():
        return text
    
    # Remove specific site names (patterns like COL313I, ABC123, etc.)
    text = re.sub(r'\b[A-Z]+\d+[A-Z]*\b', 'the area', text)
    
    # Remove specific numerical coordinates and measurements
    text = re.sub(r'\b\d+\.\d+\b', '', text)  # Remove decimal numbers
    text = re.sub(r'\b\d+\b', '', text)  # Remove integers
    text = re.sub(r'\b\d+dBm\b', 'current signal level', text)  # Remove dBm values
    text = re.sub(r'\b\d+%\b', '', text)  # Remove percentages
    
    # Clean up multiple spaces and commas
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r',\s*,', ',', text)
    text = re.sub(r'\s,', ',', text)
    
    return text.strip()

def format_solution_paragraphs(solution_text: str, solution_type: str) -> str:
    """
    Format solution text into proper paragraph format and remove specific locations
    """
    if not solution_text or not solution_text.strip():
        return "No solution generated."
    
    # Clean the solution text of specific locations and numbers
    cleaned_text = remove_specific_locations(solution_text.strip())
    
    # Ensure paragraph format is consistent
    paragraphs = []
    
    # Split by numbered points
    raw_paragraphs = re.split(r'\n\s*\d+\.', cleaned_text)
    raw_paragraphs = [para.strip() for para in raw_paragraphs if para.strip()]
    
    if len(raw_paragraphs) >= 2:
        # Use the detected paragraphs with numbering
        for i, paragraph in enumerate(raw_paragraphs[:4], 1):  # Max 4 paragraphs
            if paragraph:
                # Clean up the paragraph text and remove any remaining location references
                clean_para = remove_specific_locations(paragraph)
                clean_para = re.sub(r'^\s*[A-Z ]+:\s*', '', clean_para)  # Remove prefix labels if any
                paragraphs.append(f"{i}. {clean_para.strip()}")
    else:
        # If no clear paragraphs detected, split by double newlines
        raw_paragraphs = re.split(r'\n\s*\n', cleaned_text)
        raw_paragraphs = [para.strip() for para in raw_paragraphs if para.strip()]
        
        for i, paragraph in enumerate(raw_paragraphs[:4], 1):
            if paragraph:
                clean_para = remove_specific_locations(paragraph)
                paragraphs.append(f"{i}. {clean_para}")
    
    # Ensure we have at least one paragraph
    if not paragraphs:
        paragraphs = ["1. Please contact technical support for detailed analysis and assistance with this issue."]
    
    # Add metadata
    formatted_solution = "\n\n".join(paragraphs)
    formatted_solution += f"\n\n[Solution type: {solution_type}]"
    
    return formatted_solution

def limit_to_paragraphs(text: str, max_paragraphs: int = 4) -> str:
    """
    Limit text to maximum number of paragraphs and remove specific locations
    """
    if not text or not text.strip():
        return text
    
    # Clean text first
    clean_text = remove_specific_locations(text)
    
    # Split by numbered points or double newlines
    paragraphs = re.split(r'\n\s*\d+\.|\n\s*\n', clean_text.strip())
    paragraphs = [para.strip() for para in paragraphs if para.strip()]
    
    # Take only the first max_paragraphs
    if len(paragraphs) > max_paragraphs:
        paragraphs = paragraphs[:max_paragraphs]
    
    # Reformat with consistent numbering
    result = '\n\n'.join([f"{i+1}. {para}" for i, para in enumerate(paragraphs)])
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
        # Normalize complaint text and remove locations for signature matching
        normalized_complaint = remove_specific_locations(complaint_text.lower().strip())
        
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
            
            hist_issue_desc = remove_specific_locations(str(row.get('Issue Description', '')))
            hist_signature = self.create_complaint_signature(hist_issue_desc, hist_conditions)
            
            if current_signature == hist_signature:
                # Format the exact match solution in paragraph format and remove locations
                exact_solution = str(row.get('Solution', 'No solution found'))
                return format_solution_paragraphs(exact_solution, "exact_match")
        
        return None
    
    def get_location_context(self, complaint_details: Dict) -> Optional[Dict]:
        """Get location context from coordinates or location name (without specific site names)"""
        if self.location_data is None or self.location_data.empty:
            return None
        
        # Try to find location context using coordinates
        longitude = complaint_details.get('longitude')
        latitude = complaint_details.get('latitude')
        location_name = complaint_details.get('location')
        
        location_context = None
        
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
                location_context = {
                    'coverage_quality': 'Good' if float(str(closest_location.get('RSRP >-105dBm (%)', '0')).replace('%', '')) > 70 else 'Poor',
                    'signal_distribution': f"Good coverage, Fair coverage, Poor coverage based on area signal quality"
                }
        
        # Try to find by location name if coordinates not available
        if not location_context and location_name:
            for _, row in self.location_data.iterrows():
                site_name = str(row.get('Site Name', '')).lower()
                if location_name.lower() in site_name or site_name in location_name.lower():
                    location_context = {
                        'coverage_quality': 'Good' if float(str(row.get('RSRP >-105dBm (%)', '0')).replace('%', '')) > 70 else 'Poor',
                        'signal_distribution': f"Good coverage, Fair coverage, Poor coverage based on area signal quality"
                    }
                    break
        
        return location_context
    
    def find_similar_complaints(self, complaint_text: str, top_n: int = 5) -> List[Dict]:
        """Find similar complaints with different conditions"""
        if self.complaint_data is None or self.complaint_data.empty:
            return []
        
        # Clean complaint text for similarity matching
        clean_complaint = remove_specific_locations(complaint_text.lower())
        complaint_words = set(clean_complaint.split())
        similarities = []
        
        for _, row in self.complaint_data.iterrows():
            issue_desc = remove_specific_locations(str(row.get('Issue Description', '')))
            issue_words = set(issue_desc.lower().split())
            
            # Calculate Jaccard similarity
            intersection = len(complaint_words & issue_words)
            union = len(complaint_words | issue_words)
            similarity = intersection / union if union > 0 else 0
            
            if similarity > 0.1:  # Only include if some similarity
                # Clean the solution of specific locations before returning
                row_data = row.to_dict()
                if 'Solution' in row_data:
                    row_data['Solution'] = remove_specific_locations(str(row_data['Solution']))
                if 'Issue Description' in row_data:
                    row_data['Issue Description'] = remove_specific_locations(str(row_data['Issue Description']))
                
                similarities.append({
                    'similarity': similarity,
                    'data': row_data
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
                        "max_tokens": 350,
                        "top_p": 0.9
                    }
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                # Clean the response of any specific locations that might have been generated
                response_text = result.get('response', 'Unable to generate solution')
                return remove_specific_locations(response_text)
            else:
                return f"LLM Error: HTTP {response.status_code}"
                
        except requests.exceptions.ConnectionError:
            return "Error: Unable to connect to Ollama. Please ensure Ollama is running with llama3.2:1b model."
        except Exception as e:
            return f"Error calling LLM: {str(e)}"
    
    def generate_solution(self, complaint_details: Dict) -> Tuple[str, str]:
        """
        Main method to generate solution following the specified logic
        """
        complaint_text = complaint_details.get('complaint', '')
        
        # Step 1: Check for exact match (same complaint + same conditions)
        exact_solution = self.find_exact_match(complaint_text, complaint_details)
        if exact_solution:
            return exact_solution, "exact_match"
        
        # Get location context for LLM prompts (without specific site names)
        location_context = self.get_location_context(complaint_details)
        
        # Step 2: Check for similar complaints (different conditions)
        similar_complaints = self.find_similar_complaints(complaint_text)
        if similar_complaints:
            if location_context and len(similar_complaints) >= 3:
                prompt = create_complaint_solution_prompt(complaint_details, similar_complaints, location_context)
                solution = self.call_ollama_llm(prompt)
                return format_solution_paragraphs(solution, "comprehensive_analysis"), "comprehensive_analysis"
            else:
                prompt = create_pattern_analysis_prompt(complaint_text, similar_complaints)
                solution = self.call_ollama_llm(prompt)
                return format_solution_paragraphs(solution, "pattern_analysis"), "pattern_analysis"
        
        # Step 3: New complaint - use LLM with location context
        prompt = create_new_complaint_prompt(complaint_details, location_context)
        solution = self.call_ollama_llm(prompt)
        return format_solution_paragraphs(solution, "new_complaint"), "new_complaint"

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
    Simplified interface for generating solutions without specific location names
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
    
    return solution