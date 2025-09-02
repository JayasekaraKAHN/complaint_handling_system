#!/usr/bin/env python3
"""
Location and RSRP Analysis Module for Sri Lankan Telecom Complaint System
Analyzes geographical location and signal strength data to generate targeted solutions
for data and voice issues.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Optional, Tuple
import os
import json

class LocationRSRPAnalyzer:
    def __init__(self, data_file="data/datafinal.csv"):
        self.data_file = data_file
        self.location_patterns = {}
        self.rsrp_thresholds = {
            'excellent': (-85, -70),    # RSRP > -85 dBm
            'good': (-95, -85),         # -95 to -85 dBm  
            'fair': (-105, -95),        # -105 to -95 dBm
            'poor': (-115, -105),       # -115 to -105 dBm
            'very_poor': (-140, -115)   # < -115 dBm
        }
        self.district_solutions = {}
        self.signal_solutions = {}
        self.load_location_data()
    
    def load_location_data(self):
        """Load and analyze location patterns from datafinal.csv"""
        try:
            try:
                df = pd.read_csv(self.data_file, encoding='utf-8')
            except UnicodeDecodeError:
                df = pd.read_csv(self.data_file, encoding='latin1')
            
            self.analyze_district_patterns(df)
            self.analyze_signal_patterns(df)
            self.analyze_location_clusters(df)
            
        except Exception as e:
            print(f"Error loading location data: {e}")
    
    def analyze_district_patterns(self, df):
        """Analyze solution patterns by district"""
        district_solutions = {}
        
        for _, row in df.iterrows():
            try:
                district = str(row.get('DISTRICT', '')).strip()
                solution = str(row.get('Solution', '')).strip()
                issue = str(row.get('Issue_Description', '')).lower()
                
                if district and solution and solution != 'nan':
                    if district not in district_solutions:
                        district_solutions[district] = {
                            'voice_issues': [],
                            'data_issues': [],
                            'coverage_issues': []
                        }
                    
                    # Categorize by issue type
                    if any(keyword in issue for keyword in ['voice', 'call', 'volte']):
                        district_solutions[district]['voice_issues'].append(solution)
                    elif any(keyword in issue for keyword in ['data', 'speed', 'internet']):
                        district_solutions[district]['data_issues'].append(solution)
                    elif any(keyword in issue for keyword in ['coverage', 'signal', 'poor']):
                        district_solutions[district]['coverage_issues'].append(solution)
                        
            except Exception:
                continue
        
        # Find most common solutions per district
        for district, issues in district_solutions.items():
            self.district_solutions[district] = {}
            for issue_type, solutions in issues.items():
                if solutions:
                    # Get most common solution
                    most_common = max(set(solutions), key=solutions.count)
                    self.district_solutions[district][issue_type] = most_common
    
    def analyze_signal_patterns(self, df):
        """Analyze solution patterns by signal strength"""
        signal_solutions = {
            'excellent': {'voice': [], 'data': [], 'coverage': []},
            'good': {'voice': [], 'data': [], 'coverage': []},
            'fair': {'voice': [], 'data': [], 'coverage': []},
            'poor': {'voice': [], 'data': [], 'coverage': []},
            'very_poor': {'voice': [], 'data': [], 'coverage': []}
        }
        
        for _, row in df.iterrows():
            try:
                signal_str = str(row.get('Signal_Strength', ''))
                solution = str(row.get('Solution', '')).strip()
                issue = str(row.get('Issue_Description', '')).lower()
                
                if solution and solution != 'nan':
                    rsrp_value = self.extract_rsrp_value(signal_str)
                    signal_quality = self.categorize_signal_strength(rsrp_value)
                    
                    if signal_quality:
                        # Categorize by issue type
                        if any(keyword in issue for keyword in ['voice', 'call', 'volte']):
                            signal_solutions[signal_quality]['voice'].append(solution)
                        elif any(keyword in issue for keyword in ['data', 'speed', 'internet']):
                            signal_solutions[signal_quality]['data'].append(solution)
                        elif any(keyword in issue for keyword in ['coverage', 'signal', 'poor']):
                            signal_solutions[signal_quality]['coverage'].append(solution)
                            
            except Exception:
                continue
        
        # Get most common solutions per signal quality
        for quality, issue_types in signal_solutions.items():
            self.signal_solutions[quality] = {}
            for issue_type, solutions in issue_types.items():
                if solutions:
                    most_common = max(set(solutions), key=solutions.count)
                    self.signal_solutions[quality][issue_type] = most_common
    
    def extract_rsrp_value(self, signal_str: str) -> Optional[float]:
        """Extract RSRP value from signal strength string"""
        if not signal_str or signal_str.lower() in ['no coverage', 'nan', '-']:
            return None
        
        # Look for RSRP patterns like "-87 to -94 dBm" or "-95dbm"
        rsrp_patterns = [
            r'RSRP[:\s]*(-?\d+)(?:\s*to\s*(-?\d+))?\s*dBm',
            r'(-?\d+)\s*to\s*(-?\d+)\s*dBm',
            r'(-?\d+)\s*dBm'
        ]
        
        for pattern in rsrp_patterns:
            match = re.search(pattern, signal_str, re.IGNORECASE)
            if match:
                groups = match.groups()
                if len(groups) >= 2 and groups[1]:
                    # Range found, take average
                    return (float(groups[0]) + float(groups[1])) / 2
                elif groups and groups[0]:
                    return float(groups[0])
        
        return None
    
    def categorize_signal_strength(self, rsrp_value: Optional[float]) -> Optional[str]:
        """Categorize signal strength based on RSRP value"""
        if rsrp_value is None:
            return 'very_poor'  # No signal
        
        for category, (min_val, max_val) in self.rsrp_thresholds.items():
            if min_val <= rsrp_value < max_val:
                return category
        
        return 'very_poor'  # Default for very weak signals
    
    def analyze_location_clusters(self, df):
        """Analyze geographical clusters and their common issues"""
        location_clusters = {}
        
        for _, row in df.iterrows():
            try:
                lat = float(row.get('Lat', 0))
                lon = float(row.get('Lon', 0))
                district = str(row.get('DISTRICT', '')).strip()
                site_name = str(row.get('SITE_NAME', '')).strip()
                solution = str(row.get('Solution', '')).strip()
                
                if lat and lon and district and solution and solution != 'nan':
                    # Create location key based on district and approximate coordinates
                    location_key = f"{district}_{round(lat, 2)}_{round(lon, 2)}"
                    
                    if location_key not in location_clusters:
                        location_clusters[location_key] = {
                            'solutions': [],
                            'sites': [],
                            'coordinates': (lat, lon),
                            'district': district
                        }
                    
                    location_clusters[location_key]['solutions'].append(solution)
                    if site_name:
                        location_clusters[location_key]['sites'].append(site_name)
                        
            except Exception:
                continue
        
        # Find dominant solutions per location cluster
        for location_key, data in location_clusters.items():
            if data['solutions']:
                most_common = max(set(data['solutions']), key=data['solutions'].count)
                self.location_patterns[location_key] = {
                    'dominant_solution': most_common,
                    'coordinates': data['coordinates'],
                    'district': data['district'],
                    'sites': list(set(data['sites'])),
                    'solution_count': len(data['solutions'])
                }
    
    def get_location_based_solution(self, complaint_text: str, location: str = "", 
                                   lat: Optional[float] = None, lon: Optional[float] = None,
                                   district: str = "") -> Optional[Dict]:
        """Get solution based on location analysis"""
        
        issue_type = self.classify_issue_type(complaint_text)
        
        # Try district-based solution first
        if district and district in self.district_solutions:
            district_solution = self.district_solutions[district].get(issue_type)
            if district_solution:
                return {
                    'solution': district_solution,
                    'reason': f'District-specific solution for {district} area',
                    'confidence': 'high',
                    'analysis_type': 'district_pattern'
                }
        
        # Try coordinate-based solution
        if lat and lon:
            location_key = f"{district}_{round(lat, 2)}_{round(lon, 2)}"
            if location_key in self.location_patterns:
                pattern = self.location_patterns[location_key]
                return {
                    'solution': pattern['dominant_solution'],
                    'reason': f'Location cluster analysis for coordinates ({lat:.2f}, {lon:.2f})',
                    'confidence': 'medium',
                    'analysis_type': 'coordinate_cluster',
                    'supporting_sites': pattern['sites']
                }
        
        return None
    
    def get_rsrp_based_solution(self, complaint_text: str, signal_strength: str = "",
                               rsrp_value: Optional[float] = None) -> Optional[Dict]:
        """Get solution based on RSRP analysis"""
        
        issue_type = self.classify_issue_type(complaint_text)
        
        # Extract RSRP if not provided
        if rsrp_value is None and signal_strength:
            rsrp_value = self.extract_rsrp_value(signal_strength)
        
        signal_quality = self.categorize_signal_strength(rsrp_value)
        
        if signal_quality and signal_quality in self.signal_solutions:
            solution = self.signal_solutions[signal_quality].get(issue_type)
            if solution:
                return {
                    'solution': solution,
                    'reason': f'Signal quality analysis: {signal_quality.replace("_", " ")} ({rsrp_value:.1f} dBm)' if rsrp_value else f'Signal quality: {signal_quality.replace("_", " ")}',
                    'confidence': 'high',
                    'analysis_type': 'rsrp_pattern',
                    'signal_quality': signal_quality,
                    'rsrp_value': rsrp_value
                }
        
        return None
    
    def classify_issue_type(self, complaint_text: str) -> str:
        """Classify complaint into voice, data, or coverage issue"""
        complaint_lower = complaint_text.lower()
        
        if any(keyword in complaint_lower for keyword in ['voice', 'call', 'volte', 'vowifi']):
            return 'voice_issues'
        elif any(keyword in complaint_lower for keyword in ['data', 'speed', 'internet', 'browsing']):
            return 'data_issues'
        elif any(keyword in complaint_lower for keyword in ['coverage', 'signal', 'poor', 'weak']):
            return 'coverage_issues'
        else:
            return 'coverage_issues'  # Default
    
    def generate_enhanced_solution(self, complaint_text: str, location: str = "",
                                 lat: Optional[float] = None, lon: Optional[float] = None,
                                 district: str = "", signal_strength: str = "",
                                 rsrp_value: Optional[float] = None) -> Optional[Dict]:
        """Generate enhanced solution using both location and RSRP analysis"""
        
        # Get location-based solution
        location_solution = self.get_location_based_solution(
            complaint_text, location, lat, lon, district
        )
        
        # Get RSRP-based solution
        rsrp_solution = self.get_rsrp_based_solution(
            complaint_text, signal_strength, rsrp_value
        )
        
        # Combine analyses
        if location_solution and rsrp_solution:
            # Both analyses available - choose higher confidence or merge
            if location_solution['solution'] == rsrp_solution['solution']:
                return {
                    'solution': location_solution['solution'],
                    'reason': f"{location_solution['reason']} + {rsrp_solution['reason']}",
                    'confidence': 'very_high',
                    'analysis_type': 'combined_location_rsrp',
                    'supporting_data': {
                        'location_analysis': location_solution,
                        'rsrp_analysis': rsrp_solution
                    }
                }
            else:
                # Different solutions - prefer RSRP for technical issues
                primary = rsrp_solution if rsrp_value else location_solution
                secondary = location_solution if rsrp_value else rsrp_solution
                
                return {
                    'solution': primary['solution'],
                    'reason': f"{primary['reason']} (primary), {secondary['reason']} (secondary)",
                    'confidence': 'high',
                    'analysis_type': 'prioritized_analysis',
                    'alternative_solution': secondary['solution']
                }
        
        elif location_solution:
            return location_solution
        elif rsrp_solution:
            return rsrp_solution
        
        return None

    def get_analysis_summary(self) -> Dict:
        """Get summary of analysis patterns discovered"""
        return {
            'districts_analyzed': len(self.district_solutions),
            'signal_patterns': len([q for q, solutions in self.signal_solutions.items() if solutions]),
            'location_clusters': len(self.location_patterns),
            'district_coverage': list(self.district_solutions.keys()),
            'signal_qualities': list(self.signal_solutions.keys())
        }

# Global instance
location_rsrp_analyzer = LocationRSRPAnalyzer()

def analyze_location_and_rsrp(complaint_text: str, location: str = "",
                             lat: Optional[float] = None, lon: Optional[float] = None,
                             district: str = "", signal_strength: str = "",
                             rsrp_value: Optional[float] = None) -> Optional[Dict]:
    """Main function to analyze location and RSRP data for solution generation"""
    return location_rsrp_analyzer.generate_enhanced_solution(
        complaint_text, location, lat, lon, district, signal_strength, rsrp_value
    )

def get_location_analysis_summary() -> Dict:
    """Get summary of location analysis capabilities"""
    return location_rsrp_analyzer.get_analysis_summary()
