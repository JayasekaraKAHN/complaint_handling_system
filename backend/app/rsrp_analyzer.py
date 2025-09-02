"""
Enhanced RSRP Analysis System for Sri Lankan Telecom Network
Analyzes RSRP data from Huawei and ZTE equipment to provide intelligent
location-based signal quality assessments and targeted solutions.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple, Optional
import os
from pathlib import Path

class RSRPAnalyzer:
    """Advanced RSRP analyzer for telecom network optimization"""
    
    def __init__(self):
        self.base_path = Path(__file__).parent.parent / "data"
        self.huawei_data = None
        self.zte_data = None
        self.location_data = None
        self.rsrp_thresholds = {
            'excellent': -85,    # > -85 dBm
            'good': -95,         # -85 to -95 dBm  
            'fair': -105,        # -95 to -105 dBm
            'poor': -110,        # -105 to -110 dBm
            'very_poor': -115,   # -110 to -115 dBm
            'critical': -120     # < -115 dBm
        }
        self.load_rsrp_data()
    
    def load_rsrp_data(self):
        """Load RSRP data from CSV files"""
        try:
            # Load Huawei RSRP data
            huawei_path = self.base_path / "Huawei_RSRP.csv"
            if huawei_path.exists():
                self.huawei_data = pd.read_csv(huawei_path)
                print(f"✅ Loaded {len(self.huawei_data)} Huawei RSRP records")
            
            # Load ZTE RSRP data  
            zte_path = self.base_path / "ZTE_RSRP.csv"
            if zte_path.exists():
                self.zte_data = pd.read_csv(zte_path)
                print(f"✅ Loaded {len(self.zte_data)} ZTE RSRP records")
            
            # Load location reference data
            location_path = self.base_path / "Reference_Data_Cell_Locations_20250403.csv"
            if location_path.exists():
                self.location_data = pd.read_csv(location_path)
                print(f"✅ Loaded {len(self.location_data)} location reference records")
                
        except Exception as e:
            print(f"⚠️ Error loading RSRP data: {e}")
    
    def extract_site_info(self, site_name: str) -> Dict[str, str]:
        """Extract location information from site name"""
        # Parse site names like "LKY021_0UL00_Ambagamuwa_Ka(81111)"
        site_info = {
            'site_code': '',
            'district': '',
            'location': '',
            'province': ''
        }
        
        if not site_name:
            return site_info
        
        # Extract site code (first part before underscore)
        parts = site_name.split('_')
        if parts:
            site_info['site_code'] = parts[0]
            
            # Extract district code from site code
            if len(parts[0]) >= 2:
                district_code = parts[0][:2]
                site_info['district'] = district_code
        
        # Extract location name from middle part
        if len(parts) >= 3:
            location_part = parts[2]
            site_info['location'] = location_part
        
        # Map district codes to provinces
        district_to_province = {
            'LK': 'Central', 'KY': 'Central', 'ML': 'Central', 'NU': 'Central',
            'CO': 'Western', 'GM': 'Western', 'KL': 'Western',
            'PT': 'North Western', 'KU': 'North Western',
            'PO': 'North Central', 'AN': 'North Central',
            'BD': 'Sabaragamuwa', 'RT': 'Sabaragamuwa',
            'MT': 'Uva', 'MN': 'Uva',
            'GL': 'Southern', 'MT': 'Southern', 'HB': 'Southern'
        }
        
        site_info['province'] = district_to_province.get(site_info['district'], 'Unknown')
        
        return site_info
    
    def analyze_rsrp_distribution(self, site_id: str) -> Dict[str, float]:
        """Analyze RSRP distribution for a specific site"""
        result = {
            'excellent_pct': 0.0,
            'good_pct': 0.0, 
            'fair_pct': 0.0,
            'poor_pct': 0.0,
            'very_poor_pct': 0.0,
            'critical_pct': 0.0,
            'equipment_type': 'unknown',
            'total_cells': 0
        }
        
        # Check Huawei data
        if self.huawei_data is not None:
            huawei_records = self.huawei_data[self.huawei_data['Site_ID'] == site_id]
            if not huawei_records.empty:
                result['equipment_type'] = 'Huawei'
                result['total_cells'] = len(huawei_records)
                
                # Calculate averages across all cells for this site
                result['excellent_pct'] = self._safe_mean(huawei_records['RSRP Range 1 (>-105dBm) %'])
                result['good_pct'] = self._safe_mean(huawei_records['RSRP Range 2 (-105~-110dBm) %'])
                result['fair_pct'] = self._safe_mean(huawei_records['RSRP Range 3 (-110~-115dBm) %'])
                result['critical_pct'] = self._safe_mean(huawei_records['RSRP < -115dBm'])
                
                return result
        
        # Check ZTE data
        if self.zte_data is not None:
            zte_records = self.zte_data[self.zte_data['Site_ID'] == site_id]
            if not zte_records.empty:
                result['equipment_type'] = 'ZTE'
                result['total_cells'] = len(zte_records)
                
                # Calculate averages for ZTE (different column structure)
                result['excellent_pct'] = self._safe_mean(zte_records['RSRP Range 1 (>-105dBm) %'])
                result['good_pct'] = self._safe_mean(zte_records['RSRP Range 2 (-105~-110dBm) %'])
                result['fair_pct'] = self._safe_mean(zte_records['RSRP Range 3 (-110~-115dBm) %'])
                result['critical_pct'] = self._safe_mean(zte_records['RSRP < -115dBm'])
                
                return result
        
        return result
    
    def _safe_mean(self, series) -> float:
        """Safely calculate mean, handling string percentages and #DIV/0! errors"""
        try:
            # Convert to numeric, handling percentage strings and errors
            numeric_series = pd.to_numeric(series.astype(str).str.replace('%', '').str.replace('#DIV/0!', 'NaN'), errors='coerce')
            return float(numeric_series.mean()) if not numeric_series.isna().all() else 0.0
        except:
            return 0.0
    
    def find_nearby_sites(self, location: str, district: str | None = None) -> List[Dict]:
        """Find sites near a given location"""
        nearby_sites = []
        
        if not location:
            return nearby_sites
        
        # Search in both datasets
        for dataset, equipment in [(self.huawei_data, 'Huawei'), (self.zte_data, 'ZTE')]:
            if dataset is None:
                continue
                
            # Search by location name in Site Name
            location_matches = dataset[dataset['Site Name'].str.contains(location, case=False, na=False)]
            
            # If district provided, also filter by district
            if district:
                district_matches = dataset[dataset['Site Name'].str.contains(district, case=False, na=False)]
                location_matches = pd.concat([location_matches, district_matches]).drop_duplicates()
            
            for _, row in location_matches.head(10).iterrows():  # Limit to 10 matches
                site_info = self.extract_site_info(row['Site Name'])
                rsrp_analysis = self.analyze_rsrp_distribution(row['Site_ID'])
                
                nearby_sites.append({
                    'site_id': row['Site_ID'],
                    'site_name': row['Site Name'],
                    'cell_name': row['Cell Name'],
                    'equipment': equipment,
                    'district': site_info['district'],
                    'location': site_info['location'],
                    'rsrp_analysis': rsrp_analysis
                })
        
        return nearby_sites
    
    def categorize_signal_quality(self, rsrp_dbm: float) -> str:
        """Categorize signal quality based on RSRP value"""
        if rsrp_dbm >= self.rsrp_thresholds['excellent']:
            return 'excellent'
        elif rsrp_dbm >= self.rsrp_thresholds['good']:
            return 'good'
        elif rsrp_dbm >= self.rsrp_thresholds['fair']:
            return 'fair'
        elif rsrp_dbm >= self.rsrp_thresholds['poor']:
            return 'poor'
        elif rsrp_dbm >= self.rsrp_thresholds['very_poor']:
            return 'very_poor'
        else:
            return 'critical'
    
    def analyze_area_signal_quality(self, district: str | None = None, location: str | None = None) -> Dict:
        """Analyze overall signal quality for an area"""
        analysis = {
            'total_sites': 0,
            'huawei_sites': 0,
            'zte_sites': 0,
            'avg_excellent_pct': 0.0,
            'avg_good_pct': 0.0,
            'avg_fair_pct': 0.0,
            'avg_poor_pct': 0.0,
            'avg_critical_pct': 0.0,
            'signal_quality_rating': 'unknown',
            'recommendations': []
        }
        
        # Find sites in the area
        nearby_sites = self.find_nearby_sites(location or '', district)
        
        if not nearby_sites:
            return analysis
        
        analysis['total_sites'] = len(nearby_sites)
        
        # Calculate statistics
        excellent_values = []
        good_values = []
        fair_values = []
        poor_values = []
        critical_values = []
        
        for site in nearby_sites:
            rsrp = site['rsrp_analysis']
            if rsrp['equipment_type'] == 'Huawei':
                analysis['huawei_sites'] += 1
            elif rsrp['equipment_type'] == 'ZTE':
                analysis['zte_sites'] += 1
            
            excellent_values.append(rsrp['excellent_pct'])
            good_values.append(rsrp['good_pct'])
            fair_values.append(rsrp['fair_pct'])
            poor_values.append(rsrp['poor_pct'])
            critical_values.append(rsrp['critical_pct'])
        
        # Calculate averages
        analysis['avg_excellent_pct'] = np.mean(excellent_values) if excellent_values else 0
        analysis['avg_good_pct'] = np.mean(good_values) if good_values else 0
        analysis['avg_fair_pct'] = np.mean(fair_values) if fair_values else 0
        analysis['avg_poor_pct'] = np.mean(poor_values) if poor_values else 0
        analysis['avg_critical_pct'] = np.mean(critical_values) if critical_values else 0
        
        # Determine overall signal quality
        if analysis['avg_excellent_pct'] > 60:
            analysis['signal_quality_rating'] = 'excellent'
        elif analysis['avg_excellent_pct'] + analysis['avg_good_pct'] > 70:
            analysis['signal_quality_rating'] = 'good'
        elif analysis['avg_critical_pct'] > 30:
            analysis['signal_quality_rating'] = 'poor'
        else:
            analysis['signal_quality_rating'] = 'fair'
        
        # Generate recommendations
        analysis['recommendations'] = self._generate_recommendations(analysis)
        
        return analysis
    
    def _generate_recommendations(self, analysis: Dict) -> List[str]:
        """Generate recommendations based on signal analysis"""
        recommendations = []
        
        if analysis['avg_critical_pct'] > 25:
            recommendations.append("Critical signal levels detected - immediate infrastructure upgrade required")
        
        if analysis['avg_poor_pct'] + analysis['avg_critical_pct'] > 40:
            recommendations.append("Poor coverage area - consider additional cell sites or power optimization")
        
        if analysis['avg_excellent_pct'] < 30:
            recommendations.append("Low excellent signal coverage - antenna optimization and power adjustment needed")
        
        if analysis['huawei_sites'] > 0 and analysis['zte_sites'] > 0:
            recommendations.append("Mixed equipment detected - ensure inter-vendor compatibility and optimization")
        
        if analysis['total_sites'] < 3:
            recommendations.append("Limited site density - consider network densification for better coverage")
        
        return recommendations
    
    def generate_targeted_solution(self, complaint_type: str, rsrp_analysis: Dict, 
                                 location_info: Dict) -> Dict[str, str]:
        """Generate targeted solution based on RSRP analysis and complaint type"""
        
        solution_data = {
            'solution': '',
            'reason': '',
            'confidence': 'medium',
            'technical_details': ''
        }
        
        signal_quality = location_info.get('signal_quality_rating', 'unknown')
        critical_pct = rsrp_analysis.get('avg_critical_pct', 0)
        poor_pct = rsrp_analysis.get('avg_poor_pct', 0)
        excellent_pct = rsrp_analysis.get('avg_excellent_pct', 0)
        
        # Data/Voice call issues with poor RSRP
        if 'data' in complaint_type.lower() or 'voice' in complaint_type.lower():
            if critical_pct > 30:
                solution_data['solution'] = "Critical RSRP levels require immediate site power optimization and antenna realignment"
                solution_data['reason'] = f"RSRP analysis shows {critical_pct:.1f}% critical signal levels in this area"
                solution_data['confidence'] = 'very_high'
                solution_data['technical_details'] = "Critical RSRP (<-115 dBm) indicates severe coverage gaps requiring infrastructure intervention"
                
            elif poor_pct + critical_pct > 40:
                solution_data['solution'] = "Poor signal coverage requires network optimization and possible additional cell deployment"
                solution_data['reason'] = f"Combined poor/critical RSRP levels at {poor_pct + critical_pct:.1f}%"
                solution_data['confidence'] = 'high'
                solution_data['technical_details'] = "Poor RSRP distribution indicates coverage optimization needed"
                
            elif excellent_pct < 30:
                solution_data['solution'] = "Serving cell power increase and antenna optimization to enhance coverage"
                solution_data['reason'] = f"Only {excellent_pct:.1f}% excellent signal coverage in area"
                solution_data['confidence'] = 'high'
                solution_data['technical_details'] = "Low excellent RSRP percentage indicates power/antenna adjustment needed"
                
            else:
                solution_data['solution'] = "Signal levels adequate - investigate device-specific or network configuration issues"
                solution_data['reason'] = f"RSRP distribution shows acceptable signal quality ({excellent_pct:.1f}% excellent)"
                solution_data['confidence'] = 'medium'
                solution_data['technical_details'] = "Good RSRP levels suggest non-coverage related issue"
        
        # Coverage issues
        elif 'coverage' in complaint_type.lower():
            if signal_quality == 'poor':
                solution_data['solution'] = "Area requires network densification with additional cell sites"
                solution_data['reason'] = f"Poor overall signal quality with {critical_pct:.1f}% critical RSRP"
                solution_data['confidence'] = 'very_high'
                solution_data['technical_details'] = "Coverage gaps require infrastructure expansion"
            else:
                solution_data['solution'] = "Optimize existing cell parameters and antenna configuration"
                solution_data['reason'] = f"Signal quality is {signal_quality} but can be improved"
                solution_data['confidence'] = 'high'
                solution_data['technical_details'] = "Coverage optimization through parameter adjustment"
        
        # Default solution
        if not solution_data['solution']:
            solution_data['solution'] = "Perform detailed RF analysis and optimize network parameters"
            solution_data['reason'] = "RSRP data requires detailed analysis for optimal solution"
            solution_data['confidence'] = 'medium'
            solution_data['technical_details'] = "Standard network optimization approach"
        
        return solution_data

# Global analyzer instance
rsrp_analyzer = RSRPAnalyzer()

def analyze_rsrp_data(location: str = "", district: str = "", site_id: str = "", 
                      complaint_type: str = "") -> Dict:
    """Main function to analyze RSRP data for location-based solutions"""
    
    # Get area analysis
    area_analysis = rsrp_analyzer.analyze_area_signal_quality(district, location)
    
    # Get specific site analysis if site_id provided
    site_analysis = {}
    if site_id:
        site_analysis = rsrp_analyzer.analyze_rsrp_distribution(site_id)
    
    # Generate targeted solution
    solution = rsrp_analyzer.generate_targeted_solution(
        complaint_type, area_analysis, area_analysis
    )
    
    return {
        'area_analysis': area_analysis,
        'site_analysis': site_analysis,
        'solution': solution['solution'],
        'reason': solution['reason'],
        'confidence': solution['confidence'],
        'technical_details': solution['technical_details'],
        'signal_quality': area_analysis['signal_quality_rating'],
        'recommendations': area_analysis['recommendations']
    }
