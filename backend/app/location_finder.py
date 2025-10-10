import math
import pandas as pd
from typing import Dict, List, Optional, Tuple
import json

def clean_nan_values(obj):
    """Recursively clean NaN values from nested dictionaries and lists for JSON serialization."""
    if isinstance(obj, dict):
        return {key: clean_nan_values(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [clean_nan_values(item) for item in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    elif pd.isna(obj):
        return None
    else:
        return obj

class LocationService:
    def __init__(self):
        """Initialize the LocationService with reference data."""
        import os
        # Get the absolute path to the data files
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        self.data_file = os.path.join(backend_dir, 'data', 'reference_data_cell_locations_20250910 (1).csv')
        self.location_mapping_file = os.path.join(backend_dir, 'data', 'location_data_mapping.xlsx')
        self.cell_data = None
        self.location_mapping_data = None
        self.load_cell_data()
        self.load_location_mapping_data()
    
    def load_cell_data(self):
        """Load cell location data from CSV file."""
        try:
            self.cell_data = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.cell_data)} cell locations from reference data")
        except Exception as e:
            print(f"Error loading cell data: {e}")
            self.cell_data = pd.DataFrame()
    
    def load_location_mapping_data(self):
        """Load location mapping data with RSRP and utilization info from Excel file."""
        try:
            self.location_mapping_data = pd.read_excel(self.location_mapping_file)
            # Update column mapping for new Excel structure
            self.location_mapping_data = self.location_mapping_data.rename(columns={
                'location_corrected.lat': 'Lat',
                'location_corrected.lon': 'Lon',
                'RSRP Range 1 (>-105dBm) %': 'RSRP_Level_1',
                'RSRP Range 2 (-105~-110dBm) %': 'RSRP_Level_2', 
                'RSRP Range 3 (-110~-115dBm) %': 'RSRP_Level_3',
                'RSRP < -115dBm': 'RSRP_Level_4'
            })
            print(f"Loaded {len(self.location_mapping_data)} location mapping records with RSRP data")
        except Exception as e:
            print(f"Error loading location mapping data: {e}")
            self.location_mapping_data = pd.DataFrame()
    
    def haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """
        Calculate the great circle distance between two points on Earth using Haversine formula.
        Returns distance in kilometers.
        """
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlon = lon2_rad - lon1_rad
        
        a = (math.sin(dlat / 2) ** 2 + 
             math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2) ** 2)
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        
        # Earth's radius in kilometers
        R = 6371.0
        
        return R * c
    
    def find_nearest_sites(self, user_lat: float, user_lon: float, max_distance_km: float = 2.0) -> dict:
        """
        Find the nearest cell sites within the specified distance from given coordinates.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            max_distance_km: Maximum distance to search for sites (default 2km)
            
        Returns:
            Dictionary containing nearest site info and list of all sites within range
        """
        if self.cell_data is None or self.cell_data.empty:
            return {"error": "Cell data not available"}
        
        sites_within_range = []
        min_distance = float('inf')
        nearest_site = None
        
        for index, row in self.cell_data.iterrows():
            try:
                # Convert lat/lon from string to float and check for NaN
                site_lat = float(row['lat'])
                site_lon = float(row['lon'])
                
                # Skip if coordinates are NaN or invalid
                if math.isnan(site_lat) or math.isnan(site_lon):
                    continue
                
                # Validate coordinates are within Sri Lanka bounds
                # Sri Lanka: Latitude ~5.9-9.9, Longitude ~79.4-81.9
                if not (5.5 <= site_lat <= 10.0 and 79.0 <= site_lon <= 82.0):
                    continue
                
                # Calculate distance
                distance = self.haversine_distance(user_lat, user_lon, site_lat, site_lon)
                
                # Site information
                site_info = {
                    "site_id": row.get('key', 'Unknown'),
                    "cell_code": row.get('cellcode', 'Unknown'),
                    "site_name": row.get('sitename', 'Unknown'),
                    "site_name_long": row.get('site_name_long', ''),
                    "cell_id": row.get('cellid', 'Unknown'),
                    "lac": row.get('lac', 'Unknown'),
                    "cgi": row.get('cgi', 'Unknown'),
                    "technology": row.get('type', 'Unknown'),
                    "region": row.get('region', 'Unknown'),
                    "district": row.get('district', 'Unknown'),
                    "province": row.get('province', 'Unknown'),
                    "latitude": site_lat,
                    "longitude": site_lon,
                    "distance_km": round(distance, 3),
                    "bearing": str(row.get('bore', 'Unknown')),  # Convert to string to avoid NaN
                    "status": str(row.get('status', 'Unknown')).strip()
                }
                
                # Check if within range
                if distance <= max_distance_km:
                    sites_within_range.append(site_info)
                
                # Track nearest site
                if distance < min_distance:
                    min_distance = distance
                    nearest_site = site_info
                    
            except (ValueError, TypeError) as e:
                # Skip rows with invalid lat/lon data
                continue
        
        # Sort sites by distance
        sites_within_range.sort(key=lambda x: x['distance_km'])
        
        # If no sites within range but we have a nearest site, include it
        if not sites_within_range and nearest_site:
            sites_within_range = [nearest_site]
        
        result = {
            "nearest_site": nearest_site,
            "sites_within_range": sites_within_range,
            "total_sites_found": len(sites_within_range),
            "search_radius_km": max_distance_km,
            "user_coordinates": {"lat": user_lat, "lon": user_lon}
        }
        
        # Clean NaN values for JSON serialization
        return result
    
    def get_site_by_coordinates(self, lat: float, lon: float, tolerance_km: float = 0.1) -> Optional[Dict]:
        """
        Find a site at specific coordinates (within tolerance).
        
        Args:
            lat: Target latitude
            lon: Target longitude
            tolerance_km: Distance tolerance in kilometers
            
        Returns:
            Site information if found, None otherwise
        """
        if self.cell_data is None or self.cell_data.empty:
            return None
        
        for index, row in self.cell_data.iterrows():
            try:
                site_lat = float(row['lat'])
                site_lon = float(row['lon'])
                
                distance = self.haversine_distance(lat, lon, site_lat, site_lon)
                
                if distance <= tolerance_km:
                    return {
                        "site_id": row.get('key', 'Unknown'),
                        "cell_code": row.get('cellcode', 'Unknown'),
                        "site_name": row.get('sitename', 'Unknown'),
                        "site_name_long": row.get('site_name_long', ''),
                        "distance_km": round(distance, 3)
                    }
            except (ValueError, TypeError):
                continue
        
        return None

    def get_rsrp_and_utilization_analysis(self, user_lat: float, user_lon: float, max_distance_km: float = 2.0) -> Dict:
        """
        Get RSRP and utilization analysis for locations near the given coordinates.
        
        Args:
            user_lat: User's latitude
            user_lon: User's longitude
            max_distance_km: Maximum distance to search for analysis data
            
        Returns:
            Dictionary containing RSRP and utilization analysis
        """
        if self.location_mapping_data is None or self.location_mapping_data.empty:
            return {"error": "Location mapping data not available"}
        
        nearby_locations = []
        min_distance = float('inf')
        closest_location = None
        
        for index, row in self.location_mapping_data.iterrows():
            try:
                # Get coordinates from location mapping data
                site_lat = float(row.get('Lat', 0))
                site_lon = float(row.get('Lon', 0))
                
                # Skip if coordinates are NaN or invalid
                if math.isnan(site_lat) or math.isnan(site_lon) or site_lat == 0 or site_lon == 0:
                    continue
                
                # Calculate distance
                distance = self.haversine_distance(user_lat, user_lon, site_lat, site_lon)
                
                if distance <= max_distance_km:
                    # Extract RSRP and utilization data using new column structure
                    location_analysis = {
                        "site_name": str(row.get('Site Name', 'Unknown')),
                        "distance_km": round(distance, 3),
                        "latitude": site_lat,
                        "longitude": site_lon,
                        "rsrp_analysis": {
                            "rsrp_level_1": self._safe_percentage(row.get('RSRP_Level_1', 0)),  # >-105dBm
                            "rsrp_level_2": self._safe_percentage(row.get('RSRP_Level_2', 0)),  # -105~-110dBm
                            "rsrp_level_3": self._safe_percentage(row.get('RSRP_Level_3', 0)),  # -110~-115dBm
                            "rsrp_level_4": self._safe_percentage(row.get('RSRP_Level_4', 0))   # <-115dBm
                        },
                        "coverage_quality": self._assess_coverage_quality_new(row),
                        "network_performance": "Good"  # Default since utilization columns not in new file
                    }
                    nearby_locations.append(location_analysis)
                
                # Track closest location
                if distance < min_distance:
                    min_distance = distance
                    closest_location = location_analysis if distance <= max_distance_km else None
                    
            except (ValueError, TypeError) as e:
                continue
        
        # Sort by distance
        nearby_locations.sort(key=lambda x: x['distance_km'])
        
        # Generate overall analysis
        overall_analysis = self._generate_overall_analysis(nearby_locations)
        
        result = {
            "user_coordinates": {"lat": user_lat, "lon": user_lon},
            "closest_location": closest_location,
            "nearby_locations": nearby_locations[:5],  # Limit to top 5
            "total_locations_analyzed": len(nearby_locations),
            "search_radius_km": max_distance_km,
            "overall_analysis": overall_analysis
        }
        
        return result
    
    def _safe_percentage(self, value) -> float:
        """Safely convert percentage values, handling NaN and invalid data."""
        try:
            if pd.isna(value):
                return 0.0
            if isinstance(value, str):
                # Remove % symbol if present
                value = value.replace('%', '').strip()
            return float(value)
        except (ValueError, TypeError):
            return 0.0
    
    def _assess_coverage_quality_new(self, row) -> str:
        """
        Assess coverage quality based on new RSRP logic:
        If average(RSRP Level 1 + RSRP Level 2) > average(RSRP Level 3 + RSRP Level 4) 
        then signal quality = "Good"
        """
        try:
            # Get RSRP level percentages
            level_1 = self._safe_percentage(row.get('RSRP_Level_1', 0))  # >-105dBm  
            level_2 = self._safe_percentage(row.get('RSRP_Level_2', 0))  # -105~-110dBm
            level_3 = self._safe_percentage(row.get('RSRP_Level_3', 0))  # -110~-115dBm
            level_4 = self._safe_percentage(row.get('RSRP_Level_4', 0))  # <-115dBm
            
            # Calculate averages as per your logic
            good_signal_avg = (level_1 + level_2) / 2 if (level_1 + level_2) > 0 else 0
            poor_signal_avg = (level_3 + level_4) / 2 if (level_3 + level_4) > 0 else 0
            
            # Apply your RSRP logic
            if good_signal_avg > poor_signal_avg:
                # Further categorize based on strength of good signals
                if level_1 > 50:  # Strong excellent signals
                    return "Excellent"
                elif good_signal_avg > 60:  # High percentage of good signals
                    return "Good"
                else:
                    return "Fair"
            else:
                # Poor signal conditions
                if poor_signal_avg > 70:  # Predominantly poor signals
                    return "Poor"
                else:
                    return "Fair"
                    
        except Exception as e:
            print(f"Error assessing coverage quality: {e}")
            return "Unknown"
    
    def _assess_coverage_quality(self, row) -> str:
        """Assess coverage quality based on RSRP data."""
        try:
            good_rsrp = self._safe_percentage(row.get('RSRP >-105dBm (%)', 0))
            fair_rsrp = self._safe_percentage(row.get('RSRP -105 to -115dBm (%)', 0))
            poor_rsrp = self._safe_percentage(row.get('RSRP <-115dBm (%)', 0))
            
            if good_rsrp > 70:
                return "Excellent"
            elif good_rsrp > 50:
                return "Good"
            elif fair_rsrp > 60:
                return "Fair"
            else:
                return "Poor"
        except:
            return "Unknown"
    
    def _assess_network_performance(self, row) -> str:
        """Assess network performance based on utilization data."""
        try:
            high_util = self._safe_percentage(row.get('Utilization >90% (%)', 0))
            medium_util = self._safe_percentage(row.get('Utilization 70-90% (%)', 0))
            low_util = self._safe_percentage(row.get('Utilization 0-70% (%)', 0))
            
            if high_util > 50:
                return "Congested"
            elif medium_util > 60:
                return "Moderate Load"
            elif low_util > 70:
                return "Low Load"
            else:
                return "Balanced"
        except:
            return "Unknown"
    
    def _generate_overall_analysis(self, locations: List[Dict]) -> Dict:
        """Generate overall analysis from multiple locations using new RSRP logic."""
        if not locations:
            return {
                "coverage_summary": "No data available",
                "performance_summary": "No data available",
                "recommendations": ["Unable to analyze - no nearby location data found"]
            }
        
        # Aggregate RSRP data using new structure
        good_signal_sites = 0
        total_sites = len(locations)
        
        total_level_1 = 0
        total_level_2 = 0
        total_level_3 = 0
        total_level_4 = 0
        
        for loc in locations:
            rsrp = loc['rsrp_analysis']
            level_1 = rsrp.get('rsrp_level_1', 0)
            level_2 = rsrp.get('rsrp_level_2', 0)
            level_3 = rsrp.get('rsrp_level_3', 0)
            level_4 = rsrp.get('rsrp_level_4', 0)
            
            # Apply your RSRP logic per site
            good_signal_avg = (level_1 + level_2) / 2 if (level_1 + level_2) > 0 else 0
            poor_signal_avg = (level_3 + level_4) / 2 if (level_3 + level_4) > 0 else 0
            
            if good_signal_avg > poor_signal_avg:
                good_signal_sites += 1
            
            # Aggregate for overall percentages
            total_level_1 += level_1
            total_level_2 += level_2
            total_level_3 += level_3
            total_level_4 += level_4
        
        # Calculate overall averages
        avg_level_1 = total_level_1 / total_sites
        avg_level_2 = total_level_2 / total_sites
        avg_level_3 = total_level_3 / total_sites
        avg_level_4 = total_level_4 / total_sites
        
        avg_good_signals = (avg_level_1 + avg_level_2) / 2
        avg_poor_signals = (avg_level_3 + avg_level_4) / 2
        
        good_sites_percentage = (good_signal_sites / total_sites) * 100
        
        # Generate coverage summary based on your logic
        if avg_good_signals > avg_poor_signals:
            if good_sites_percentage > 80:
                coverage_summary = "Excellent signal coverage in the area"
            elif good_sites_percentage > 60:
                coverage_summary = "Good signal coverage in the area"
            else:
                coverage_summary = "Fair signal coverage in the area"
        else:
            coverage_summary = "Poor signal coverage detected in the area"
        
        # Performance summary (simplified since no utilization data in new file)
        performance_summary = "Network performance data not available in current dataset"
        
        # Generate recommendations based on your RSRP logic
        recommendations = []
        if avg_good_signals <= avg_poor_signals:
            recommendations.append("Signal strength improvement needed")
            recommendations.append("Consider indoor signal boosters or positioning near windows")
        elif good_sites_percentage > 70:
            recommendations.append("Good signal coverage - standard troubleshooting should work")
        else:
            recommendations.append("Mixed signal conditions - location-specific solutions may be needed")
        
        if avg_level_1 > 50:
            recommendations.append("Excellent signal strength available for optimal performance")
        
        if not recommendations:
            recommendations.append("Standard network optimization should resolve most issues")
        
        return {
            "coverage_summary": coverage_summary,
            "performance_summary": performance_summary,
            "good_signal_sites_percentage": round(good_sites_percentage, 1),
            "average_excellent_signal_percentage": round(avg_level_1, 1),
            "average_good_signal_percentage": round(avg_level_2, 1),
            "average_poor_signal_percentage": round(avg_level_4, 1),
            "signal_quality_logic": f"Good signals avg: {round(avg_good_signals, 1)}% vs Poor signals avg: {round(avg_poor_signals, 1)}%",
            "recommendations": recommendations
        }

# Global instance for easy import
location_service = LocationService()

def find_nearest_sites_from_coordinates(lat: float, lon: float, max_distance: float = 2.0) -> Dict:
    """
    Convenience function to find nearest sites from coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        max_distance: Maximum search distance in km
        
    Returns:
        Dictionary with site information
    """
    return location_service.find_nearest_sites(lat, lon, max_distance)

def get_rsrp_utilization_analysis(lat: float, lon: float, max_distance: float = 2.0) -> Dict:
    """
    Convenience function to get RSRP and utilization analysis from coordinates.
    
    Args:
        lat: Latitude
        lon: Longitude
        max_distance: Maximum search distance in km
        
    Returns:
        Dictionary with RSRP and utilization analysis
    """
    return location_service.get_rsrp_and_utilization_analysis(lat, lon, max_distance)

def get_location_info_for_complaint(latitude: float, longitude: float) -> Dict:
    """
    Get location information specifically formatted for complaint resolution.
    
    Args:
        latitude: User's latitude from complaint form
        longitude: User's longitude from complaint form
        
    Returns:
        Formatted location information for display including RSRP and utilization analysis
    """
    # Get basic site information
    basic_result = location_service.find_nearest_sites(latitude, longitude, 2.0)
    
    # Get RSRP and utilization analysis
    rsrp_analysis = location_service.get_rsrp_and_utilization_analysis(latitude, longitude, 2.0)
    
    if basic_result.get("error") and rsrp_analysis.get("error"):
        return {"error": "Location and network analysis data not available"}
    
    nearest = basic_result.get("nearest_site")
    sites_in_range = basic_result.get("sites_within_range", [])
    
    final_result = {
        "user_location": {
            "latitude": latitude,
            "longitude": longitude
        },
        "nearest_site": nearest,
        "sites_within_2km": sites_in_range,
        "total_nearby_sites": len(sites_in_range),
        "location_analysis": {
            "has_nearby_coverage": len(sites_in_range) > 0,
            "nearest_distance_km": nearest["distance_km"] if nearest else None,
            "coverage_quality": "Good" if len(sites_in_range) >= 3 else "Limited" if len(sites_in_range) > 0 else "Poor"
        },
        "network_analysis": rsrp_analysis if not rsrp_analysis.get("error") else None,
        # Add RSRP analysis at top level for easier frontend access
        "rsrp_analysis": rsrp_analysis.get("overall_analysis", {}) if not rsrp_analysis.get("error") else {},
        "overall_analysis": rsrp_analysis.get("overall_analysis", {}) if not rsrp_analysis.get("error") else {},
        "combined_recommendations": []
    }
    
    # Generate combined recommendations based on both analyses
    recommendations = []
    
    # Basic coverage recommendations
    if len(sites_in_range) == 0:
        recommendations.append("No nearby cell towers detected - consider location-based solutions")
    elif len(sites_in_range) < 3:
        recommendations.append("Limited tower coverage - signal may be weak")
    
    # RSRP-based recommendations
    if not rsrp_analysis.get("error") and rsrp_analysis.get("overall_analysis"):
        overall_analysis = rsrp_analysis["overall_analysis"]
        recommendations.extend(overall_analysis.get("recommendations", []))
        
        # Add specific technical recommendations based on new RSRP logic
        good_sites_percentage = overall_analysis.get("good_signal_sites_percentage", 0)
        avg_excellent = overall_analysis.get("average_excellent_signal_percentage", 0)
        avg_poor = overall_analysis.get("average_poor_signal_percentage", 0)
        
        if good_sites_percentage < 50:
            recommendations.append("Majority of nearby sites show poor signal quality - consider signal boosters")
        elif good_sites_percentage > 80:
            recommendations.append("Excellent signal coverage detected - basic troubleshooting should resolve issues")
        
        if avg_excellent > 50:
            recommendations.append("Strong signal strength available - check device settings and positioning")
        
        if avg_poor > 30:
            recommendations.append("High percentage of poor signal areas - indoor coverage may be affected")
        
        # Network utilization recommendations (if available)
        if overall_analysis.get("average_high_utilization_percentage", 0) > 40:
            recommendations.append("Network congestion detected - try using services during off-peak hours")
    
    final_result["combined_recommendations"] = recommendations[:5]  # Limit to top 5 recommendations
    
    # Clean NaN values for JSON serialization
    cleaned_result = clean_nan_values(final_result)
    return cleaned_result if isinstance(cleaned_result, dict) else final_result