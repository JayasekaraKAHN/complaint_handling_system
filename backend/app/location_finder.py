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
        # Get the absolute path to the data file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        backend_dir = os.path.dirname(current_dir)
        self.data_file = os.path.join(backend_dir, 'data', 'reference_data_cell_locations_20250910 (1).csv')
        self.cell_data = None
        self.load_cell_data()
    
    def load_cell_data(self):
        """Load cell location data from CSV file."""
        try:
            self.cell_data = pd.read_csv(self.data_file)
            print(f"Loaded {len(self.cell_data)} cell locations from reference data")
        except Exception as e:
            print(f"Error loading cell data: {e}")
            self.cell_data = pd.DataFrame()
    
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

def get_location_info_for_complaint(latitude: float, longitude: float) -> Dict:
    """
    Get location information specifically formatted for complaint resolution.
    
    Args:
        latitude: User's latitude from complaint form
        longitude: User's longitude from complaint form
        
    Returns:
        Formatted location information for display
    """
    result = location_service.find_nearest_sites(latitude, longitude, 2.0)
    
    if result.get("error"):
        return {"error": result["error"]}
    
    nearest = result.get("nearest_site")
    sites_in_range = result.get("sites_within_range", [])
    
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
        }
    }
    
    # Clean NaN values for JSON serialization
    return final_result