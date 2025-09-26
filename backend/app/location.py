import math
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import pandas as pd
from typing import List, Dict

# Load the Excel data
file_path = 'data/location_data_mapping.xlsx'
excel_data = pd.read_excel(file_path)

# Haversine function to calculate the distance between two points
def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    lat1_rad, lon1_rad = math.radians(lat1), math.radians(lon1)
    lat2_rad, lon2_rad = math.radians(lat2), math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = math.sin(dlat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


# APIRouter setup
from fastapi import APIRouter

location_router = APIRouter()

class Coordinates(BaseModel):
    lat: float
    lon: float
    radius_km: float

# @location_router.post("/get_location")
# def get_location(coords: Coordinates):
#     user_lat = coords.lat
#     user_lon = coords.lon
#     radius_km = coords.radius_km
#     all_cells = []
#     nearby_sites = []
#     for index, row in excel_data.iterrows():
#         site_lat = row['location_corrected.lat']
#         site_lon = row['location_corrected.lon']
#         distance = haversine_distance(user_lat, user_lon, site_lat, site_lon)
#         cell_info = {
#             "site_name": row["Site Name"],
#             "cell_name": row["Cell Name"],
#             "site_id": row["Site_ID"],
#             "rsrp_range_1": row["RSRP Range 1 (>-105dBm) %"],
#             "rsrp_range_2": row["RSRP Range 2 (-105~-110dBm) %"],
#             "rsrp_range_3": row["RSRP Range 3 (-110~-115dBm) %"],
#             "rsrp_below_115": row["RSRP < -115dBm"],
#             "distance_km": distance
#         }
#         all_cells.append(cell_info)
#         if distance <= radius_km:
#             nearby_sites.append(cell_info)
#     solution = "Based on your complaint, we recommend improving signal strength or relocating the site."
#     all_cells_sorted = sorted(all_cells, key=lambda x: x['distance_km'])
#     return {
#         "solution": solution,
#         "nearby_cells": nearby_sites,
#         "all_cells": all_cells_sorted
#     }


@location_router.get("/location_data")
def get_location_data(lat: float, lon: float):
    min_distance = float('inf')
    nearest_site = None
    nearby_sites = []
    for index, row in excel_data.iterrows():
        site_lat = row['location_corrected.lat']
        site_lon = row['location_corrected.lon']
        distance = haversine_distance(lat, lon, site_lat, site_lon)
        site_info = {
            "site_name": row["Site Name"],
            "rsrp_range_1": row["RSRP Range 1 (>-105dBm) %"],
            "rsrp_range_2": row["RSRP Range 2 (-105~-110dBm) %"],
            "rsrp_range_3": row["RSRP Range 3 (-110~-115dBm) %"],
            "rsrp_below_115": row["RSRP < -115dBm"],
            "distance_km": distance
        }
        if distance < min_distance:
            min_distance = distance
            nearest_site = site_info
        if distance <= 2.0:
            nearby_sites.append(site_info)
    # If no sites within 2km, show only the nearest site in the list
    if not nearby_sites and nearest_site:
        nearby_sites = [nearest_site]
    if nearest_site:
        return JSONResponse({"nearest_site": nearest_site, "nearby_sites": nearby_sites})
    else:
        return JSONResponse({"error": "No nearby site found."}, status_code=404)


# Export for import in main.py
__all__ = ["location_router", "haversine_distance"]
