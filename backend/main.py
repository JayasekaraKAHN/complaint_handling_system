import sys
import os
import math
import numpy as np
import pandas as pd
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

def clean_json_data(obj):
    """
    Recursively clean data to make it JSON-serializable by handling NaN, inf, and numpy types
    """
    if isinstance(obj, dict):
        return {k: clean_json_data(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_json_data(v) for v in obj]
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return None
        return val
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    elif pd.isna(obj):
        return None
    else:
        return obj

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app.location import location_router
from app.simple_solution import generate_solution  # Use simplified solution generator
from app.location_finder import get_location_info_for_complaint

# Import MSISDN dashboard data and functions
from app.msisdn_dashboard import get_msisdn_data, tac_df, vlrd_df, ref_location_df

# ----------------- FastAPI App -----------------
app = FastAPI(title="Simplified Complaint Solution API")

# Mount static files
app.mount("/static", StaticFiles(directory=os.path.join(backend_dir, "static")), name="static")

app.include_router(location_router, prefix="/location", tags=["location"])

# Allow CORS for dev (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory=os.path.join(backend_dir, "templates"))

sys.path.append(os.path.join(backend_dir, "app"))

# ----------------- Request Model -----------------
class ComplaintRequest(BaseModel):
    msisdn: str | None = None
    complaint: str
    device_type_settings_vpn_apn: str | None = None
    signal_strength: str | None = None
    quality_of_signal: str | None = None
    site_kpi_alarm: str | None = None
    past_data_analysis: str | None = None
    indoor_outdoor_coverage_issue: str | None = None
    location: str | None = None
    longitude: float | None = None
    latitude: float | None = None
    
    def model_post_init(self, __context):
        """Clean and validate coordinates after model initialization"""
        import math
        # Clean longitude
        if self.longitude is not None:
            if math.isnan(self.longitude) or math.isinf(self.longitude):
                self.longitude = None
        # Clean latitude  
        if self.latitude is not None:
            if math.isnan(self.latitude) or math.isinf(self.latitude):
                self.latitude = None

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/solution_template", response_class=HTMLResponse)
async def solution_template(request: Request):
    return templates.TemplateResponse("solution_template.html", {"request": request})

# MSISDN Dashboard Routes
@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard_route(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "result": None})

@app.get("/api/msisdn_dashboard")
async def api_msisdn_dashboard_route(msisdn: str = ""):
    try:
        print(f"API called with MSISDN: {msisdn}")  # Debug logging
        # If MSISDN is provided, get data for that MSISDN
        if msisdn:
            result = get_msisdn_data(msisdn)
            print(f"get_msisdn_data returned: {type(result)}")  # Debug logging
            if isinstance(result, dict) and "error" in result:
                print(f"Error in result: {result['error']}")  # Debug logging
                return JSONResponse({"error": result["error"]}, status_code=404)
            print(f"Returning result with Usage_Data: {'Usage_Data' in result if isinstance(result, dict) else 'Not a dict'}")  # Debug logging
            
            # Clean result data for JSON serialization
            result_serializable = clean_json_data(result)
            return JSONResponse({"result": result_serializable})
        
        # Return empty response for now - can be enhanced later
        return JSONResponse({"table": [], "dash_url": "/dashboard/usage-graph/"})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"Exception in API: {str(e)}\n{error_trace}")  # Debug logging
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/msisdn_details", response_class=HTMLResponse)
async def msisdn_details_route(request: Request, msisdn: Optional[str] = None):
    if msisdn:
        result = get_msisdn_data(msisdn)
        if "error" in result:
            return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "msisdn": msisdn})
    else:
        # No MSISDN provided, render template with search form only
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "msisdn": None})

@app.get("/msisdn_details/{msisdn}", response_class=HTMLResponse)
async def msisdn_details_path_route(request: Request, msisdn: str):
    result = get_msisdn_data(msisdn)
    if "error" in result:
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
    
    return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "msisdn": msisdn})

@app.post("/api/solution")
async def get_solution(req: ComplaintRequest):
    """
    Receives complaint input from frontend, generates personalized solution.
    """
    try:
        # Clean input coordinates to handle NaN values
        import math
        longitude = req.longitude
        latitude = req.latitude
        
        # Convert NaN or inf values to None
        if longitude is not None and (math.isnan(longitude) or math.isinf(longitude)):
            longitude = None
        if latitude is not None and (math.isnan(latitude) or math.isinf(latitude)):
            latitude = None
        
        # Get location information if coordinates are provided
        location_info = None
        if latitude and longitude:
            try:
                location_info = get_location_info_for_complaint(latitude, longitude)
            except Exception as loc_error:
                print(f"Location lookup error: {loc_error}")
                location_info = {"error": f"Location lookup failed: {str(loc_error)}"}
        
        # Call simplified solution generation - now returns structured data
        solution_data = generate_solution(
            msisdn=req.msisdn,
            complaint_text=req.complaint,
            device_type_settings_vpn_apn=req.device_type_settings_vpn_apn,
            signal_strength=req.signal_strength,
            quality_of_signal=req.quality_of_signal,
            site_kpi_alarm=req.site_kpi_alarm,
            past_data_analysis=req.past_data_analysis,
            indoor_outdoor_coverage_issue=req.indoor_outdoor_coverage_issue,
            location=req.location,
            longitude=longitude,  # Use cleaned longitude
            latitude=latitude,    # Use cleaned latitude
        )

        # Clean solution data for JSON serialization
        cleaned_solution_data = clean_json_data(solution_data)

        response_data = {
            "solutions": cleaned_solution_data.get("solutions") if isinstance(cleaned_solution_data, dict) else solution_data.get("solutions"),
            "solution_type": cleaned_solution_data.get("solution_type") if isinstance(cleaned_solution_data, dict) else solution_data.get("solution_type"),
            "status": "success"
        }
        
        # Add location information if available
        if location_info:
            response_data["location_info"] = clean_json_data(location_info)

        return JSONResponse(response_data)
    
    except Exception as e:
        # Return error details for debugging
        import traceback
        error_details = traceback.format_exc()
        return JSONResponse(
            {"error": str(e), "details": error_details, "status": "error"}, 
            status_code=500
        )

# Removed MSISDN details endpoints to simplify the system


# ----------------- Run Server -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
