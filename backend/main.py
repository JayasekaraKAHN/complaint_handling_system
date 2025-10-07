import sys
import os
from fastapi import FastAPI, Request, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from typing import Optional

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app.location import location_router
from app.simple_solution import generate_solution  # Use simplified solution generator

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
    msisdn: str
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
        # If MSISDN is provided, get data for that MSISDN
        if msisdn:
            result = get_msisdn_data(msisdn)
            if isinstance(result, dict) and "error" in result:
                return JSONResponse({"error": result["error"]}, status_code=404)
        
        # Return empty response for now - can be enhanced later
        return JSONResponse({"table": [], "dash_url": "/dashboard/usage-graph/"})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
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
            longitude=req.longitude,
            latitude=req.latitude,
        )

        return JSONResponse({
            "solutions": solution_data["solutions"],
            "solution_type": solution_data["solution_type"],
            "status": "success"
        })
    
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
