from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import sys
import os
from app.location import location_router

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app.simple_solution import generate_solution  # Use simplified solution generator

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

@app.post("/api/solution")
async def get_solution(req: ComplaintRequest):
    """
    Receives complaint input from frontend, generates personalized solution.
    """
    try:
        # Call simplified solution generation
        solution_text = generate_solution(
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

        return JSONResponse({"solution": solution_text, "status": "success"})
    
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
