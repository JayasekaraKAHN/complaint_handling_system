from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
import sys
import os

# Add the backend directory to Python path
backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from app.prompts import SOLUTION_PROMPT
from app.enhanced_solution import generate_solution  # Use enhanced solution generator
from app.vlrd_handler import get_vlrd_handler

# ----------------- FastAPI App -----------------
app = FastAPI(title="Complaint Solution API")

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

# ----------------- Routes -----------------
@app.get("/", response_class=HTMLResponse)
async def root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/api/solution")
async def get_solution(req: ComplaintRequest):
    """
    Receives complaint input from frontend, generates personalized prompt,
    and returns solution.
    """
    try:
        # Call your updated generate_solution function with all input fields
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

@app.get("/msisdn_details", response_class=HTMLResponse)
async def msisdn_details_page(request: Request, msisdn: str | None = None):
    """
    Display MSISDN details page with search functionality
    """
    try:
        msisdn_data = []
        if msisdn:
            vlrd_handler = get_vlrd_handler()
            msisdn_data = vlrd_handler.search_msisdn(msisdn)
        
        return templates.TemplateResponse(
            "msisdn_details.html", 
            {
                "request": request, 
                "msisdn": msisdn, 
                "msisdn_data": msisdn_data
            }
        )
    except Exception as e:
        return templates.TemplateResponse(
            "msisdn_details.html", 
            {
                "request": request, 
                "msisdn": msisdn, 
                "msisdn_data": [],
                "error": str(e)
            }
        )

@app.get("/api/msisdn/{msisdn}")
async def get_msisdn_data(msisdn: str):
    """
    API endpoint to get MSISDN data in JSON format
    """
    try:
        vlrd_handler = get_vlrd_handler()
        msisdn_data = vlrd_handler.search_msisdn(msisdn)
        
        return JSONResponse({
            "msisdn": msisdn,
            "data": msisdn_data,
            "found": len(msisdn_data) > 0,
            "count": len(msisdn_data),
            "status": "success"
        })
    except Exception as e:
        return JSONResponse(
            {"error": str(e), "status": "error"}, 
            status_code=500
        )


# ----------------- Run Server -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
