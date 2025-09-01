from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from datetime import datetime
from .app.prompts import SOLUTION_PROMPT
from .app.solution import generate_solution  # Your updated solution.py

# ----------------- FastAPI App -----------------
app = FastAPI(title="Complaint Solution API")

# Allow CORS for dev (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="backend/templates")

# ----------------- Request Model -----------------
class ComplaintRequest(BaseModel):
    msisdn: str
    complaint: str
    location: str | None = None
    site_alarm: str | None = None
    kpi: str | None = None
    billing: str | None = None

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
    # Call your updated generate_solution function with all input fields
    solution_text = generate_solution(
        msisdn=req.msisdn,
        complaint_text=req.complaint,
        location=req.location,
        site_alarm=req.site_alarm,
        kpi=req.kpi,
        billing=req.billing,
    )

    return JSONResponse({"solution": solution_text})


# ----------------- Run Server -----------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
