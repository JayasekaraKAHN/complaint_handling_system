from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from .solution import generate_solution

router = APIRouter()

@router.get("/solution/{msisdn}")
def get_solution(msisdn: str):
    # GET: no complaint provided, generate from MSISDN only
    solution = generate_solution(msisdn=msisdn, complaint_text=None)
    return {"msisdn": msisdn, "solution": solution}

@router.post("/api/solution")
async def post_solution(request: Request):
    data = await request.json()
    # Accept multiple key variants gracefully
    msisdn = data.get("msisdn") or data.get("MSISDN") or data.get("Impacted MSISDN") or ""
    complaint = data.get("complaint") or data.get("Complaint") or ""
    solution = generate_solution(msisdn=msisdn, complaint_text=complaint)
    # Format solution for user-friendly UI
    if solution and solution != "MSISDN not found in dataset. No solution available.":
        formatted = f"<div style='padding:8px 0;'><strong>Recommended Solution:</strong><br><span style='color:#0078d7;font-size:1.08em;'>{solution}</span></div>"
    else:
        formatted = "<div style='padding:8px 0;'><strong>No solution found for this complaint.</strong></div>"
    return JSONResponse({"solution": formatted})
