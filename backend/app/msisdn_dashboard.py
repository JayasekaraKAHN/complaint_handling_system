from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
from typing import Optional

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# File paths
TAC_FILE = "data/TACD_UPDATED.csv"
VLRD_FILE = "data/VLRD_2025_08 (1).csv"
REFERENCE_LOCATION_FILE = "data/reference_data_cell_locations_20250910 (1).csv"

# Load data files
try:
    tac_df = pd.read_csv(TAC_FILE, low_memory=False)
    vlrd_df = pd.read_csv(VLRD_FILE, sep='\t')
    ref_location_df = pd.read_csv(REFERENCE_LOCATION_FILE, low_memory=False)
    print(f"Loaded TAC data: {len(tac_df)} records")
    print(f"Loaded VLRD data: {len(vlrd_df)} records")
    print(f"Loaded Reference Location data: {len(ref_location_df)} records")
except Exception as e:
    print(f"Error loading data files: {e}")
    tac_df = pd.DataFrame()
    vlrd_df = pd.DataFrame()
    ref_location_df = pd.DataFrame()

def get_common_cell_locations(cell_code, site_name, district, limit=5):
    """
    Find common cell locations from reference data based on cell code, site name, or district
    """
    try:
        if ref_location_df.empty:
            return []
        
        common_locations = []
        
        # Search by cell code first
        if cell_code and cell_code != 'Unknown':
            cell_matches = ref_location_df[
                ref_location_df['cellcode'].astype(str).str.contains(str(cell_code), case=False, na=False)
            ]
            common_locations.extend(cell_matches.to_dict('records'))
        
        # Search by site name
        if site_name and site_name != 'Unknown' and len(common_locations) < limit:
            site_matches = ref_location_df[
                ref_location_df['sitename'].astype(str).str.contains(str(site_name), case=False, na=False)
            ]
            for record in site_matches.to_dict('records'):
                if record not in common_locations:
                    common_locations.append(record)
        
        # Search by district if we still need more
        if district and district != 'Unknown' and len(common_locations) < limit:
            district_matches = ref_location_df[
                ref_location_df['district'].astype(str).str.contains(str(district), case=False, na=False)
            ].head(limit - len(common_locations))
            for record in district_matches.to_dict('records'):
                if record not in common_locations:
                    common_locations.append(record)
        
        # Limit results and format
        formatted_locations = []
        for loc in common_locations[:limit]:
            formatted_loc = {
                'CELL_CODE': loc.get('cellcode', 'Unknown'),
                'SITE_NAME': loc.get('sitename', 'Unknown'),
                'LAT': loc.get('lat', 'Not Found'),
                'LON': loc.get('lon', 'Not Found'),
                'DISTRICT': loc.get('district', 'Unknown'),
                'REGION': loc.get('region', 'Unknown'),
                'LAC': loc.get('lac', 'Unknown'),
                'CELL': loc.get('cellid', 'Unknown'),
                'TYPE': loc.get('type', 'Unknown'),
                'STATUS': loc.get('status', 'Unknown'),
                'PROVINCE': loc.get('province', 'Unknown'),
                'BTS_CONFIGURED_DATE': loc.get('bts_configured_date', 'Unknown'),
                'ONAIR_DATE': loc.get('onair_date', 'Unknown')
            }
            formatted_locations.append(formatted_loc)
        
        return formatted_locations
        
    except Exception as e:
        print(f"Error finding common cell locations: {e}")
        return []

def get_msisdn_data(msisdn):
    """
    Get MSISDN data from VLRD and TAC files, plus common location data
    """
    try:
        # Search in VLRD data
        vlrd_match = vlrd_df[vlrd_df['MSISDN'].astype(str) == str(msisdn)]
        
        if vlrd_match.empty:
            return {"error": "MSISDN not found in VLRD data"}
        
        # Get the first match
        vlrd_row = vlrd_match.iloc[0]
        
        # Extract TAC from the data (first 8 digits)
        tac = str(vlrd_row['TAC'])[:8] if pd.notna(vlrd_row['TAC']) else "Unknown"
        
        # Get device information from TAC data
        device_info = {"Brand": "Unknown", "Model": "Unknown", "Device Type": "Unknown", 
                      "Technology": "Unknown", "VoLTE": "Unknown", "Marketing Name": "Unknown"}
        
        if tac.isdigit():
            tac_match = tac_df[tac_df['tac'] == int(tac)]
            if not tac_match.empty:
                tac_row = tac_match.iloc[0]
                device_info = {
                    "Brand": tac_row.get('brand', 'Unknown'),
                    "Model": tac_row.get('model', 'Unknown'),
                    "Device Type": tac_row.get('device_type', 'Unknown'),
                    "Technology": tac_row.get('technology', 'Unknown'),
                    "VoLTE": tac_row.get('volte', 'Unknown'),
                    "Marketing Name": tac_row.get('marketing_name', 'Unknown'),
                    "OS": tac_row.get('software_os_name', 'Unknown'),
                    "Year Released": tac_row.get('year_released', 'Unknown')
                }
        
        # Get common location data based on LAC and CELL
        common_locations = []
        if not ref_location_df.empty:
            lac = str(vlrd_row.get('LAC', ''))
            cell = str(vlrd_row.get('CELL', ''))
            district = str(vlrd_row.get('DISTRICT', '')).strip()
            
            # Find locations with same LAC or in same district
            if lac and lac != 'Unknown':
                lac_matches = ref_location_df[
                    (ref_location_df['lac'].astype(str) == lac) |
                    (ref_location_df['district'].astype(str).str.strip().str.upper() == district.upper())
                ].head(10)  # Limit to 10 results
                
                for _, row in lac_matches.iterrows():
                    location_data = {
                        'CELL_CODE': row.get('cellcode', 'Unknown'),
                        'SITE_NAME': row.get('sitename', 'Unknown'), 
                        'LAC': row.get('lac', 'Unknown'),
                        'CELL': row.get('cellid', 'Unknown'),
                        'LAT': row.get('lat', 'Not Found'),
                        'LON': row.get('lon', 'Not Found'),
                        'DISTRICT': row.get('district', 'Unknown'),
                        'REGION': row.get('region', 'Unknown'),
                        'TYPE': row.get('type', 'Unknown'),
                        'STATUS': row.get('status', 'Unknown')
                    }
                    common_locations.append(location_data)
        
        # Prepare result
        result = {
            "MSISDN": msisdn,
            "LAC": vlrd_row.get('LAC', 'Unknown'),
            "CELL": vlrd_row.get('CELL', 'Unknown'),
            "CELL_CODE": vlrd_row.get('CELL_CODE', 'Unknown'),
            "SITE_NAME": vlrd_row.get('SITE_NAME', 'Unknown'),
            "DISTRICT": vlrd_row.get('DISTRICT', 'Unknown'),
            "TAC": tac,
            "PREPOS": vlrd_row.get('PREPOS', 'Unknown'),
            "SIMUSIM": vlrd_row.get('SIMUSIM', 'Unknown'),
            "Common_Locations": common_locations,
            **device_info
        }
        
        return result
        
    except Exception as e:
        return {"error": f"Error processing MSISDN data: {str(e)}"}

# API endpoint for dashboard data
@app.get("/api/msisdn_dashboard")
async def api_msisdn_dashboard(msisdn: str = ""):
    try:
        if msisdn:
            result = get_msisdn_data(msisdn)
            if isinstance(result, dict) and "error" in result:
                return JSONResponse({"error": result["error"]}, status_code=404)
            return JSONResponse({"result": result})
        
        return JSONResponse({"message": "Please provide an MSISDN to search"})
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"API Error: {str(e)}\n{error_trace}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "result": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, msisdn: str = Form(...)):
    result = get_msisdn_data(msisdn)
    if "error" in result:
        return templates.TemplateResponse("index.html", {"request": request, "error": result["error"]})
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result})

@app.get("/msisdn_details", response_class=HTMLResponse)
async def msisdn_details_search(request: Request, msisdn: Optional[str] = None):
    if msisdn:
        result = get_msisdn_data(msisdn)
        if "error" in result:
            return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "msisdn": msisdn})
    else:
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "msisdn": None})

@app.get("/msisdn_details/{msisdn}", response_class=HTMLResponse)
async def msisdn_details(request: Request, msisdn: str):
    result = get_msisdn_data(msisdn)
    if "error" in result:
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
    
    return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "msisdn": msisdn})

# Export the function for use in main.py
__all__ = ['get_msisdn_data', 'tac_df', 'vlrd_df', 'ref_location_df']