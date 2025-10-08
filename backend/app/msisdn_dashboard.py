from fastapi.responses import JSONResponse
from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import math
from typing import Optional

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# File paths
TAC_FILE = "data/TACD_UPDATED.csv"
VLRD_FILE = "data/VLRD_2025_08 (1).csv"
REFERENCE_LOCATION_FILE = "data/reference_data_cell_locations_20250910 (1).csv"

# Usage data files
USERTD_FILES = {
    "2025-06": "data/USERTD_2025_06.txt",
    "2025-07": "data/USERTD_2025_07.txt", 
    "2025-08": "data/USERTD_2025_08.txt"
}

# Load data files
try:
    tac_df = pd.read_csv(TAC_FILE, low_memory=False)
    vlrd_df = pd.read_csv(VLRD_FILE, sep='\t')
    ref_location_df = pd.read_csv(REFERENCE_LOCATION_FILE, low_memory=False)
    print(f"Loaded TAC data: {len(tac_df)} records")
    print(f"Loaded VLRD data: {len(vlrd_df)} records")
    print(f"Loaded Reference Location data: {len(ref_location_df)} records")
    
    # Load usage data files (these are large, so we'll load them on demand)
    usertd_data = {}
    for month, file_path in USERTD_FILES.items():
        try:
            # Test if file exists and is readable
            import os
            if os.path.exists(file_path):
                print(f"USERTD file available for {month}: {file_path}")
                usertd_data[month] = file_path
            else:
                print(f"USERTD file not found for {month}: {file_path}")
        except Exception as e:
            print(f"Error checking USERTD file for {month}: {e}")
    
except Exception as e:
    print(f"Error loading data files: {e}")
    tac_df = pd.DataFrame()
    vlrd_df = pd.DataFrame()
    ref_location_df = pd.DataFrame()
    usertd_data = {}

def get_usage_data(msisdn):
    """
    Get usage data for a specific MSISDN from USERTD files
    Returns data for the last 3 months in chronological order
    """
    usage_data = {
        "months": ["2025-06", "2025-07", "2025-08"],
        "outgoing_voice": [],
        "incoming_voice": [],
        "outgoing_sms": [],
        "incoming_sms": [],
        "volume_2g_mb": [],
        "volume_3g_mb": [],
        "volume_4g_mb": [],
        "volume_5g_mb": [],
        "total_data_mb": []
    }
    
    try:
        for month in usage_data["months"]:
            if month in usertd_data:
                file_path = usertd_data[month]
                # Convert to absolute path if it's relative
                import os
                if not os.path.isabs(file_path):
                    file_path = os.path.abspath(file_path)
                
                found_data = False
                
                # Read file in chunks to handle large files efficiently
                try:
                    # Use grep-like approach to find the specific MSISDN
                    import subprocess
                    import sys
                    
                    if sys.platform == "win32":
                        # Windows findstr command (remove '^' for literal match)
                        result = subprocess.run(
                            ['findstr', f'{msisdn}', file_path],
                            capture_output=True, text=True, timeout=30
                        )
                        print(f"findstr output for {msisdn} in {file_path}: {result.stdout}")
                        if result.returncode == 0 and result.stdout.strip():
                            # Find the line that starts with the MSISDN
                            lines = [l for l in result.stdout.strip().split('\n') if l.strip().startswith(str(msisdn))]
                            if lines:
                                line = lines[0].strip()
                                # Split by whitespace (tabs/spaces) and filter out empty strings
                                parts = [p for p in line.split() if p]
                                print(f"Matched line for {msisdn}: {parts}")
                                if len(parts) >= 9 and parts[0] == str(msisdn):
                                    # Parse the usage data
                                    try:
                                        outgoing_voice = float(parts[1]) if parts[1] and parts[1] != '' else 0
                                        incoming_voice = float(parts[2]) if parts[2] and parts[2] != '' else 0
                                        outgoing_sms = float(parts[3]) if parts[3] and parts[3] != '' else 0
                                        incoming_sms = float(parts[4]) if parts[4] and parts[4] != '' else 0
                                        volume_2g = float(parts[5]) if parts[5] and parts[5] != '' else 0
                                        volume_3g = float(parts[6]) if parts[6] and parts[6] != '' else 0
                                        volume_4g = float(parts[7]) if parts[7] and parts[7] != '' else 0
                                        volume_5g = float(parts[8]) if parts[8] and parts[8] != '' else 0
                                        
                                        usage_data["outgoing_voice"].append(outgoing_voice)
                                        usage_data["incoming_voice"].append(incoming_voice)
                                        usage_data["outgoing_sms"].append(outgoing_sms)
                                        usage_data["incoming_sms"].append(incoming_sms)
                                        usage_data["volume_2g_mb"].append(volume_2g)
                                        usage_data["volume_3g_mb"].append(volume_3g)
                                        usage_data["volume_4g_mb"].append(volume_4g)
                                        usage_data["volume_5g_mb"].append(volume_5g)
                                        usage_data["total_data_mb"].append(volume_2g + volume_3g + volume_4g + volume_5g)
                                        found_data = True
                                        print(f"Successfully parsed data for {msisdn} in {month}: voice_out={outgoing_voice}, data_total={volume_2g + volume_3g + volume_4g + volume_5g}")
                                    except (ValueError, IndexError) as e:
                                        print(f"Error parsing usage data for {msisdn}: {e}")
                                        found_data = False
                    else:
                        # Unix/Linux grep command
                        result = subprocess.run(
                            ['grep', f'^{msisdn}', file_path],
                            capture_output=True, text=True, timeout=30
                        )
                        if result.returncode == 0 and result.stdout.strip():
                            line = result.stdout.strip().split('\n')[0]  # Get first match
                            # Split by whitespace (tabs/spaces) and filter out empty strings
                            parts = [p for p in line.split() if p]
                            if len(parts) >= 9 and parts[0] == str(msisdn):
                                # Parse the usage data (same as Windows)
                                try:
                                    outgoing_voice = float(parts[1]) if parts[1] and parts[1] != '' else 0
                                    incoming_voice = float(parts[2]) if parts[2] and parts[2] != '' else 0
                                    outgoing_sms = float(parts[3]) if parts[3] and parts[3] != '' else 0
                                    incoming_sms = float(parts[4]) if parts[4] and parts[4] != '' else 0
                                    volume_2g = float(parts[5]) if parts[5] and parts[5] != '' else 0
                                    volume_3g = float(parts[6]) if parts[6] and parts[6] != '' else 0
                                    volume_4g = float(parts[7]) if parts[7] and parts[7] != '' else 0
                                    volume_5g = float(parts[8]) if parts[8] and parts[8] != '' else 0
                                    
                                    usage_data["outgoing_voice"].append(outgoing_voice)
                                    usage_data["incoming_voice"].append(incoming_voice)
                                    usage_data["outgoing_sms"].append(outgoing_sms)
                                    usage_data["incoming_sms"].append(incoming_sms)
                                    usage_data["volume_2g_mb"].append(volume_2g)
                                    usage_data["volume_3g_mb"].append(volume_3g)
                                    usage_data["volume_4g_mb"].append(volume_4g)
                                    usage_data["volume_5g_mb"].append(volume_5g)
                                    usage_data["total_data_mb"].append(volume_2g + volume_3g + volume_4g + volume_5g)
                                    found_data = True
                                except (ValueError, IndexError) as e:
                                    print(f"Error parsing usage data for {msisdn}: {e}")
                                    found_data = False
                                
                except subprocess.TimeoutExpired:
                    print(f"Timeout searching for MSISDN {msisdn} in {file_path}")
                except Exception as e:
                    print(f"Error searching MSISDN {msisdn} in {file_path}: {e}")
                
                # If no data found, append zeros
                if not found_data:
                    usage_data["outgoing_voice"].append(0)
                    usage_data["incoming_voice"].append(0)
                    usage_data["outgoing_sms"].append(0)
                    usage_data["incoming_sms"].append(0)
                    usage_data["volume_2g_mb"].append(0)
                    usage_data["volume_3g_mb"].append(0)
                    usage_data["volume_4g_mb"].append(0)
                    usage_data["volume_5g_mb"].append(0)
                    usage_data["total_data_mb"].append(0)
            else:
                # File not available, append zeros
                usage_data["outgoing_voice"].append(0)
                usage_data["incoming_voice"].append(0)
                usage_data["outgoing_sms"].append(0)
                usage_data["incoming_sms"].append(0)
                usage_data["volume_2g_mb"].append(0)
                usage_data["volume_3g_mb"].append(0)
                usage_data["volume_4g_mb"].append(0)
                usage_data["volume_5g_mb"].append(0)
                usage_data["total_data_mb"].append(0)
        
        return usage_data
        
    except Exception as e:
        print(f"Error getting usage data for MSISDN {msisdn}: {e}")
        # Return empty data structure
        for key in usage_data.keys():
            if key != "months":
                usage_data[key] = [0, 0, 0]
        return usage_data

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
        
        # Get common location data based on LAC and CELL from VLRD
        common_locations = []
        if not ref_location_df.empty:
            vlrd_lac = str(vlrd_row.get('LAC', '')).strip()
            vlrd_cell = str(vlrd_row.get('CELL', '')).strip()
            
            # Primary search: Find exact matches by LAC and CELL
            if vlrd_lac and vlrd_cell and vlrd_lac != 'Unknown' and vlrd_cell != 'Unknown' and vlrd_lac != '' and vlrd_cell != '':
                # Convert LAC and CELL to appropriate types for comparison
                try:
                    vlrd_lac_int = int(vlrd_lac)
                    vlrd_cell_int = int(vlrd_cell)
                    
                    print(f"Searching for LAC={vlrd_lac_int}, CELL={vlrd_cell_int}")
                    
                    # Filter out rows with NaN/infinite values first, then find matches
                    valid_ref_data = ref_location_df.dropna(subset=['lac', 'cellid'])
                    lac_cell_matches = valid_ref_data[
                        (valid_ref_data['lac'].astype(int) == vlrd_lac_int) &
                        (valid_ref_data['cellid'].astype(int) == vlrd_cell_int)
                    ]
                    
                    print(f"Found {len(lac_cell_matches)} matches using integer comparison")
                    
                except (ValueError, TypeError) as e:
                    print(f"Integer conversion failed: {e}, trying string comparison")
                    # If conversion fails, try string comparison
                    lac_cell_matches = ref_location_df[
                        (ref_location_df['lac'].astype(str).str.strip() == vlrd_lac) &
                        (ref_location_df['cellid'].astype(str).str.strip() == vlrd_cell)
                    ]
                    print(f"Found {len(lac_cell_matches)} matches using string comparison")
                
                
                for _, row in lac_cell_matches.iterrows():
                    # Safe conversion for LAC and CELL
                    lac_val = row.get('lac')
                    cell_val = row.get('cellid')
                    
                    # Helper function to safely handle NaN values
                    def safe_value(val, default='Unknown'):
                        if pd.isna(val) or val is None:
                            return default
                        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
                            return default
                        return val
                    
                    location_data = {
                        'CELL_CODE': safe_value(row.get('cellcode')),
                        'SITE_NAME': safe_value(row.get('sitename')), 
                        'SITE_NAME_LONG': safe_value(row.get('site_name_long')),
                        'LAC': int(lac_val) if pd.notna(lac_val) and lac_val is not None else 'Unknown',
                        'CELL': int(cell_val) if pd.notna(cell_val) and cell_val is not None else 'Unknown',
                        'LAT': str(safe_value(row.get('lat'), 'Not Found')),
                        'LON': str(safe_value(row.get('lon'), 'Not Found')),
                        'DISTRICT': safe_value(row.get('district')),
                        'REGION': safe_value(row.get('region')),
                        'PROVINCE': safe_value(row.get('province')),
                        'TYPE': safe_value(row.get('type')),
                        'STATUS': safe_value(row.get('status')),
                        'BTS_CONFIGURED_DATE': safe_value(row.get('bts_configured_date')),
                        'ONAIR_DATE': safe_value(row.get('onair_date')),
                        'BORE': str(safe_value(row.get('bore')))
                    }
                    common_locations.append(location_data)
        
        # Get usage data for the MSISDN
        usage_data = get_usage_data(msisdn)
        
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
            "Usage_Data": usage_data,
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