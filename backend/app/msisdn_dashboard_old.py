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

# API endpoint for dashboard data
from fastapi.responses import JSONResponse
@app.get("/api/msisdn_dashboard")
async def api_msisdn_dashboard(msisdn: str = ""):
    global latest_result
    
    try:
        # If MSISDN is provided, get data for that MSISDN
        if msisdn:
            result = get_msisdn_data(msisdn, ref_df, tac_df, usage_df, SIM_TYPE_MAPPING)
            if isinstance(result, dict) and "error" in result:
                return JSONResponse({"error": result["error"]}, status_code=404)
        
        table = []
        if latest_result:
            # Calculate totals for voice and SMS
            voice_total = 0
            sms_total = 0
            
            if latest_result.get("Monthly Usage"):
                monthly = latest_result["Monthly Usage"]
                
                # Safely handle voice data
                if "incoming_voice" in monthly and "outgoing_voice" in monthly:
                    try:
                        incoming = sum(monthly["incoming_voice"] or [0])
                        outgoing = sum(monthly["outgoing_voice"] or [0])
                        voice_total = f"{(incoming + outgoing):.1f}"
                    except (TypeError, ValueError):
                        voice_total = "0.0"
                
                # Safely handle SMS data
                if "incoming_sms" in monthly and "outgoing_sms" in monthly:
                    try:
                        sms_total = sum(monthly["incoming_sms"] or [0]) + sum(monthly["outgoing_sms"] or [0])
                    except (TypeError, ValueError):
                        sms_total = 0
                
                # Safely handle data usage
                try:
                    data_usage = sum(latest_result.get("Monthly Usage", {}).get("Total", []) or [0])
                except (TypeError, ValueError):
                    data_usage = 0
            else:
                data_usage = 0
            
            table.append({
                "msisdn": latest_result.get("MSISDN", "-"),
                "device_type": latest_result.get("Device Type", "-"),
                "location": f"{latest_result.get('Region', '-')}, {latest_result.get('District', '-')}",
                "data_usage": data_usage,
                "voice_usage": voice_total,
                "sms_count": sms_total
            })
            
        dash_url = "/dashboard/usage-graph/"  # Dash is mounted here
        
        # Add device details for extended information
        device_details = {}
        cell_locations = []
        common_cell_locations = []
        rsrp_data = []
        lte_util_data = []
        
        if latest_result:
            for key in ["Brand", "Model", "OS", "Marketing Name", "Technology", "VoLTE"]:
                if key in latest_result:
                    device_details[key] = latest_result[key]
                    
            # Add cell location data if available
            if "Cell Locations" in latest_result:
                cell_locations = latest_result["Cell Locations"]
                
            # Add common cell locations if available
            if "Common Cell Locations" in latest_result:
                common_cell_locations = latest_result["Common Cell Locations"]
            
            # Add RSRP and LTE utilization data if available
            if "RSRP Data" in latest_result:
                rsrp_data = latest_result["RSRP Data"]
                
            if "LTE Utilization Data" in latest_result:
                lte_util_data = latest_result["LTE Utilization Data"]
        
        return JSONResponse({
            "table": table, 
            "dash_url": dash_url,
            "device_details": device_details,
            "cell_locations": cell_locations,
            "common_cell_locations": common_cell_locations,
            "rsrp_data": rsrp_data,
            "lte_util_data": lte_util_data
        })
    except Exception as e:
        import traceback
        error_trace = traceback.format_exc()
        print(f"API Error: {str(e)}\n{error_trace}")
        return JSONResponse({"error": f"Internal server error: {str(e)}"}, status_code=500)

# Load reference data
ref_df = pd.read_csv(REFERENCE_FILE)
# Convert numeric columns to strings for consistent comparison
for col in ['lac', 'cellid']:
    if col in ref_df.columns:
        ref_df[col] = ref_df[col].astype(str)
tac_df = pd.read_csv(TAC_FILE, low_memory=False)

def load_usage_data():
    df_list = []
    for month, file in USAGE_FILES.items():
        df = pd.read_csv(file, sep="\t")
        df["Month"] = month
        df_list.append(df)
    return pd.concat(df_list, ignore_index=True)

usage_df = load_usage_data()

SIM_TYPE_MAPPING = {
    '1': ("ESIM", "PRE"),
    '2': ("USIM", "PRE"),
    '3': ("SIM", "PRE"),
    '7': ("ESIM", "POS"),
    '8': ("USIM", "POS"),
    '9': ("SIM", "POS")
}

latest_result = {}

def get_msisdn_data(msisdn, ref_df, tac_df, usage_df, sim_type_mapping):
    global latest_result
    with open(INPUT_FILE, "r") as file:
        lines = file.readlines()
    for line in lines:
        columns = line.strip().split(";")
        if len(columns) < 5:
            continue
        imsi = columns[0]
        msisdn_entry = columns[1]
        tac = columns[2][:8]
        location = columns[4]
        if msisdn_entry == msisdn:
            sitename = cellcode = lon = lat = region = district = "Not Found"
            sim_type = connection_type = "Unknown"
            if len(imsi) >= 8:
                imsi_digit = imsi[7]
                if imsi_digit in sim_type_mapping:
                    sim_type, connection_type = sim_type_mapping[imsi_digit]
            lac_dec = sac_dec = "Not Found"
            cell_locations = []
            common_cell_locations = []
            if location.strip():
                match = re.match(r"(\d+)-(\w+)-([a-fA-F0-9]+)", location)
                if match:
                    try:
                        lac_dec = int(match.group(2), 16)
                        sac_dec = int(match.group(3), 16)
                        # First, find the exact match for the current cell
                        matched_row = ref_df[(ref_df['lac'] == str(lac_dec)) & (ref_df['cellid'] == str(sac_dec))]
                        if not matched_row.empty:
                            row = matched_row.iloc[0]
                            sitename = row['sitename']
                            cellcode = row['cellcode']
                            lon = row['lon']
                            lat = row['lat']
                            region = row['region']
                            district = row['district']
                            
                        # Get all cell locations in the same LAC
                        lac_cells = ref_df[ref_df['lac'] == str(lac_dec)]
                        if not lac_cells.empty:
                            for idx, row in lac_cells.iterrows():
                                cell_info = {
                                    'key': row.get('key', ''),
                                    'cgi': row.get('cgi', ''),
                                    'lac': row.get('lac', ''),
                                    'cellid': row.get('cellid', ''),
                                    'node': row.get('node', ''),
                                    'sitename': row.get('sitename', ''),
                                    'site_name_long': row.get('site_name_long', ''),
                                    'cellcode': row.get('cellcode', ''),
                                    'lon': row.get('lon', ''),
                                    'lat': row.get('lat', ''),
                                    'bore': row.get('bore', ''),
                                    'type': row.get('type', ''),
                                    'region': row.get('region', ''),
                                    'district': row.get('district', ''),
                                    'province': row.get('province', ''),
                                    'status': row.get('status', '')
                                }
                                cell_locations.append(cell_info)
                                
                                # Prepare common cell locations for the template
                                if row['cellcode'] != cellcode:  # Skip the primary cell
                                    common_loc = {
                                        'CELL_CODE': row.get('cellcode', ''),
                                        'SITE_NAME': row.get('sitename', ''),
                                        'LAT': row.get('lat', ''),
                                        'LON': row.get('lon', ''),
                                        'DISTRICT': row.get('district', ''),
                                        'REGION': row.get('region', ''),
                                        'LAC': row.get('lac', ''),
                                        'CELL': row.get('cellid', ''),
                                        'TYPE': row.get('type', ''),
                                        'PROVINCE': row.get('province', '')
                                    }
                                    common_cell_locations.append(common_loc)
                                    
                                    # Add placeholder for RSRP and LTE data that could be populated later
                                    common_loc['RSRP_DATA'] = []
                                    common_loc['LTE_UTIL_DATA'] = []
                    except ValueError:
                        return {"error": "Invalid hex values for LAC or SAC"}
            brand = model = software_os_name = marketing_name = year_released = device_type = volte = technology = primary_hardware_type = "Not Found"
            if tac.isdigit():
                tac_row = tac_df[tac_df['tac'] == int(tac)]
                if not tac_row.empty:
                    row = tac_row.iloc[0]
                    brand = row['brand']
                    model = row['model']
                    software_os_name = row['software_os_name']
                    marketing_name = row['marketing_name']
                    year_released = row['year_released']
                    device_type = row['device_type']
                    volte = row['volte']
                    technology = row['technology']
                    primary_hardware_type = row['primary_hardware_type']
            # Ensure msisdn is convertible to int and usage values are safe
            try:
                msisdn_int = int(float(msisdn))
            except Exception:
                msisdn_int = None
            usage_records = usage_df[usage_df["MSISDN"] == msisdn_int] if msisdn_int is not None else pd.DataFrame()
            monthly_usage = {
                "months": [],
                "2G": [],
                "3G": [],
                "4G": [],
                "5G": [],
                "Total": []
            }
            def safe_int(val):
                try:
                    if hasattr(val, 'real'):
                        return int(val.real)
                    return int(val)
                except Exception:
                    return 0
            if not usage_records.empty:
                # If the file only has one month, just use that
                months = usage_records["Month"].unique().tolist() if "Month" in usage_records.columns else ["May"]
                grouped = usage_records.groupby("Month").sum(numeric_only=True)
                for month in months:
                    monthly_usage["months"].append(month)
                    monthly_usage["2G"].append(safe_int(grouped.at[month, 'volume_2g_mb']) if month in grouped.index and 'volume_2g_mb' in grouped.columns else 0)
                    monthly_usage["3G"].append(safe_int(grouped.at[month, 'volume_3g_mb']) if month in grouped.index and 'volume_3g_mb' in grouped.columns else 0)
                    monthly_usage["4G"].append(safe_int(grouped.at[month, 'volume_4g_mb']) if month in grouped.index and 'volume_4g_mb' in grouped.columns else 0)
                    monthly_usage["5G"].append(safe_int(grouped.at[month, 'volume_5g_mb']) if month in grouped.index and 'volume_5g_mb' in grouped.columns else 0)
                    total = 0
                    if month in grouped.index:
                        total = (
                            safe_int(grouped.at[month, 'volume_2g_mb']) if 'volume_2g_mb' in grouped.columns else 0
                            + safe_int(grouped.at[month, 'volume_3g_mb']) if 'volume_3g_mb' in grouped.columns else 0
                            + safe_int(grouped.at[month, 'volume_4g_mb']) if 'volume_4g_mb' in grouped.columns else 0
                            + safe_int(grouped.at[month, 'volume_5g_mb']) if 'volume_5g_mb' in grouped.columns else 0
                        )
                    monthly_usage["Total"].append(total)
            # Add IMEI for the template
            imei = "Unknown"  # Default value
            
            result = {
                "MSISDN": msisdn,
                "IMSI": imsi,
                "IMEI": imei,
                "SIM Type": sim_type,
                "Connection Type": connection_type,
                "LAC": lac_dec,
                "SAC": sac_dec,
                "Sitename": sitename,
                "Cellcode": cellcode,
                "Lon": lon,
                "Lat": lat,
                "Region": region,
                "District": district,
                "TAC": tac,
                "Brand": brand,
                "Model": model,
                "OS": software_os_name,
                "Marketing Name": marketing_name,
                "Year Released": year_released,
                "Device Type": device_type,
                "VoLTE": volte,
                "Technology": technology,
                "Primary Hardware Type": primary_hardware_type,
                "Monthly Usage": monthly_usage,
                "Cell Locations": cell_locations,
                "Common Cell Locations": common_cell_locations,
                "RSRP Data": [],
                "LTE Utilization Data": []
            }
            latest_result = result
            return result
    return {"error": "MSISDN not found"}


@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request, "result": None})

@app.post("/search", response_class=HTMLResponse)
async def search(request: Request, msisdn: str = Form(...)):
    result = get_msisdn_data(msisdn, ref_df, tac_df, usage_df, SIM_TYPE_MAPPING)
    if "error" in result:
        return templates.TemplateResponse("index.html", {"request": request, "error": result["error"]})
    
    # Add has_map flag for the template
    has_map = bool(result.get("Lat") and result.get("Lon") and result.get("Lat") != "Not Found" and result.get("Lon") != "Not Found")
    
    return templates.TemplateResponse("index.html", {"request": request, "result": result, "has_map": has_map})

@app.get("/msisdn_details", response_class=HTMLResponse)
async def msisdn_details_search(request: Request, msisdn: Optional[str] = None):
    # Check if msisdn is provided as a query parameter
    if msisdn:
        result = get_msisdn_data(msisdn, ref_df, tac_df, usage_df, SIM_TYPE_MAPPING)
        if "error" in result:
            return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
        # Add has_map flag for the template
        has_map = bool(result.get("Lat") and result.get("Lon") and result.get("Lat") != "Not Found" and result.get("Lon") != "Not Found")
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "has_map": has_map, "msisdn": msisdn})
    else:
        # No MSISDN provided, render template with search form only
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "msisdn": None})

@app.get("/msisdn_details/{msisdn}", response_class=HTMLResponse)
async def msisdn_details(request: Request, msisdn: str):
    result = get_msisdn_data(msisdn, ref_df, tac_df, usage_df, SIM_TYPE_MAPPING)
    if "error" in result:
        return templates.TemplateResponse("msisdn_details.html", {"request": request, "error": result["error"], "msisdn": msisdn})
    
    # Add has_map flag for the template
    has_map = bool(result.get("Lat") and result.get("Lon") and result.get("Lat") != "Not Found" and result.get("Lon") != "Not Found")
    
    return templates.TemplateResponse("msisdn_details.html", {"request": request, "result": result, "has_map": has_map, "msisdn": msisdn})

flask_server = Flask(__name__)
dash_app = Dash(__name__, server=flask_server, url_base_pathname="/usage-graph/")

def get_usage_figure():
    if not latest_result.get("Monthly Usage"):
        return go.Figure()
    usage = latest_result["Monthly Usage"]
    fig = go.Figure()
    
    # Data usage by technology
    for tech in ["2G", "3G", "4G", "5G"]:
        if tech in usage:
            fig.add_trace(go.Scatter(x=usage["months"], y=usage[tech],
                                    mode='lines+markers', name=tech))
    
    # Add voice usage on secondary y-axis if available
    has_voice = "incoming_voice" in usage and "outgoing_voice" in usage
    if has_voice:
        incoming_voice = usage["incoming_voice"]
        outgoing_voice = usage["outgoing_voice"]
        
        # Create secondary Y axis for voice
        fig.add_trace(go.Scatter(
            x=usage["months"], 
            y=incoming_voice,
            mode='lines+markers', 
            name='Incoming Voice',
            line=dict(dash='dot', color='darkgreen'),
            yaxis='y2'
        ))
        
        fig.add_trace(go.Scatter(
            x=usage["months"], 
            y=outgoing_voice,
            mode='lines+markers', 
            name='Outgoing Voice',
            line=dict(dash='dot', color='darkred'),
            yaxis='y2'
        ))
        
        # Update layout for dual axis
        fig.update_layout(
            yaxis2=dict(
                title="Voice Usage (min)",
                overlaying="y",
                side="right"
            )
        )
    
    fig.update_layout(
        title=f"Monthly Usage for {latest_result.get('MSISDN', 'Unknown')}",
        xaxis_title="Month",
        yaxis_title="Data Usage (MB)",
        template="plotly_white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
    )
    
    return fig

def get_total_usage_figure():
    if not latest_result.get("Monthly Usage"):
        return go.Figure()
    
    try:
        usage = latest_result["Monthly Usage"]
        fig = go.Figure()
        
        # Check if Total data exists
        if "Total" in usage and usage["Total"] and len(usage["months"]) == len(usage["Total"]):
            fig.add_trace(go.Bar(x=usage["months"], y=usage["Total"], name="Total Usage"))
        else:
            # Create empty figure with message
            fig = go.Figure()
            fig.add_annotation(
                text="No usage data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False,
                font=dict(size=16)
            )
        
        fig.update_layout(title="Total Monthly Usage",
                        xaxis_title="Month", yaxis_title="Total Usage (MB)",
                        template="plotly_white")
        return fig
    except Exception as e:
        # Return empty figure on error
        fig = go.Figure()
        fig.add_annotation(
            text=f"Error loading graph: {str(e)}",
            xref="paper", yref="paper",
            x=0.5, y=0.5, showarrow=False,
            font=dict(size=14, color="red")
        )
        return fig

def create_dash_layout():
    try:
        return html.Div([
            html.H2("Line Graph - Monthly Usage"),
            dcc.Graph(id='usage-graph', figure=get_usage_figure()),
            html.H2("Bar Graph - Total Monthly Usage"),
            dcc.Graph(id='total-usage-graph', figure=get_total_usage_figure()),
            html.Div("Search an MSISDN to update charts", style={'margin-top': '20px', 'text-align': 'center'})
        ])
    except Exception as e:
        return html.Div([
            html.H3("Error Loading Dashboard", style={'color': 'red'}),
            html.P(f"Error: {str(e)}"),
            html.P("Please try searching for a valid MSISDN")
        ])

dash_app.layout = create_dash_layout()

app.mount("/usage-graph", WSGIMiddleware(flask_server))

# You need a Jinja2 template named 'index.html' in your templates folder.
# It should have a form for MSISDN and display the result.

# To run:
# uvicorn msisdn_dashboard_fastapi:app --host 127.0.0.1 --port 8000
