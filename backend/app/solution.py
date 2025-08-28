from .ollama_client import query_ollama
from .prompts import SOLUTION_PROMPT
from .features import get_customer_context, retrieve_similar_solutions
from .classifier import classify_complaint
import os
import pandas as pd

def generate_solution(msisdn: str = "", complaint_text: str | None = None) -> str:
    context = get_customer_context(msisdn)

    # Determine complaint text priority: payload > customer's latest historical complaint
    if not complaint_text:
        if context.get("complaints"):
            last = context["complaints"][-1]
            complaint_text = last.get("Complaint") or ""
        else:
            complaint_text = ""

    complaint_type = classify_complaint(complaint_text or "")

    # Retrieve similar solutions from KB using complaint text
    similar = retrieve_similar_solutions(complaint_text or "", k=3)
    similar_formatted = "\n- ".join(similar) if similar else "None available"


    # Extract fields for prompt
    usage_info = context.get("usage", [{}])[0] if context.get("usage") else {}
    complaint_info = context.get("complaints", [{}])[0] if context.get("complaints") else {}

    prompt = SOLUTION_PROMPT.format(
        msisdn=msisdn or "Unknown",
        received_date=complaint_info.get("Received Date") or complaint_info.get("received_date") or "Unknown",
        lon=usage_info.get("Lon") or usage_info.get("lon") or "Unknown",
        lat=usage_info.get("Lat") or usage_info.get("lat") or "Unknown",
        issue_description=complaint_text or complaint_info.get("Complaint") or "Unknown",
        device_info=usage_info.get("Device Info") or usage_info.get("device_info") or "Unknown",
        signal_strength=usage_info.get("Signal Strength") or usage_info.get("signal_strength") or "Unknown",
        signal_quality=usage_info.get("Signal Quality") or usage_info.get("signal_quality") or "Unknown",
        complaint_type=complaint_type,
        site_kpi=usage_info.get("Site KPI/Alarm") or usage_info.get("site_kpi") or "Unknown",
        previous_complaints="\n".join([str(c.get("Complaint", "")) for c in context.get("complaints", [])]) or "None",
        similar_solutions=similar_formatted
    )


    # Check Complains_Soulutions.csv for direct complaint match (ignore MSISDN and fuzzy matching)
    complaints_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Complains_Soulutions.csv')
    try:
        try:
            dfc = pd.read_csv(complaints_path, encoding='utf-8')
        except UnicodeDecodeError:
            dfc = pd.read_csv(complaints_path, encoding='latin1')
        if complaint_text:
            for idx, row in dfc.iterrows():
                comp = row['Issue Description'] if 'Issue Description' in row else row.get('Complaint', '')
                sol = row['Solution'] if 'Solution' in row else ''
                quality = row['Qulity of Signal'] if 'Qulity of Signal' in row else ''
                site_kpi = row['Site KPI/Alarm'] if 'Site KPI/Alarm' in row else ''
                past_data = row['Past Data analysis'] if 'Past Data analysis' in row else ''
                coverage_issue = row['Indoor/Outdoor coverage issue'] if 'Indoor/Outdoor coverage issue' in row else ''
                # Direct match (case-insensitive, strip whitespace)
                if comp and sol and comp.strip().lower() == complaint_text.strip().lower() and sol.strip():
                    details = []
                    if quality: details.append(f"Quality of Signal: {quality}")
                    if site_kpi: details.append(f"Site KPI/Alarm: {site_kpi}")
                    if past_data: details.append(f"Past Data analysis: {past_data}")
                    if coverage_issue: details.append(f"Indoor/Outdoor coverage issue: {coverage_issue}")
                    extra = "\n".join(details)
                    return f"Solution: {sol.strip()}" + (f"\n{extra}" if extra else "")
    except Exception as e:
        error_msg = f"Error reading Complains_Soulutions.csv: {e}"
        print(error_msg)
        return str(error_msg)

    # If no exact match, use AI solution
    ai = query_ollama(prompt)
    if isinstance(ai, str) and ai.lower().startswith("error:"):
        if similar:
            return "\n".join(["Suggested steps (KB-based):"] + [f"- {s}" for s in similar])
        return "We couldn't reach the AI model right now. Please try again, and meanwhile follow standard troubleshooting: reboot device, check APN, verify signal (RSRP/RSRQ), and escalate if persists."
    return ai
