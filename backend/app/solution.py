from .ollama_client import query_ollama
from .prompts import SOLUTION_PROMPT
from .features import get_customer_context, retrieve_similar_solutions
from .classifier import classify_complaint
import os
import pandas as pd

def generate_solution(
    msisdn: str = "",
    complaint_text: str | None = None,
    location: str | None = None,
    site_alarm: str | None = None,
    kpi: str | None = None,
    billing: str | None = None
) -> str:
    """
    Generates a personalized telecom complaint solution.
    """
    context = get_customer_context(msisdn)

    # Use latest historical complaint if no current complaint_text
    if not complaint_text:
        if context.get("complaints"):
            complaint_text = context["complaints"][-1].get("Complaint", "")
        else:
            complaint_text = ""

    complaint_type = classify_complaint(complaint_text or "")

    # Retrieve similar solutions from KB
    similar = retrieve_similar_solutions(complaint_text or "", k=3)
    similar_formatted = "\n- ".join(similar) if similar else "None available"

    # Extract fields for prompt from context
    usage_info = context.get("usage", [{}])[0] if context.get("usage") else {}
    complaint_info = context.get("complaints", [{}])[0] if context.get("complaints") else {}

    prompt = SOLUTION_PROMPT.format(
        msisdn=msisdn or "Unknown",
        received_date=complaint_info.get("Received Date") or "Unknown",
        lon=usage_info.get("Lon") or "Unknown",
        lat=usage_info.get("Lat") or "Unknown",
        issue_description=complaint_text or "Unknown",
        device_info=usage_info.get("Device Info") or "Unknown",
        signal_strength=usage_info.get("Signal Strength") or "Unknown",
        signal_quality=usage_info.get("Signal Quality") or "Unknown",
        complaint_type=complaint_type,
        site_kpi=site_alarm or kpi or usage_info.get("Site KPI/Alarm") or "Unknown",
        previous_complaints="\n".join([str(c.get("Complaint", "")) for c in context.get("complaints", [])]) or "None",
        similar_solutions=similar_formatted
    )

    # Optional: append additional fields like location or billing
    extra_info = []
    if location:
        extra_info.append(f"Location info: {location}")
    if billing:
        extra_info.append(f"Billing info: {billing}")
    if extra_info:
        prompt += "\n\nAdditional Info:\n" + "\n".join(extra_info)

    # Check Complains_Soulutions.csv for direct match
    complaints_path = os.path.join(os.path.dirname(__file__), '..', 'data', 'Complains_Soulutions.csv')
    try:
        try:
            dfc = pd.read_csv(complaints_path, encoding='utf-8')
        except UnicodeDecodeError:
            dfc = pd.read_csv(complaints_path, encoding='latin1')

        if complaint_text:
            for idx, row in dfc.iterrows():
                comp = row.get('Issue Description', row.get('Complaint', '')).strip()
                sol = row.get('Solution', '').strip()
                if comp.lower() == complaint_text.strip().lower() and sol:
                    return f"Solution: {sol}"

    except Exception as e:
        print(f"Error reading Complains_Soulutions.csv: {e}")
        return f"Error reading solutions file: {e}"

    # If no exact match, use AI
    ai_solution = query_ollama(prompt)
    if isinstance(ai_solution, str) and ai_solution.lower().startswith("error:"):
        if similar:
            return "\n".join(["Suggested steps (KB-based):"] + [f"- {s}" for s in similar])
        return "AI model unavailable. Follow standard troubleshooting: reboot device, check APN, verify signal, escalate if persists."

    return ai_solution
