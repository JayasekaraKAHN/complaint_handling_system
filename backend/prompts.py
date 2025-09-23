"""
Simple prompts for Ollama Llama 3.2 1B LLM integration
"""

def create_complaint_solution_prompt(complaint_details: dict, similar_cases: list | None = None, location_info: dict | None = None):
    """
    Create a comprehensive prompt for generating complaint solutions
    
    Args:
        complaint_details: Dict with complaint information
        similar_cases: List of similar complaint cases from database
        location_info: Dict with location-based network information
    """
    
    prompt = f"""You are a Sri Lankan telecom expert AI assistant. Generate a technical solution for the following customer complaint:

COMPLAINT DETAILS:
- MSISDN: {complaint_details.get('msisdn', 'N/A')}
- Issue Description: {complaint_details.get('complaint', 'N/A')}
- Device/Settings/VPN/APN: {complaint_details.get('device_type_settings_vpn_apn', 'N/A')}
- Signal Strength: {complaint_details.get('signal_strength', 'N/A')}
- Quality of Signal: {complaint_details.get('quality_of_signal', 'N/A')}
- Site KPI/Alarm: {complaint_details.get('site_kpi_alarm', 'N/A')}
- Past Data Analysis: {complaint_details.get('past_data_analysis', 'N/A')}
- Indoor/Outdoor Coverage: {complaint_details.get('indoor_outdoor_coverage_issue', 'N/A')}
- Location: {complaint_details.get('location', 'N/A')}
- Longitude: {complaint_details.get('longitude', 'N/A')}
- Latitude: {complaint_details.get('latitude', 'N/A')}

"""

    if similar_cases:
        prompt += "\nSIMILAR PAST CASES:\n"
        for i, case in enumerate(similar_cases[:3], 1):  # Limit to top 3 similar cases
            prompt += f"{i}. Issue: {case.get('Issue Description', 'N/A')}\n"
            prompt += f"   Solution: {case.get('Solution', 'N/A')}\n"
            prompt += f"   Signal: {case.get('Signal Strength', 'N/A')}\n\n"
    
    if location_info:
        prompt += f"\nLOCATION NETWORK DATA:\n"
        prompt += f"- Site Name: {location_info.get('site_name', 'N/A')}\n"
        prompt += f"- RSRP Range 1 (>-105dBm): {location_info.get('rsrp_range_1', 'N/A')}%\n"
        prompt += f"- RSRP Range 2 (-105~-110dBm): {location_info.get('rsrp_range_2', 'N/A')}%\n"
        prompt += f"- RSRP Range 3 (-110~-115dBm): {location_info.get('rsrp_range_3', 'N/A')}%\n"
        prompt += f"- RSRP < -115dBm: {location_info.get('rsrp_weak', 'N/A')}%\n"
    
    prompt += """
SOLUTION REQUIREMENTS:
1. Provide a specific, actionable technical solution
2. Consider signal strength and coverage issues
3. Include any necessary site interventions or customer actions
4. Keep the solution brief and concise (maximum 5 sentences)
5. Focus on practical steps to resolve the issue
6. Use simple, clear language that customers can understand

Generate only the solution text without additional formatting or explanations:"""

    return prompt

def create_pattern_analysis_prompt(complaint_text: str, historical_data: list):
    """
    Create prompt for analyzing patterns in similar complaints
    """
    
    prompt = f"""Analyze the following complaint patterns to generate an appropriate solution:

CURRENT COMPLAINT: {complaint_text}

HISTORICAL PATTERNS:
"""
    
    for i, case in enumerate(historical_data[:5], 1):
        prompt += f"{i}. Issue: {case.get('Issue Description', 'N/A')}\n"
        prompt += f"   Solution: {case.get('Solution', 'N/A')}\n"
        prompt += f"   Conditions: Signal={case.get('Signal Strength', 'N/A')}, KPI={case.get('Site KPI/Alarm', 'N/A')}\n\n"
    
    prompt += """
Based on these patterns, identify:
1. Common root causes
2. Effective solution approaches
3. Any environmental factors (signal, location, device)

Generate a brief, tailored solution (maximum 5 sentences) considering the patterns above:"""
    
    return prompt

def create_new_complaint_prompt(complaint_details: dict, location_context: dict | None = None):
    """
    Create prompt for completely new complaints not in dataset
    """
    
    prompt = f"""You are analyzing a new type of complaint for Sri Lankan telecom network. Generate a technical solution:

NEW COMPLAINT:
- Issue: {complaint_details.get('complaint', 'N/A')}
- MSISDN: {complaint_details.get('msisdn', 'N/A')}
- Device/Settings/VPN/APN: {complaint_details.get('device_type_settings_vpn_apn', 'N/A')}
- Signal Strength: {complaint_details.get('signal_strength', 'N/A')}
- Quality of Signal: {complaint_details.get('quality_of_signal', 'N/A')}
- Site KPI/Alarm: {complaint_details.get('site_kpi_alarm', 'N/A')}
- Past Data Analysis: {complaint_details.get('past_data_analysis', 'N/A')}
- Coverage Type: {complaint_details.get('indoor_outdoor_coverage_issue', 'N/A')}
- Location: {complaint_details.get('location', 'N/A')}
- Longitude: {complaint_details.get('longitude', 'N/A')}
- Latitude: {complaint_details.get('latitude', 'N/A')}
"""

    if location_context:
        prompt += f"\nLOCATION ANALYSIS:\n"
        prompt += f"- Network Coverage: {location_context.get('coverage_quality', 'Unknown')}\n"
        prompt += f"- Signal Quality Distribution: {location_context.get('signal_distribution', 'Unknown')}\n"
        prompt += f"- Nearby Sites: {location_context.get('nearby_sites', 'Unknown')}\n"

    prompt += """
SOLUTION APPROACH:
1. Analyze the technical symptoms
2. Consider common telecom issues (signal, interference, device, network)
3. Provide specific troubleshooting steps
4. Keep the solution brief and actionable (maximum 5 sentences)
5. Use clear, customer-friendly language

Generate a practical solution (maximum 5 sentences):"""
    
    return prompt