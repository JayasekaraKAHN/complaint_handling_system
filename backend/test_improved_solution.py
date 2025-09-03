"""
Test script to verify the improved solution explanations without redundant confidence and technical analysis
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.enhanced_solution import generate_solution

def test_improved_explanations():
    print("=== TESTING IMPROVED SOLUTION EXPLANATIONS ===\n")
    
    # Test case 1: Known pattern from datafinal.csv
    print("Test 1: Voice call issue (should match exact pattern)")
    print("-" * 60)
    
    result1 = generate_solution(
        msisdn="714640863",
        complaint_text="Sudden voice call issue for all devices",
        device_type_settings_vpn_apn="Mobile devices",
        signal_strength="Serving cell - KLPOR5G RSRP -87 to -94 dBm",
        site_kpi_alarm="Abnormal KPIs for KLPOR5",
        location="6.721305732,80.12328967"
    )
    
    print(result1)
    print("\n" + "="*80 + "\n")
    
    # Test case 2: Coverage issue
    print("Test 2: Coverage drop issue")
    print("-" * 60)
    
    result2 = generate_solution(
        msisdn="711580161",
        complaint_text="Sudden coverage drop for all devices",
        device_type_settings_vpn_apn="Huawei Router",
        signal_strength="No coverage",
        site_kpi_alarm="Cell unavailability KLPET1",
        location="6.549388625,80.10738611"
    )
    
    print(result2)
    print("\n" + "="*80 + "\n")

if __name__ == "__main__":
    test_improved_explanations()
