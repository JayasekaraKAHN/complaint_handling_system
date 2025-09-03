#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced solution generator with meaningful related case analysis
"""

import sys
import os

# Add the current directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from app.enhanced_solution import generate_solution

def test_enhanced_solution():
    print("🧪 TESTING ENHANCED SOLUTION GENERATOR")
    print("=" * 60)
    
    # Test case 1: Exact match from datafinal.csv
    print("\n📋 TEST CASE 1: Voice Call Issue (Known Pattern)")
    print("-" * 50)
    
    solution1 = generate_solution(
        msisdn="703009908",
        complaint_text="Sudden voice call issue for all devices",
        device_type_settings_vpn_apn="Mobile devices",
        signal_strength="Serving cell - KLPOR5G RSRP -87 to -94 dBm, RSRQ -7 to -10 dB",
        site_kpi_alarm="Abnormal KPIs for KLPOR5",
        past_data_analysis="Past data; serving site KLPOR5",
        location="6.721305732\t80.12328967"
    )
    
    print(solution1)
    
    # Test case 2: Coverage issue
    print("\n\n📋 TEST CASE 2: Coverage Drop Issue")
    print("-" * 50)
    
    solution2 = generate_solution(
        msisdn="711580161", 
        complaint_text="Sudden coverage drop for all devices",
        device_type_settings_vpn_apn="Huawei Router",
        signal_strength="No coverage",
        site_kpi_alarm="Cell unavailability KLPET1",
        past_data_analysis="Past data ; Serving site for cx no data KLPET1",
        location="6.549388625\t80.10738611"
    )
    
    print(solution2)
    
    # Test case 3: Generic issue to test pattern matching
    print("\n\n📋 TEST CASE 3: Generic Network Issue")
    print("-" * 50)
    
    solution3 = generate_solution(
        msisdn="999888777",
        complaint_text="Network connectivity problems with voice calls",
        device_type_settings_vpn_apn="Mobile device",
        signal_strength="Weak signal",
        site_kpi_alarm="Site alarms present",
        location="6.7\t80.1"
    )
    
    print(solution3)

if __name__ == "__main__":
    test_enhanced_solution()
