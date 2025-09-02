#!/usr/bin/env python3
"""
Test script to demonstrate the enhanced detailed explanation with 8+ points
using matched data from data files
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from app.solution import use_enhanced_model_analysis, generate_enhanced_ai_explanation

def test_detailed_explanation():
    """Test the enhanced explanation with detailed points"""
    
    print("🔍 TESTING ENHANCED DETAILED EXPLANATION")
    print("=" * 60)
    
    # Test case based on actual data from datafinal.csv
    test_complaint = "Sudden voice call issue for all devices"
    test_device = "Mobile devices"
    test_signal = "Serving cell - KLPOR5G RSRP -87 to -94 dBm, RSRQ -7 to -10 dB"
    test_location = "KL"
    test_site_alarm = "Abnormal KPIs for KLPOR5"
    test_kpi = "Past data; serving site KLPOR5"
    test_billing = "Active subscriber"
    
    print(f"📝 TEST COMPLAINT: {test_complaint}")
    print(f"📱 DEVICE: {test_device}")
    print(f"📶 SIGNAL: {test_signal}")
    print(f"📍 LOCATION: {test_location}")
    print(f"🚨 SITE ALARM: {test_site_alarm}")
    print("\n" + "=" * 60)
    
    # Get enhanced analysis
    enhanced_analysis = use_enhanced_model_analysis(
        test_complaint, test_device, test_signal, test_location, test_site_alarm
    )
    
    if enhanced_analysis:
        print(f"Category: {enhanced_analysis['complaint_category']}")
        print(f"Confidence: {enhanced_analysis['confidence']:.1%}")
        print(f"Solution: {enhanced_analysis['recommended_solution']}")
        
        print("\n" + "=" * 60)
        print("📋 DETAILED EXPLANATION (8+ POINTS):")
        print("=" * 60)
        
        # Generate detailed explanation
        detailed_explanation = generate_enhanced_ai_explanation(
            enhanced_analysis, test_complaint, test_device, {},
            test_location, test_site_alarm, test_kpi, test_billing
        )
        
        print(detailed_explanation)
        
        print("\n" + "=" * 60)
        print("✅ TEST COMPLETED SUCCESSFULLY")
    else:
        print("❌ Error: Could not generate enhanced analysis")

if __name__ == "__main__":
    test_detailed_explanation()
