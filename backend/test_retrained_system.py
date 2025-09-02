#!/usr/bin/env python3
"""
Test script to verify the retrained model integration
"""

from app.solution import generate_solution
from app.classifier import classify_complaint_with_retrained_model, get_solution_from_patterns
import json

def test_retrained_system():
    """Test the retrained complaint handling system"""
    print("🧪 Testing Retrained Complaint Handling System")
    print("=" * 60)
    
    test_cases = [
        {
            "name": "Voice Call Issue - Exact Match",
            "complaint": "Sudden voice call issue for all devices",
            "expected_solution": "There were cell unavailabilities in the site, Clear the alarms solved the issue"
        },
        {
            "name": "KLPET1 Site Issue",
            "complaint": "Coverage drop issue",
            "site_alarm": "Cell unavailability KLPET1",
            "expected_solution": "KLPET1 Site on aired and solved"
        },
        {
            "name": "KLPOR5 Site Issue",
            "complaint": "Signal problems",
            "site_alarm": "Abnormal KPIs for KLPOR5",
            "expected_solution": "There were cell unavailabilities in the site, Clear the alarms solved the issue"
        },
        {
            "name": "Data Speed Issue",
            "complaint": "Data speed is slow",
            "location": "Colombo",
            "billing": "regular"
        }
    ]
    
    passed = 0
    total = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\\n🔍 Test {i}: {test_case['name']}")
        print("-" * 40)
        
        try:
            # Generate solution
            kwargs = {
                'msisdn': '947712345678',
                'complaint_text': test_case['complaint']
            }
            
            # Add optional parameters
            for param in ['location', 'site_alarm', 'kpi', 'billing']:
                if param in test_case:
                    kwargs[param] = test_case[param]
            
            result = generate_solution(**kwargs)
            
            # Extract solution (first line before EXPLANATION)
            solution_line = result.split('\\n')[0].strip()
            
            print(f"📝 Complaint: {test_case['complaint']}")
            if 'site_alarm' in test_case:
                print(f"🚨 Site Alarm: {test_case['site_alarm']}")
            print(f"✅ Solution: {solution_line}")
            
            # Check if expected solution matches
            if 'expected_solution' in test_case:
                if test_case['expected_solution'] in solution_line:
                    print("✅ PASS - Expected solution matched")
                    passed += 1
                else:
                    print(f"❌ FAIL - Expected: {test_case['expected_solution']}")
            else:
                print("✅ PASS - Solution generated")
                passed += 1
                
        except Exception as e:
            print(f"❌ ERROR: {e}")
    
    print(f"\\n📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Retrained model is working correctly.")
    else:
        print("⚠️ Some tests failed. Check the implementation.")
    
    return passed == total

def test_pattern_matching():
    """Test the pattern matching functionality"""
    print("\\n🔍 Testing Pattern Matching")
    print("=" * 40)
    
    patterns_test = [
        ("mobile device klpor5", "Mobile Device"),
        ("huawei router klpet1", "Huawei Router"),
        ("s10 router data slow", "S10 Router"),
        ("coverage klpor5 alarm", "KLPOR5 site")
    ]
    
    for text, description in patterns_test:
        solution = get_solution_from_patterns(text)
        print(f"🔍 {description}: {solution if solution else 'No pattern match'}")

if __name__ == "__main__":
    success = test_retrained_system()
    test_pattern_matching()
    
    if success:
        print("\\n🚀 System ready for production deployment!")
    else:
        print("\\n⚠️ Issues detected - review before deployment.")
