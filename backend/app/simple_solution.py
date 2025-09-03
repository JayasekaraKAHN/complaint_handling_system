"""
Simplified solution generator using only the new trained model
"""
import pickle
import os
import json

# Load the new trained models
try:
    with open('models/complaint_classifier.pkl', 'rb') as f:
        classifier_model = pickle.load(f)
    with open('models/solution_patterns.pkl', 'rb') as f:
        solution_patterns = pickle.load(f)
    with open('models/model_metadata.json', 'r') as f:
        model_metadata = json.load(f)
    
    MODEL_LOADED = True
    print("✅ New trained models loaded successfully")
    print(f"📊 Model categories: {model_metadata.get('categories', [])}")
    print(f"🔍 Solution patterns: {len(solution_patterns)}")
    
except Exception as e:
    print(f"❌ Error loading models: {e}")
    MODEL_LOADED = False
    classifier_model = None
    solution_patterns = {}
    model_metadata = {}

def generate_simple_solution(msisdn: str, complaint_text: str, 
                           location: str | None = "", site_alarm: str | None = "", 
                           kpi: str | None = "", billing: str | None = "") -> str:
    """
    Generate solution using the new trained model
    """
    if not MODEL_LOADED or classifier_model is None:
        return "System unavailable - models not loaded properly."
    
    try:
        # Prepare input text
        full_complaint = f"{complaint_text} {location or ''} {site_alarm or ''} {kpi or ''} {billing or ''}".strip()
        
        # Get category prediction
        predicted_category = classifier_model.predict([full_complaint])[0]
        confidence_scores = classifier_model.predict_proba([full_complaint])[0]
        confidence = max(confidence_scores)
        
        # Try to find exact solution match
        exact_match = solution_patterns.get(full_complaint.lower().strip())
        if exact_match:
            return f"""
SOLUTION FOR MSISDN: {msisdn}
COMPLAINT CATEGORY: {predicted_category} (Confidence: {confidence:.2f})

RECOMMENDED SOLUTION:
{exact_match}

            """.strip()
        
        # Try keyword-based matching
        complaint_words = full_complaint.lower().split()
        keyword_solutions = []
        
        for pattern_key, solution in solution_patterns.items():
            if pattern_key.startswith('keyword_'):
                keyword = pattern_key.replace('keyword_', '')
                if keyword in complaint_words:
                    keyword_solutions.append(solution)
        
        if keyword_solutions:
            # Use the most common solution among keyword matches
            best_solution = max(set(keyword_solutions), key=keyword_solutions.count)
            return f"""
SOLUTION FOR MSISDN: {msisdn}
COMPLAINT CATEGORY: {predicted_category} (Confidence: {confidence:.2f})

RECOMMENDED SOLUTION:
{best_solution}


            """.strip()
        
        # Generate category-based solution
        category_solutions = {
            'voice_call': "Enable VoLTE on the device. Check network configuration and ensure proper signal strength.",
            'data_internet': "Check APN settings, verify data plan activation, and test signal strength in the area.",
            'coverage_signal': "Investigate nearby cell tower status, check for site alarms, and consider signal boosting solutions.",
            'site_infrastructure': "Contact BTS team to check site status, clear any alarms, and verify equipment functionality.",
            'device_hardware': "Verify device compatibility, check SIM card status, and recommend device troubleshooting steps.",
            'network_config': "Review network settings, update APN configuration, and verify service plan compatibility."
        }
        
        category_solution = category_solutions.get(predicted_category, 
                                                 "Contact technical support for detailed troubleshooting assistance.")
        
        return f"""
SOLUTION FOR MSISDN: {msisdn}
COMPLAINT CATEGORY: {predicted_category} (Confidence: {confidence:.2f})

RECOMMENDED SOLUTION:
{category_solution}

ADDITIONAL CONTEXT:
• MSISDN: {msisdn}
• Location: {location if location else 'Not specified'}
• Issue Type: {predicted_category.replace('_', ' ').title()}

        """.strip()
        
    except Exception as e:
        return f"""
SOLUTION FOR MSISDN: {msisdn}
ERROR: Unable to process complaint - {str(e)}

FALLBACK SOLUTION:
1. Check basic network connectivity
2. Verify device settings
3. Contact customer support for manual assistance

Please try again or contact technical support.
        """.strip()

# Alias for backward compatibility
def generate_solution(msisdn: str, complaint_text: str, location: str | None = "", 
                     site_alarm: str | None = "", kpi: str | None = "", billing: str | None = "") -> str:
    """Main solution generation function"""
    return generate_simple_solution(msisdn, complaint_text, location, site_alarm, kpi, billing)
