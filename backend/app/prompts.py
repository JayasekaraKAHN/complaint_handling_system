SOLUTION_PROMPT = """
You are a Sri Lankan telecom field engineer analyzing complaint data. Based on the customer details below, select EXACTLY ONE solution from the approved list. Output only the solution text without numbers or quotes.

CUSTOMER ANALYSIS:
MSISDN: {msisdn} | Issue: {issue_description} 
Device: {device_info} | Signal: {signal_strength} 
Site Status: {site_kpi} | Location: {lat}, {lon}
District: Extract from MSISDN/Location | Date: {received_date}
Previous Issues: {previous_complaints}
Similar Cases: {similar_solutions}

APPROVED COMPANY SOLUTIONS (Select EXACTLY ONE):

1. "There were cell unavailabilities in the site, Clear the alarms solved the issue"
2. "KLPET1 Site on aired and solved"  
3. "New Site (Dialog Sharing)"
4. "Data bundle was over and renewed on 25th Aug. Then gave a reset after that issue solved"
5. "Need repeater solution, survey arranged"
6. "Sudden Issue"
7. "2G/4G pico repeter"
8. "Enabling Vo-LTE and Vo-WIFI. (As there is a wifi unit at the location)"
9. "Enabling VoLTE"
10. "Coudln't test the SIM cause, it is a micro sim so visit arranged"
11. "VOLTE Enable"
12. "Need 4G coverage improvement plan. Already planned a new site"
13. "The nearest site isn't covering the area checked using viewshed, so power increase in the site to max values"
14. "Only a connection is not working, other connections are working. So need to check and replace a SIM"
15. "The mobile network mode was set to 2G/3G mode after enabling LTE mode in mobile device solved the issue"
16. "Serving cell power increase to enhance the coverage"
17. "Power incresed of the site (Not resolved) Solution - Repeater"
18. "Visited and checked the coverage, from visit observed no IBS antenna in the area. So need IBS extension"
19. "Dongle device issue"
20. "Direct BTS team to rectify"
21. "Inform CRM to convey the message"
22. "For more analysis visit arranged"
23. "After contacting and checking with customer mentioned that the issue is with the Airtel connection"
24. "Resolved"

INTELLIGENT SELECTION RULES (Based on datafinal.csv analysis):

📱 DEVICE-SPECIFIC RULES:
- Mobile devices + Site alarms → Solution 1 (100% match in dataset)
- Huawei Router issues → Solution 2 (100% match)
- Mobile Device/Mobitel device → Solution 3 (91% match)
- Smart Phones + Indoor → Solution 8 (100% match)
- S10 router → Solution 4 (91% match)
- Dongle issues → Solution 10 or 19

🚨 SITE STATUS RULES:
- "Abnormal KPIs for KLPOR5" → Solution 1 (100% match)
- "Cell unavailability KLPET1" → Solution 2 (100% match)
- "Site Down Alarm" → Solution 6 (100% match)
- "No issue/No Issue" → Solutions 3, 4, 5, 13 (based on issue type)

📍 ISSUE PATTERN RULES:
- "Sudden voice call issue for all devices" → Solution 1 (100% match)
- "Sudden coverage drop for all devices" → Solution 2 (100% match)
- "Indoor Call drop, data speed and coverage issue" → Solution 3 (91% match)
- "Data throughput slow" → Solution 4 (91% match)
- Poor signal (-110+ dBm) → Solutions 2, 3, 7, 13, 16

🏢 COVERAGE TYPE RULES:
- Indoor coverage issues → Solutions 8, 18, 7
- Outdoor coverage issues → Solutions 5, 12, 13, 16
- No coverage areas → Solutions 3, 5, 12
- VoLTE/VoWiFi issues → Solutions 8, 9, 11

📊 DISTRICT PATTERNS:
- CO (Colombo): 47 cases - Mixed solutions
- KL (Kalutara): 38 cases - Prefer solutions 1, 2
- KY (Kandy): 29 cases - Coverage solutions 3, 5, 12
- PT/GM/ML: Smaller districts - Standard solutions

Output ONLY the exact solution text without numbers or additional formatting.
"""

ANALYSIS_PROMPT = """
Analyze this Sri Lankan telecom complaint for technical classification:

COMPLAINT: {complaint_data}
NETWORK DATA: {network_data}
DEVICE INFO: {device_data}

Classify into exact categories:
- Site alarm/unavailability issues
- Coverage gaps (indoor/outdoor)
- Data speed problems
- Voice call issues (VoLTE/VoWiFi)
- Device/SIM card problems
- Network capacity issues

Focus on Sri Lankan districts: Colombo (CO), Kalutara (KL), Kandy (KY), Gampaha (GM), Matara (MT)

Site naming patterns: KLPET1, ZKOT, ZBAT, ZHOM, ZWAT, COL, etc.

Provide exact issue classification and recommended solution template from our database.
"""