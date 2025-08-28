import os

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
USAGE_DATA = os.path.join(DATA_DIR, "result_dataset.csv")
COMPLAINTS_DATA = os.path.join(DATA_DIR, "Complains_Soulutions.csv")
# TRACKER_DATA = os.path.join(DATA_DIR, "Cluster Level Complain Tracker - June 2025 with Extra columns.xlsx")

OLLAMA_MODEL = "llama3.2:1b"   # Ollama LLaMA model

