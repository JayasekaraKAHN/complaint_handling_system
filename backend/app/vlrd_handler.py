import pandas as pd
import os
from typing import List, Dict, Any, Optional

class VLRDHandler:
    def __init__(self, data_path: str = "data/VLRD_Sample.csv"):
        self.data_path = data_path
        self.df = None
        self._load_data()
    
    def _load_data(self):
        """Load VLRD data from CSV file"""
        try:
            if os.path.exists(self.data_path):
                print(f"Loading VLRD data from {self.data_path}...")
                # Read CSV with tab separator as VLRD files are often tab-separated
                self.df = pd.read_csv(self.data_path, sep='\t', low_memory=False)
                print(f"Loaded {len(self.df)} records from VLRD data")
                print(f"Columns: {list(self.df.columns)}")
            else:
                print(f"VLRD data file not found at {self.data_path}")
                self.df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading VLRD data: {str(e)}")
            self.df = pd.DataFrame()
    
    def search_msisdn(self, msisdn: str) -> List[Dict[str, Any]]:
        """Search for MSISDN in VLRD data"""
        if self.df is None or self.df.empty:
            return []
        
        try:
            # Clean and normalize MSISDN
            search_msisdn = str(msisdn).strip()
            
            # Search in all columns that might contain MSISDN
            possible_msisdn_columns = []
            for col in self.df.columns:
                col_lower = col.lower()
                if any(keyword in col_lower for keyword in ['msisdn', 'number', 'mobile', 'phone']):
                    possible_msisdn_columns.append(col)
            
            # If no specific MSISDN columns found, search in all columns
            if not possible_msisdn_columns:
                possible_msisdn_columns = list(self.df.columns)
            
            # Search for exact matches
            mask = pd.Series([False] * len(self.df))
            for col in possible_msisdn_columns:
                if col in self.df.columns:
                    mask |= self.df[col].astype(str).str.contains(search_msisdn, na=False, case=False)
            
            results = self.df[mask]
            
            # Convert to list of dictionaries
            if not results.empty:
                # Ensure all keys are strings for type compatibility
                return [{str(k): v for k, v in record.items()} for record in results.to_dict('records')]
            else:
                return []
                
        except Exception as e:
            print(f"Error searching MSISDN {msisdn}: {str(e)}")
            return []
    
    def get_data_info(self) -> Dict[str, Any]:
        """Get information about the loaded data"""
        if self.df is None or self.df.empty:
            return {"status": "No data loaded", "records": 0, "columns": []}
        
        return {
            "status": "Data loaded successfully",
            "records": len(self.df),
            "columns": list(self.df.columns),
            "sample_data": self.df.head(3).to_dict('records') if not self.df.empty else []
        }

# Global instance
vlrd_handler = None

def get_vlrd_handler():
    """Get or create VLRD handler instance"""
    global vlrd_handler
    if vlrd_handler is None:
        vlrd_handler = VLRDHandler()
    return vlrd_handler
