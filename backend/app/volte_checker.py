"""
VoLTE Provisioning Checker Module

This module provides functionality to check if an MSISDN is provisioned for VoLTE
by searching the VoLTE provisioned base CSV file.
"""

import pandas as pd
import os

class VoLTEChecker:
    def __init__(self):
        """Initialize the VoLTE checker with the CSV file path"""
        self.volte_file_path = "data/VoLTE_Provisioned_Base_20250708.csv"
        self.volte_df = None
        self.load_volte_data()
    
    def load_volte_data(self):
        """Load VoLTE provisioned base data from CSV file"""
        try:
            if os.path.exists(self.volte_file_path):
                # Read the CSV file without headers and assign proper column names
                self.volte_df = pd.read_csv(
                    self.volte_file_path,
                    header=None,  # No headers in the CSV
                    names=['MSISDN', 'CELL_CODE', 'DISTRICT', 'PREPOS', 'SIM_TYPE', 'TAC'],
                    dtype={'MSISDN': str, 'TAC': str, 'CELL_CODE': str},
                    encoding='utf-8-sig'  # Handle BOM character
                )
                
                print(f"Loaded VoLTE data: {len(self.volte_df)} records")
                print(f"Column names: {list(self.volte_df.columns)}")
                
                # Clean the data
                if 'MSISDN' in self.volte_df.columns:
                    self.volte_df['MSISDN'] = self.volte_df['MSISDN'].astype(str).str.strip()
                    print(f"Sample MSISDNs: {self.volte_df['MSISDN'].head(3).tolist()}")
                
                print(f"Successfully loaded VoLTE data with {len(self.volte_df)} records")
            else:
                print(f"VoLTE file not found: {self.volte_file_path}")
                self.volte_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading VoLTE data: {e}")
            import traceback
            print(traceback.format_exc())
            self.volte_df = pd.DataFrame()
    
    def normalize_msisdn(self, msisdn):
        """Normalize MSISDN format for consistent comparison"""
        if not msisdn:
            return None
        
        # Convert to string and strip whitespace
        msisdn = str(msisdn).strip()
        
        # Remove any non-digit characters
        msisdn_digits = ''.join(filter(str.isdigit, msisdn))
        
        # Return different variations for checking
        variations = [
            msisdn_digits,  # Just digits
            msisdn,  # Original format
        ]
        
        # Add country code variations if applicable
        if msisdn_digits.startswith('94') and len(msisdn_digits) > 10:
            # Remove country code
            variations.append(msisdn_digits[2:])
        elif not msisdn_digits.startswith('94') and len(msisdn_digits) >= 9:
            # Add country code
            variations.append('94' + msisdn_digits)
        
        return list(set(variations))  # Remove duplicates
    
    def check_volte_provisioning(self, msisdn):
        """
        Check if an MSISDN is provisioned for VoLTE
        
        Args:
            msisdn (str): The MSISDN to check
            
        Returns:
            dict: VoLTE provisioning information
        """
        if self.volte_df is None or self.volte_df.empty:
            return {
                "volte_provisioned": False,
                "error": "VoLTE data not available"
            }
        
        try:
            # Get normalized MSISDN variations
            msisdn_variations = self.normalize_msisdn(msisdn)
            print(f"Checking VoLTE for MSISDN variations: {msisdn_variations}")
            
            # Search for any variation of the MSISDN in the VoLTE data
            volte_match = None
            if msisdn_variations:
                for variation in msisdn_variations:
                    if variation:
                        match = self.volte_df[self.volte_df['MSISDN'] == variation]
                        if not match.empty:
                            volte_match = match
                            print(f"Found VoLTE match with variation: {variation}")
                            break
            
            if volte_match is not None and not volte_match.empty:
                # MSISDN found in VoLTE provisioned base
                volte_record = volte_match.iloc[0]
                
                # Get column values with proper handling
                def get_safe_value(record, column_name, default='Unknown'):
                    if column_name in record:
                        value = record[column_name]
                        return value if pd.notna(value) and str(value).strip() else default
                    return default
                
                result = {
                    "volte_provisioned": True,
                    "cell_code": get_safe_value(volte_record, 'CELL_CODE'),
                    "district": get_safe_value(volte_record, 'DISTRICT'),
                    "prepos": get_safe_value(volte_record, 'PREPOS'),
                    "sim_type": get_safe_value(volte_record, 'SIM_TYPE'),
                    "tac": get_safe_value(volte_record, 'TAC')
                }
                
                print(f"VoLTE provisioning result: {result}")
                return result
            else:
                # MSISDN not found in VoLTE provisioned base
                print(f"MSISDN not found in VoLTE data: {msisdn_variations}")
                return {
                    "volte_provisioned": False,
                    "message": "MSISDN not provisioned for VoLTE"
                }
                
        except Exception as e:
            print(f"Error checking VoLTE provisioning: {e}")
            import traceback
            print(traceback.format_exc())
            return {
                "volte_provisioned": False,
                "error": f"Error checking VoLTE provisioning: {str(e)}"
            }
    
    def get_volte_stats(self):
        """Get basic statistics about VoLTE provisioning data"""
        if self.volte_df is None or self.volte_df.empty:
            return {"error": "No VoLTE data available"}
        
        try:
            stats = {
                "total_provisioned": len(self.volte_df),
                "by_prepos": self.volte_df['PREPOS'].value_counts().to_dict() if 'PREPOS' in self.volte_df.columns else {},
                "by_sim_type": self.volte_df['SIM_TYPE'].value_counts().to_dict() if 'SIM_TYPE' in self.volte_df.columns else {},
                "by_district": self.volte_df['DISTRICT'].value_counts().head(10).to_dict() if 'DISTRICT' in self.volte_df.columns else {}
            }
            return stats
        except Exception as e:
            return {"error": f"Error getting VoLTE stats: {str(e)}"}

# Global instance for use in other modules
volte_checker = VoLTEChecker()

# Convenience function for easy import
def check_msisdn_volte_provisioning(msisdn):
    """
    Convenience function to check VoLTE provisioning for an MSISDN
    
    Args:
        msisdn (str): The MSISDN to check
        
    Returns:
        dict: VoLTE provisioning information
    """
    return volte_checker.check_volte_provisioning(msisdn)