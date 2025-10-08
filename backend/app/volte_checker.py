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
                # Read CSV without headers since the file doesn't seem to have them
                # Based on the sample data, columns appear to be: MSISDN, CELL_CODE, DISTRICT, PREPOS, SIM_TYPE, TAC
                self.volte_df = pd.read_csv(
                    self.volte_file_path, 
                    names=['MSISDN', 'CELL_CODE', 'DISTRICT', 'PREPOS', 'SIM_TYPE', 'TAC'],
                    dtype={'MSISDN': str, 'TAC': str}
                )
                print(f"Loaded VoLTE data: {len(self.volte_df)} records")
            else:
                print(f"VoLTE file not found: {self.volte_file_path}")
                self.volte_df = pd.DataFrame()
        except Exception as e:
            print(f"Error loading VoLTE data: {e}")
            self.volte_df = pd.DataFrame()
    
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
            # Search for the MSISDN in the VoLTE data
            volte_match = self.volte_df[self.volte_df['MSISDN'] == str(msisdn)]
            
            if not volte_match.empty:
                # MSISDN found in VoLTE provisioned base
                volte_record = volte_match.iloc[0]
                
                return {
                    "volte_provisioned": True,
                    "cell_code": volte_record.get('CELL_CODE', 'Unknown') if pd.notna(volte_record.get('CELL_CODE')) else 'Unknown',
                    "district": volte_record.get('DISTRICT', 'Unknown') if pd.notna(volte_record.get('DISTRICT')) else 'Unknown',
                    "prepos": volte_record.get('PREPOS', 'Unknown') if pd.notna(volte_record.get('PREPOS')) else 'Unknown',
                    "sim_type": volte_record.get('SIM_TYPE', 'Unknown') if pd.notna(volte_record.get('SIM_TYPE')) else 'Unknown',
                    "tac": volte_record.get('TAC', 'Unknown') if pd.notna(volte_record.get('TAC')) else 'Unknown'
                }
            else:
                # MSISDN not found in VoLTE provisioned base
                return {
                    "volte_provisioned": False,
                    "message": "MSISDN not provisioned for VoLTE"
                }
                
        except Exception as e:
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