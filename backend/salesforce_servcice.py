"""
Salesforce Service for retrieving real data from Salesforce
Make sure to set these environment variables:
- SF_USERNAME: Your Salesforce username
- SF_PASSWORD: Your Salesforce password
- SF_SECURITY_TOKEN: Your Salesforce security token
- SF_DOMAIN: Your Salesforce domain (login, test, etc.)
"""

import os
from dotenv import load_dotenv, find_dotenv
from simple_salesforce import Salesforce

# Load environment variables
env_path = find_dotenv()
if env_path:
    load_dotenv(env_path)
else:
    from pathlib import Path
    for candidate in [
        Path(__file__).parent / ".env",
        Path(__file__).parent.parent / ".env",
    ]:
        if candidate.exists():
            load_dotenv(candidate)
            break


class SalesforcesService:
    """Real Salesforce service that connects to your Salesforce instance."""
    
    def __init__(self):
        """Initialize the Salesforce service with real credentials."""
        username = os.getenv("SF_USERNAME")
        password = os.getenv("SF_PASSWORD")
        security_token = os.getenv("SF_SECURITY_TOKEN")
        domain = os.getenv("SF_DOMAIN", "login")
        
        if not all([username, password, security_token]):
            raise ValueError(
                "Missing Salesforce credentials. Please set SF_USERNAME, SF_PASSWORD, "
                "and SF_SECURITY_TOKEN environment variables in your .env file"
            )
        
        try:
            self.sf = Salesforce(
                username=username,
                password=password,
                security_token=security_token,
                instance_url=f"https://{domain}.salesforce.com"
            )
            print("[SalesforcesService] Successfully connected to Salesforce")
        except Exception as e:
            print(f"[SalesforcesService] Error connecting to Salesforce: {e}")
            raise
    
    def excute_soql(self, query: str):
        """
        Execute a SOQL query against Salesforce.
        
        Args:
            query: SOQL query string
            
        Returns:
            List of dictionaries with query results
        """
        try:
            print(f"[SalesforcesService] Executing query: {query.strip()}")
            result = self.sf.query_all(query)
            records = result.get("records", [])
            print(f"[SalesforcesService] Retrieved {len(records)} records")
            return records
        except Exception as e:
            print(f"[SalesforcesService] Error executing query: {e}")
            raise

