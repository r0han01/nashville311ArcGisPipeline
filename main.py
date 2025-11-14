#!/usr/bin/env python3
"""
Nashville GIS Data Processing Application

Professional application for fetching and processing Nashville 311 service request data
for ArcGIS Pro integration. Always fetches the last 3 months dynamically.

Usage:
    python3 main.py

Author: GIS Analysis Team
Version: 1.0.0
"""

import sys
import os
from datetime import datetime

# Add src directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from nashvilleGis import NashvilleDataFetcher


def main():
    """Main application entry point - Always fetches last 3 months dynamically."""
    # Always use 3 months - no options needed
    months = 3
    
    try:
        # Create fetcher instance
        fetcher = NashvilleDataFetcher()
        
        # Fetch data - Always last 3 months dynamically
        data = fetcher.fetchDataByMonths(months)
        
        if data and data.get('features'):
            # Generate summary
            summary = fetcher.getDataSummary(data)
            
            print(f"Nashville GIS Data Processing Complete")
            print(f"Total records: {summary['totalRecords']:,}")
            print(f"Date range: {summary['dateRange']['start'][:10]} to {summary['dateRange']['end'][:10]}")
            print(f"S3 bucket: {fetcher.bucketName}")
            
            return data
            
        else:
            print("Failed to fetch data")
            sys.exit(1)
            
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
