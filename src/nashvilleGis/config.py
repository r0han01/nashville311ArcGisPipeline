"""
Configuration module for Nashville GIS data processing.

This module contains all configuration settings, constants, and utility functions
for the Nashville 311 data processing pipeline.
"""

from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from typing import Tuple, Dict, Any


class NashvilleConfig:
    """
    Configuration class for Nashville 311 Data Processing.
    
    This class provides centralized configuration for all aspects of the
    data processing pipeline including API settings, AWS configuration,
    and data processing parameters.
    """
    
    # Contact / Support
    SUPPORT_EMAIL = "hello@rkatkam.com"
    SUPPORT_SUBJECT_TEMPLATE = "nashville311ArcGisPipeline - <describe issue>"

    # AWS Configuration
    BUCKET_NAME = "nashville311-gis-analysis-data"
    AWS_REGION = "us-east-1"
    
    # API Configuration
    API_BASE_URL = "https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services"
    API_ENDPOINT = "hubNashville_311_Service_Requests_Current_Year_view/FeatureServer/0/query"
    
    @property
    def apiUrl(self) -> str:
        """Get the complete API URL."""
        return f"{self.API_BASE_URL}/{self.API_ENDPOINT}"
    
    # Data Fetching Configuration
    PAGE_SIZE = 2000
    RATE_LIMIT_DELAY = 0.5  # seconds between requests
    REQUEST_TIMEOUT = 60  # seconds
    
    # Coordinate Reference System
    OUTPUT_CRS = "EPSG:4326"  # WGS84
    
    # Field mappings for data processing
    FIELD_MAPPINGS = {
        'request_id': 'Request__',
        'request_type': 'Request_Type',
        'subrequest_type': 'Subrequest_Type',
        'status': 'Status',
        'city': 'City',
        'district': 'Council_District',
        'zip_code': 'ZIP',
        'address': 'Address',
        'latitude': 'Latitude',
        'longitude': 'Longitude',
        'opened_date': 'Date_Time_Opened',
        'closed_date': 'Date_Time_Closed'
    }
    
    # Data quality thresholds
    MIN_LATITUDE = 35.5
    MAX_LATITUDE = 36.5
    MIN_LONGITUDE = -87.5
    MAX_LONGITUDE = -86.0
    
    @staticmethod
    def getDateRangeMonths(months: int) -> Tuple[datetime, datetime]:
        """
        Calculate date range for last N months from current date.
        
        Args:
            months: Number of months to go back
            
        Returns:
            Tuple of (start_date, end_date)
        """
        today = datetime.now()
        start_date = today - relativedelta(months=months)
        return start_date, today
    
    @staticmethod
    def getDateRangeDays(days: int) -> Tuple[datetime, datetime]:
        """
        Calculate date range for last N days from current date.
        
        Args:
            days: Number of days to go back
            
        Returns:
            Tuple of (start_date, end_date)
        """
        today = datetime.now()
        start_date = today - timedelta(days=days)
        return start_date, today
    
    @staticmethod
    def getCustomDateRange(startDateStr: str, endDateStr: str) -> Tuple[datetime, datetime]:
        """
        Create date range from custom string dates.
        
        Args:
            startDateStr: Start date in 'YYYY-MM-DD' format
            endDateStr: End date in 'YYYY-MM-DD' format
            
        Returns:
            Tuple of (start_date, end_date)
        """
        startDate = datetime.strptime(startDateStr, '%Y-%m-%d')
        endDate = datetime.strptime(endDateStr, '%Y-%m-%d')
        return startDate, endDate
    
    @staticmethod
    def validateCoordinates(lat: float, lon: float) -> bool:
        """
        Validate if coordinates are within Nashville area bounds.
        
        Args:
            lat: Latitude
            lon: Longitude
            
        Returns:
            True if coordinates are valid, False otherwise
        """
        config = NashvilleConfig()
        return (config.MIN_LATITUDE <= lat <= config.MAX_LATITUDE and
                config.MIN_LONGITUDE <= lon <= config.MAX_LONGITUDE)
    
    def getApiParams(self, pageSize: int = None, offset: int = 0) -> Dict[str, Any]:
        """
        Get standard API parameters for requests.
        
        Args:
            pageSize: Number of records per page
            offset: Starting record offset
            
        Returns:
            Dictionary of API parameters
        """
        return {
            'outFields': '*',
            'where': '1=1',
            'resultRecordCount': pageSize or self.PAGE_SIZE,
            'resultOffset': offset,
            'orderByFields': 'Date_Time_Opened',
            'f': 'geojson'
        }
