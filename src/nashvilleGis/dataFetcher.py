"""
Nashville 311 Data Fetcher Module.

This module provides functionality to fetch Nashville 311 service request data
from the ArcGIS REST API with dynamic date filtering and AWS S3 integration.
"""

import requests
import boto3
import json
import time
import pandas as pd
import io
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional

from .config import NashvilleConfig


class NashvilleDataFetcher:
    """
    Professional data fetcher for Nashville 311 service requests.
    
    This class provides methods to fetch, filter, and store Nashville 311 data
    with dynamic date range support and AWS S3 integration.
    """
    
    def __init__(self, bucketName: Optional[str] = None):
        """
        Initialize the Nashville Data Fetcher.
        
        Args:
            bucketName: Optional custom S3 bucket name
        """
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3', region_name=self.config.AWS_REGION)
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Nashville-GIS-Analysis/1.0.0'
        })
    
    def getTotalRecordCount(self) -> int:
        """
        Get the total number of records available from the API.
        
        Returns:
            Total record count
        """
        try:
            params = {
                'where': '1=1',
                'returnCountOnly': 'true',
                'f': 'json'
            }
            
            response = self.session.get(
                self.config.apiUrl, 
                params=params, 
                timeout=self.config.REQUEST_TIMEOUT
            )
            response.raise_for_status()
            
            data = response.json()
            return data.get('count', 152026)  # Fallback to known count
            
        except Exception as e:
            print(f"Warning: Could not get record count from API: {e}")
            return 152026  # Fallback to known total
    
    def fetchAllData(self) -> List[Dict[str, Any]]:
        """
        Fetch all data from the Nashville 311 API with pagination.
        
        Returns:
            List of all feature records
        """
        totalRecords = self.getTotalRecordCount()
        totalPages = (totalRecords + self.config.PAGE_SIZE - 1) // self.config.PAGE_SIZE
        
        allFeatures = []
        successfulPages = 0
        
        for page in range(totalPages):
            offset = page * self.config.PAGE_SIZE
            
            try:
                params = self.config.getApiParams(offset=offset)
                response = self.session.get(
                    self.config.apiUrl, 
                    params=params, 
                    timeout=self.config.REQUEST_TIMEOUT
                )
                response.raise_for_status()
                
                pageData = response.json()
                
                if pageData and 'features' in pageData:
                    features = pageData['features']
                    allFeatures.extend(features)
                    successfulPages += 1
                else:
                    break
                    
            except Exception as e:
                break
            
            # Rate limiting
            time.sleep(self.config.RATE_LIMIT_DELAY)
        
        return allFeatures
    
    def filterByDateRange(self, features: List[Dict], startDate: datetime, endDate: datetime) -> List[Dict]:
        """
        Filter features by date range.
        
        Args:
            features: List of feature records
            startDate: Start date for filtering
            endDate: End date for filtering
            
        Returns:
            Filtered list of features
        """
        filteredFeatures = []
        startTimestamp = int(startDate.timestamp() * 1000)
        endTimestamp = int(endDate.timestamp() * 1000)
        
        for feature in features:
            openedTime = feature.get('properties', {}).get('Date_Time_Opened')
            if openedTime and startTimestamp <= openedTime <= endTimestamp:
                filteredFeatures.append(feature)
        
        return filteredFeatures
    
    def uploadToS3(self, data: Dict[str, Any], fileName: str) -> bool:
        """
        Upload data to S3 bucket in Parquet format for production-level performance.
        
        Args:
            data: Data dictionary to upload
            fileName: Filename for the S3 object
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Convert features to DataFrame
            features = data.get('features', [])
            if not features:
                return False
            
            # Extract properties from features
            records = []
            for feature in features:
                props = feature.get('properties', {})
                records.append(props)
            
            # Create DataFrame
            df = pd.DataFrame(records)
            
            # Convert to Parquet format
            parquetBuffer = io.BytesIO()
            df.to_parquet(parquetBuffer, index=False, engine='pyarrow')
            parquetData = parquetBuffer.getvalue()
            
            # Upload Parquet to S3
            parquetFileName = fileName.replace('.json', '.parquet')
            self.s3Client.put_object(
                Bucket=self.bucketName,
                Key=f'processed-data/{parquetFileName}',
                Body=parquetData,
                ContentType='application/octet-stream',
                Metadata={
                    'created-by': 'Nashville-GIS-Analysis',
                    'version': '1.0.0',
                    'data-type': 'nashville-311-service-requests',
                    'format': 'parquet',
                    'total-records': str(len(df)),
                    'date-range': f"{data.get('properties', {}).get('dateRange', {}).get('start', 'N/A')[:10]} to {data.get('properties', {}).get('dateRange', {}).get('end', 'N/A')[:10]}"
                }
            )
            return True
        except Exception as e:
            print(f"Error uploading to S3: {e}")
            return False
    
    def loadDataFromS3(self, fileName: str) -> pd.DataFrame:
        """
        Load data from S3 in Parquet format.
        
        Args:
            fileName: Parquet filename in S3
            
        Returns:
            DataFrame with the data
        """
        try:
            response = self.s3Client.get_object(
                Bucket=self.bucketName,
                Key=f'processed-data/{fileName}'
            )
            return pd.read_parquet(io.BytesIO(response['Body'].read()))
        except Exception as e:
            print(f"Error loading data from S3: {e}")
            return pd.DataFrame()
    
    def fetchDataByMonths(self, months: int = 3) -> Dict[str, Any]:
        """
        Fetch data for the last N months.
        
        Args:
            months: Number of months to fetch (default: 3)
            
        Returns:
            Dictionary containing fetched and filtered data
        """
        # Calculate date range
        startDate, endDate = self.config.getDateRangeMonths(months)
        
        # Fetch all data
        allFeatures = self.fetchAllData()
        if not allFeatures:
            return {}
        
        # Filter by date range
        filteredFeatures = self.filterByDateRange(allFeatures, startDate, endDate)
        
        if not filteredFeatures:
            return {}
        
        # Create combined data structure
        combinedData = {
            'type': 'FeatureCollection',
            'properties': {
                'totalRecords': len(filteredFeatures),
                'originalRecords': len(allFeatures),
                'dateRange': {
                    'start': startDate.isoformat(),
                    'end': endDate.isoformat()
                },
                'fetchDate': datetime.now().isoformat(),
                'description': f'Last {months} months of Nashville 311 data',
                'monthsRequested': months,
                'apiSource': self.config.apiUrl
            },
            'features': filteredFeatures
        }
        
        # Upload to S3
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        fileName = f'nashville311_last{months}months_{startDate.strftime("%Y-%m")}_to_{endDate.strftime("%Y-%m")}_{timestamp}.json'
        
        self.uploadToS3(combinedData, fileName)
        
        return combinedData
    
    def getDataSummary(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary statistics for fetched data.
        
        Args:
            data: Fetched data dictionary
            
        Returns:
            Summary statistics dictionary
        """
        if not data or 'features' not in data:
            return {}
        
        features = data['features']
        requestTypes = {}
        districts = {}
        statuses = {}
        
        for feature in features:
            props = feature.get('properties', {})
            
            # Count request types
            reqType = props.get('Request_Type', 'Unknown')
            requestTypes[reqType] = requestTypes.get(reqType, 0) + 1
            
            # Count districts
            district = props.get('Council_District', 'Unknown')
            districts[district] = districts.get(district, 0) + 1
            
            # Count statuses
            status = props.get('Status', 'Unknown')
            statuses[status] = statuses.get(status, 0) + 1
        
        return {
            'totalRecords': len(features),
            'dateRange': data.get('properties', {}).get('dateRange', {}),
            'topRequestTypes': dict(sorted(requestTypes.items(), key=lambda x: x[1], reverse=True)[:10]),
            'topDistricts': dict(sorted(districts.items(), key=lambda x: x[1], reverse=True)[:10]),
            'statusDistribution': statuses
        }
