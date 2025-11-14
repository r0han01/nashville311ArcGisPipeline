"""
Test module for Nashville GIS Library.

This module contains comprehensive unit tests for the Nashville GIS library
to ensure proper functionality, data quality, and professional standards.
"""

import unittest
import sys
import os
from datetime import datetime, timedelta
from unittest.mock import Mock, patch

# Add src directory to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


class TestNashvilleConfig(unittest.TestCase):
    """Test cases for NashvilleConfig class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = NashvilleConfig()
    
    def testApiUrlProperty(self):
        """Test API URL construction."""
        expectedUrl = "https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/hubNashville_311_Service_Requests_Current_Year_view/FeatureServer/0/query"
        self.assertEqual(self.config.apiUrl, expectedUrl)
    
    def testGetDateRangeMonths(self):
        """Test date range calculation for months."""
        startDate, endDate = self.config.getDateRangeMonths(3)
        
        # Check that endDate is today
        self.assertAlmostEqual(endDate.timestamp(), datetime.now().timestamp(), delta=10)
        
        # Check that startDate is approximately 3 months ago (allow more tolerance)
        expectedStart = datetime.now() - timedelta(days=90)  # Approximate
        self.assertAlmostEqual(startDate.timestamp(), expectedStart.timestamp(), delta=259200)  # Within 3 days
    
    def testGetDateRangeDays(self):
        """Test date range calculation for days."""
        startDate, endDate = self.config.getDateRangeDays(30)
        
        # Check that endDate is today
        self.assertAlmostEqual(endDate.timestamp(), datetime.now().timestamp(), delta=10)
        
        # Check that startDate is 30 days ago
        expectedStart = datetime.now() - timedelta(days=30)
        self.assertAlmostEqual(startDate.timestamp(), expectedStart.timestamp(), delta=3600)  # Within 1 hour
    
    def testValidateCoordinates(self):
        """Test coordinate validation."""
        # Valid Nashville coordinates
        self.assertTrue(self.config.validateCoordinates(36.1627, -86.7816))
        
        # Invalid coordinates (outside Nashville area)
        self.assertFalse(self.config.validateCoordinates(40.7128, -74.0060))  # New York
        self.assertFalse(self.config.validateCoordinates(0.0, 0.0))  # Null Island


class TestNashvilleDataFetcher(unittest.TestCase):
    """Test cases for NashvilleDataFetcher class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = NashvilleDataFetcher()
    
    def testInitialization(self):
        """Test fetcher initialization."""
        self.assertIsNotNone(self.fetcher.config)
        self.assertIsNotNone(self.fetcher.bucketName)
        self.assertIsNotNone(self.fetcher.session)
    
    def testGetApiParams(self):
        """Test API parameters generation."""
        params = self.fetcher.config.getApiParams(pageSize=1000, offset=2000)
        
        expectedParams = {
            'outFields': '*',
            'where': '1=1',
            'resultRecordCount': 1000,
            'resultOffset': 2000,
            'orderByFields': 'Date_Time_Opened',
            'f': 'geojson'
        }
        
        self.assertEqual(params, expectedParams)
    
    def testFilterByDateRange(self):
        """Test date range filtering."""
        # Create mock features with different dates
        now = datetime.now()
        oldDate = now - timedelta(days=100)
        recentDate = now - timedelta(days=10)
        
        features = [
            {
                'properties': {
                    'Date_Time_Opened': int(oldDate.timestamp() * 1000)
                }
            },
            {
                'properties': {
                    'Date_Time_Opened': int(recentDate.timestamp() * 1000)
                }
            }
        ]
        
        # Filter for last 30 days
        startDate = now - timedelta(days=30)
        endDate = now
        
        filtered = self.fetcher.filterByDateRange(features, startDate, endDate)
        
        # Should only include the recent feature
        self.assertEqual(len(filtered), 1)
        self.assertEqual(
            filtered[0]['properties']['Date_Time_Opened'],
            int(recentDate.timestamp() * 1000)
        )
    
    def testGetDataSummary(self):
        """Test data summary generation."""
        # Create mock data
        data = {
            'features': [
                {
                    'properties': {
                        'Request_Type': 'Cart Service',
                        'Council_District': 12,
                        'Status': 'Closed'
                    }
                },
                {
                    'properties': {
                        'Request_Type': 'Cart Service',
                        'Council_District': 12,
                        'Status': 'Open'
                    }
                },
                {
                    'properties': {
                        'Request_Type': 'Noise Complaint',
                        'Council_District': 15,
                        'Status': 'Closed'
                    }
                }
            ]
        }
        
        summary = self.fetcher.getDataSummary(data)
        
        self.assertEqual(summary['totalRecords'], 3)
        self.assertEqual(summary['topRequestTypes']['Cart Service'], 2)
        self.assertEqual(summary['topRequestTypes']['Noise Complaint'], 1)
        self.assertEqual(summary['topDistricts'][12], 2)
        self.assertEqual(summary['topDistricts'][15], 1)
        self.assertEqual(summary['statusDistribution']['Closed'], 2)
        self.assertEqual(summary['statusDistribution']['Open'], 1)


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete workflow."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.fetcher = NashvilleDataFetcher()
    
    @patch('nashvilleGis.dataFetcher.boto3.client')
    def testS3UploadMock(self, mockBotoClient):
        """Test S3 upload functionality with mocked boto3."""
        # Mock S3 client
        mockS3 = Mock()
        mockBotoClient.return_value = mockS3
        
        # Create fetcher after mock is applied
        testFetcher = NashvilleDataFetcher()
        
        # Test data
        testData = {
            'type': 'FeatureCollection',
            'properties': {'test': 'data'},
            'features': []
        }
        
        # Test upload
        result = testFetcher.uploadToS3(testData, 'test.json')
        
        # Verify S3 client was called
        mockBotoClient.assert_called_once()
        mockS3.put_object.assert_called_once()
        
        # Verify upload was successful
        self.assertTrue(result)


if __name__ == '__main__':
    # Run tests with verbose output
    unittest.main(verbosity=2)
