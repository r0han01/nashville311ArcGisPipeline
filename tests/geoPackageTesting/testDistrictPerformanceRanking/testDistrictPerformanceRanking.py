#!/usr/bin/env python3
"""
Tests for District Performance Ranking GeoPackage Generator
"""

import unittest
import sys
import os
import pandas as pd
import geopandas as gpd
from unittest.mock import Mock, patch, MagicMock
import io

# Add src and geoPackageGenerators to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'src'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', '..', 'geoPackageGenerators', 'districtPerformanceRanking'))

from districtPerformanceRanking import DistrictPerformanceRankingGenerator


class TestDistrictPerformanceRankingGenerator(unittest.TestCase):
    """Test cases for DistrictPerformanceRankingGenerator class."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.generator = DistrictPerformanceRankingGenerator()
        
        # Create sample test data
        self.sampleData = pd.DataFrame({
            'Council_District': [1, 1, 2, 2, 3, 3, 1, 2],
            'Status': ['Closed', 'Closed', 'Closed', 'Open', 'Closed', 'Closed', 'Closed', 'Closed'],
            'Date_Time_Opened': [1640995200000, 1641081600000, 1641168000000, 1641254400000, 1641340800000, 1641427200000, 1641513600000, 1641600000000],  # Jan 2022 timestamps
            'Date_Time_Closed': [1640998800000, 1641085200000, 1641171600000, None, 1641344400000, 1641430800000, 1641517200000, 1641603600000]  # 1 hour later for closed ones
        })
    
    def testInitialization(self):
        """Test generator initialization."""
        self.assertIsNotNone(self.generator.config)
        self.assertIsNotNone(self.generator.bucketName)
        self.assertIsNotNone(self.generator.s3Client)
        self.assertEqual(self.generator.boundaryS3Key, 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp')
        self.assertEqual(self.generator.outputDir, 'data/shapefiles/districtPerformanceRanking')
        self.assertEqual(self.generator.outputName, 'districtPerformanceRanking.shp')
    
    def testCalculatePerformanceMetrics(self):
        """Test performance metrics calculation."""
        metrics = self.generator.calculatePerformanceMetrics(self.sampleData)
        
        # Check that metrics are calculated correctly
        self.assertIn('District_ID', metrics.columns)
        self.assertIn('totalRequests', metrics.columns)
        self.assertIn('closedRequests', metrics.columns)
        self.assertIn('avgResponseHours', metrics.columns)
        self.assertIn('medianResponseHours', metrics.columns)
        self.assertIn('Performance_Rank', metrics.columns)
        self.assertIn('Percentile_Rank', metrics.columns)
        
        # Check that districts are present
        districts = metrics['District_ID'].tolist()
        self.assertIn(1, districts)
        self.assertIn(2, districts)
        self.assertIn(3, districts)
        
        # Check that rankings are calculated
        self.assertTrue(metrics['Performance_Rank'].min() == 1)
        self.assertTrue(metrics['Performance_Rank'].max() <= len(metrics))
        
        # Check percentiles
        self.assertTrue(metrics['Percentile_Rank'].min() >= 0)
        self.assertTrue(metrics['Percentile_Rank'].max() <= 100)
    
    def testCalculatePerformanceMetricsWithMissingColumns(self):
        """Test metrics calculation with missing required columns."""
        incompleteData = pd.DataFrame({
            'Council_District': [1, 2, 3],
            'Status': ['Closed', 'Closed', 'Closed']
            # Missing Date_Time_Opened and Date_Time_Closed
        })
        
        with self.assertRaises(ValueError) as context:
            self.generator.calculatePerformanceMetrics(incompleteData)
        
        self.assertIn("Missing required column", str(context.exception))
    
    @patch('districtPerformanceRanking.boto3.client')
    def testLoadDataFromS3(self, mockBotoClient):
        """Test loading data from S3."""
        # Mock S3 client
        mockS3 = Mock()
        mockBotoClient.return_value = mockS3
        
        # Mock list_objects_v2 response
        mockS3.list_objects_v2.return_value = {
            'Contents': [
                {
                    'Key': 'processed-data/test_file.parquet',
                    'LastModified': '2025-01-01T00:00:00Z'
                }
            ]
        }
        
        # Mock get_object response
        mockParquetData = io.BytesIO()
        pd.DataFrame({'test': [1, 2, 3]}).to_parquet(mockParquetData, index=False)
        mockParquetData.seek(0)
        
        mockS3.get_object.return_value = {
            'Body': Mock(read=lambda: mockParquetData.getvalue())
        }
        
        # Create generator with mocked S3
        generator = DistrictPerformanceRankingGenerator()
        
        # Test loading data
        df = generator.loadDataFromS3()
        
        # Verify S3 calls
        mockS3.list_objects_v2.assert_called_once()
        mockS3.get_object.assert_called_once()
        
        # Verify DataFrame
        self.assertIsInstance(df, pd.DataFrame)
    
    @patch('districtPerformanceRanking.boto3.client')
    def testLoadDataFromS3NoFiles(self, mockBotoClient):
        """Test loading data from S3 when no files exist."""
        # Mock S3 client
        mockS3 = Mock()
        mockBotoClient.return_value = mockS3
        
        # Mock empty list_objects_v2 response
        mockS3.list_objects_v2.return_value = {}
        
        generator = DistrictPerformanceRankingGenerator()
        
        with self.assertRaises(ValueError) as context:
            generator.loadDataFromS3()
        
        self.assertIn("No parquet files found", str(context.exception))
    
    @patch('geopandas.read_file')
    @patch('tempfile.TemporaryDirectory')
    def testLoadDistrictBoundaries(self, mockTempDir, mockReadFile):
        """Test loading district boundaries from S3."""
        # Mock temporary directory
        mockTempDir.return_value.__enter__.return_value = '/tmp/test'
        
        # Mock GeoDataFrame
        mockGdf = gpd.GeoDataFrame({
            'DISTRICT': [1, 2, 3],
            'geometry': [None, None, None]  # Mock geometries
        })
        mockGdf.crs = None
        mockReadFile.return_value = mockGdf
        
        # Mock S3 client download
        with patch.object(self.generator.s3Client, 'download_file') as mockDownload:
            boundaries = self.generator.loadDistrictBoundaries()
            
            # Verify S3 download was called for each boundary file component
            expectedCalls = 6  # .shp, .shx, .dbf, .prj, .cpg, .shp.xml
            self.assertEqual(mockDownload.call_count, expectedCalls)
            
            # Verify District_ID column was created
            self.assertIn('District_ID', boundaries.columns)
    
    @patch('tempfile.TemporaryDirectory')
    def testLoadDistrictBoundariesFileNotFound(self, mockTempDir):
        """Test loading boundaries when S3 file doesn't exist."""
        # Mock temporary directory
        mockTempDir.return_value.__enter__.return_value = '/tmp/test'
        
        # Mock S3 download failure
        with patch.object(self.generator.s3Client, 'download_file', side_effect=Exception("S3 file not found")):
            with self.assertRaises(FileNotFoundError):
                self.generator.loadDistrictBoundaries()
    
    @patch.object(DistrictPerformanceRankingGenerator, 'loadDataFromS3')
    @patch.object(DistrictPerformanceRankingGenerator, 'loadDistrictBoundaries')
    @patch('geopandas.GeoDataFrame')
    @patch('os.makedirs')
    @patch('os.path.exists')
    @patch('os.remove')
    def testCreateShapefile(self, mockRemove, mockExists, mockMakedirs, mockGdf, mockLoadBoundaries, mockLoadData):
        """Test complete GeoPackage creation process."""
        # Mock data loading
        mockLoadData.return_value = self.sampleData
        
        # Mock boundaries
        mockBoundaries = gpd.GeoDataFrame({
            'District_ID': [1, 2, 3],
            'geometry': [None, None, None]
        })
        mockLoadBoundaries.return_value = mockBoundaries
        
        # Mock file operations
        mockExists.return_value = False
        
        # Mock GeoDataFrame operations
        mockGdfInstance = Mock()
        mockGdfInstance.merge.return_value = mockGdfInstance
        mockGdfInstance.__getitem__ = Mock(return_value=mockGdfInstance)
        mockGdfInstance.rename.return_value = mockGdfInstance
        mockGdf.return_value = mockGdfInstance
        
        # Test GeoPackage creation
        outputPath = self.generator.createShapefile()
        
        # Verify calls
        mockLoadData.assert_called_once()
        mockLoadBoundaries.assert_called_once()
        mockMakedirs.assert_called_once()
        
        # Verify output path
        expectedPath = os.path.join(self.generator.outputDir, self.generator.outputName)
        self.assertEqual(outputPath, expectedPath)


if __name__ == '__main__':
    unittest.main(verbosity=2)
