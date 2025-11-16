#!/usr/bin/env python3
"""
Performance Quartiles GeoPackage Generator

Creates a GeoPackage grouping Nashville council districts into performance quartiles
based on median response time. Uses Parquet data from S3 and district boundaries 
to generate polygon-based performance analysis.
"""

import sys
import os
import io
import boto3
import pandas as pd
import geopandas as gpd
import tempfile
from datetime import datetime
from typing import Optional, Dict, Any
from shapely.geometry import Point

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


class PerformanceQuartilesGenerator:
    """Generator for district performance quartiles GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/performanceQuartiles'
        self.outputName = 'performanceQuartiles.gpkg'
        
    def loadDataFromS3(self, fileName: Optional[str] = None) -> pd.DataFrame:
        """Load latest Parquet data from S3."""
        if fileName is None:
            # Find latest parquet file
            response = self.s3Client.list_objects_v2(
                Bucket=self.bucketName, 
                Prefix='processed-data/'
            )
            
            if 'Contents' not in response or not response['Contents']:
                raise ValueError("No parquet files found in processed-data/")
            
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            fileName = latest['Key'].split('/')[-1]
        
        # Load parquet data
        response = self.s3Client.get_object(
            Bucket=self.bucketName,
            Key=f'processed-data/{fileName}'
        )
        
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    
    def calculatePerformanceMetrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate performance metrics and quartiles by district."""
        # Ensure required columns exist
        requiredCols = ['Council_District', 'Date_Time_Opened', 'Date_Time_Closed', 'Status']
        for col in requiredCols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Calculate response times for closed requests
        closed = df[df['Status'] == 'Closed'].copy()
        closed = closed[closed['Date_Time_Opened'].notna() & closed['Date_Time_Closed'].notna()]
        
        closed['openedDt'] = pd.to_datetime(closed['Date_Time_Opened'], unit='ms', errors='coerce')
        closed['closedDt'] = pd.to_datetime(closed['Date_Time_Closed'], unit='ms', errors='coerce')
        closed = closed[closed['openedDt'].notna() & closed['closedDt'].notna()]
        
        closed['responseHours'] = (closed['closedDt'] - closed['openedDt']).dt.total_seconds() / 3600.0
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        closed['districtInt'] = pd.to_numeric(closed['Council_District'], errors='coerce').astype('Int64')
        
        # Aggregate metrics by district
        byAll = df.groupby('districtInt', dropna=True).size().rename('totalRequests').to_frame()
        byClosed = closed.groupby('districtInt')['responseHours'].agg(['count', 'mean', 'median']).rename(columns={
            'count': 'closedRequests',
            'mean': 'avgResponseHours', 
            'median': 'medianResponseHours'
        })
        
        # Merge metrics
        metrics = byAll.join(byClosed, how='left').fillna({
            'closedRequests': 0,
            'avgResponseHours': 0.0,
            'medianResponseHours': 0.0
        })
        
        metrics = metrics.reset_index().rename(columns={'districtInt': 'District_ID'})
        
        # Calculate quartiles based on Median Response Time (fully dynamic - adapts to data)
        # Sort by median response hours (ascending: lower = better performance)
        metrics = metrics.sort_values('medianResponseHours', ascending=True).reset_index(drop=True)
        
        # Create quartiles using qcut (quantile-based cut - dynamically divides into 4 equal groups)
        # This is fully dynamic - quartile boundaries change with data distribution
        n = len(metrics)
        if n >= 4:
            # Use qcut for equal-sized quartiles (returns intervals)
            metrics['Performance_Quartile'] = pd.qcut(
                metrics['medianResponseHours'],
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'  # Handle cases where multiple districts have same median
            )
            
            # Calculate dynamic labels based on actual data (no hardcoding)
            # Calculate median response time for each quartile
            quartile_medians = {}
            quartile_boundaries = {}
            
            for quartile in ['Q1', 'Q2', 'Q3', 'Q4']:
                quartile_data = metrics[metrics['Performance_Quartile'] == quartile]['medianResponseHours']
                if len(quartile_data) > 0:
                    quartile_medians[quartile] = quartile_data.median()
                    quartile_boundaries[quartile] = {
                        'min': quartile_data.min(),
                        'max': quartile_data.max()
                    }
            
            # Calculate percentile ranges dynamically
            percentile_ranges = {
                'Q1': '0-25th',
                'Q2': '25-50th',
                'Q3': '50-75th',
                'Q4': '75-100th'
            }
            
            # Create dynamic labels: "X-Yth Percentile (Z hrs median)"
            # No hardcoded interpretations - purely data-driven
            metrics['Quartile_Label'] = metrics['Performance_Quartile'].apply(
                lambda q: f"{percentile_ranges.get(q, 'Unknown')} Percentile ({quartile_medians.get(q, 0):.1f} hrs median)" if pd.notna(q) else 'Unknown'
            )
        else:
            # If fewer than 4 districts, assign all to Q2 (middle quartile)
            metrics['Performance_Quartile'] = 'Q2'
            metrics['Quartile_Label'] = '25-50th Percentile'
        
        return metrics
    
    def loadDistrictBoundaries(self) -> gpd.GeoDataFrame:
        """Load and transform district boundaries from S3."""
        import tempfile
        
        # Download boundary file components from S3 to temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            # Download all boundary file components
            shapefileExtensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.shp.xml']
            baseName = '2022_Council_Districts'
            
            for ext in shapefileExtensions:
                s3Key = f'boundaries/nashvilleCouncilDistricts/{baseName}{ext}'
                localPath = os.path.join(tempDir, f'{baseName}{ext}')
                
                try:
                    self.s3Client.download_file(self.bucketName, s3Key, localPath)
                except Exception as e:
                    if ext == '.shp.xml':  # Optional file
                        continue
                    raise FileNotFoundError(f"Could not download {s3Key}: {e}")
            
            # Load the boundary file
            shapefilePath = os.path.join(tempDir, f'{baseName}.shp')
            boundaries = gpd.read_file(shapefilePath)
        
        # Determine district column
        if 'DISTRICT' in boundaries.columns:
            distCol = 'DISTRICT'
        elif 'District' in boundaries.columns:
            distCol = 'District'
        else:
            # Find first integer-like column
            intCols = [c for c in boundaries.columns if boundaries[c].dtype.kind in ('i', 'u')]
            if not intCols:
                raise ValueError("No district ID column found in boundaries")
            distCol = intCols[0]
        
        # Convert district to numeric
        boundaries['District_ID'] = pd.to_numeric(boundaries[distCol], errors='coerce').astype('Int64')
        
        # Include district name if available
        if 'DistrictNa' in boundaries.columns:
            boundaries['District_Name'] = boundaries['DistrictNa']
        elif 'DistrictName' in boundaries.columns:
            boundaries['District_Name'] = boundaries['DistrictName']
        else:
            # Create district name from ID if not available
            boundaries['District_Name'] = 'Council District ' + boundaries['District_ID'].astype(str)
        
        # Include representative name if available
        if 'Representa' in boundaries.columns:
            boundaries['Representative_Name'] = boundaries['Representa']
        elif 'Representative' in boundaries.columns:
            boundaries['Representative_Name'] = boundaries['Representative']
        elif 'RepresentativeName' in boundaries.columns:
            boundaries['Representative_Name'] = boundaries['RepresentativeName']
        else:
            # Set to empty if not available
            boundaries['Representative_Name'] = ''
        
        # Transform CRS to EPSG:4326
        if boundaries.crs is None:
            boundaries = boundaries.set_crs('EPSG:2274', allow_override=True)
        boundaries = boundaries.to_crs('EPSG:4326')
        
        return boundaries
    
    def createRequestPointsLayer(self, df: pd.DataFrame) -> gpd.GeoDataFrame:
        """
        Create a GeoDataFrame of request points from the data.
        
        Args:
            df: DataFrame with request data
            
        Returns:
            GeoDataFrame with Point geometries and request attributes
        """
        # Ensure required columns exist
        requiredCols = ['Latitude', 'Longitude']
        for col in requiredCols:
            if col not in df.columns:
                raise ValueError(f"Missing required column for points: {col}")
        
        # Filter out rows with missing coordinates
        points_df = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        
        # Calculate response time for closed requests
        points_df['Response_Hours'] = None
        closed_mask = (
            (points_df['Status'] == 'Closed') &
            points_df['Date_Time_Opened'].notna() &
            points_df['Date_Time_Closed'].notna()
        )
        
        if closed_mask.any():
            opened = pd.to_datetime(points_df.loc[closed_mask, 'Date_Time_Opened'], unit='ms', errors='coerce')
            closed = pd.to_datetime(points_df.loc[closed_mask, 'Date_Time_Closed'], unit='ms', errors='coerce')
            points_df.loc[closed_mask, 'Response_Hours'] = (closed - opened).dt.total_seconds() / 3600.0
        
        # Create Point geometries from coordinates
        points_df['geometry'] = points_df.apply(
            lambda row: Point(row['Longitude'], row['Latitude']), axis=1
        )
        
        # Select and rename columns with descriptive, Title Case names
        cols_to_include = [
            'Request__', 'Request_Type', 'Subrequest_Type', 'Status',
            'Address', 'City', 'Council_District', 'ZIP',
            'Date_Time_Opened', 'Date_Time_Closed', 'Response_Hours'
        ]
        
        # Only include columns that exist
        available_cols = [col for col in cols_to_include if col in points_df.columns]
        available_cols.append('geometry')
        
        points_gdf = gpd.GeoDataFrame(points_df[available_cols], crs='EPSG:4326')
        
        # Rename columns to camelCase for better ArcGIS Pro compatibility
        rename_map = {
            'Request__': 'requestId',
            'Request_Type': 'requestType',
            'Subrequest_Type': 'subrequestType',
            'Status': 'status',
            'Address': 'address',
            'City': 'city',
            'Council_District': 'councilDistrict',
            'ZIP': 'zipCode',
            'Date_Time_Opened': 'dateTimeOpened',
            'Date_Time_Closed': 'dateTimeClosed',
            'Response_Hours': 'responseTimeHours'
        }
        
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in points_gdf.columns}
        points_gdf = points_gdf.rename(columns=rename_map)
        
        # Ensure numeric fields are explicitly typed
        if 'councilDistrict' in points_gdf.columns:
            points_gdf['councilDistrict'] = pd.to_numeric(points_gdf['councilDistrict'], errors='coerce').astype('Int64')
        if 'responseTimeHours' in points_gdf.columns:
            points_gdf['responseTimeHours'] = points_gdf['responseTimeHours'].astype('float64')
        
        # Convert timestamp columns to readable dates if they exist
        if 'dateTimeOpened' in points_gdf.columns:
            points_gdf['dateTimeOpened'] = pd.to_datetime(
                points_gdf['dateTimeOpened'], unit='ms', errors='coerce'
            )
        if 'dateTimeClosed' in points_gdf.columns:
            points_gdf['dateTimeClosed'] = pd.to_datetime(
                points_gdf['dateTimeClosed'], unit='ms', errors='coerce'
            )
        
        return points_gdf
    
    def uploadGeoPackageToS3(self, localGpkgPath: str) -> Dict[str, str]:
        """
        Upload GeoPackage file to S3.
        
        Args:
            localGpkgPath: Path to local GeoPackage (.gpkg file)
            
        Returns:
            Dictionary with S3 URL
        """
        s3Key = f'{self.s3ShapefilePrefix}/{self.outputName}'
        
        # Upload to S3
        self.s3Client.upload_file(
            localGpkgPath,
            self.bucketName,
            s3Key,
            ExtraArgs={'ContentType': 'application/geopackage+sqlite3'}
        )
        
        # Generate public URL
        publicUrl = f'https://{self.bucketName}.s3.amazonaws.com/{s3Key}'
        
        return {
            'gpkg': publicUrl,
            'mainUrl': publicUrl
        }
    
    def createShapefile(self, fileName: Optional[str] = None) -> Dict[str, Any]:
        """
        Create the performance quartiles GeoPackage and upload to S3.
        
        Returns:
            Dictionary with S3 URLs and summary information
        """
        # Load data
        df = self.loadDataFromS3(fileName)
        metrics = self.calculatePerformanceMetrics(df)
        boundaries = self.loadDistrictBoundaries()
        
        # Merge boundaries with metrics
        gdf = boundaries.merge(metrics, on='District_ID', how='left')
        
        # Select and rename columns with descriptive, readable names
        # Include district name and representative if available
        cols = ['District_ID', 'totalRequests', 'closedRequests', 
                'avgResponseHours', 'medianResponseHours', 
                'Performance_Quartile', 'Quartile_Label', 'geometry']
        
        # Add district name if it exists
        if 'District_Name' in gdf.columns:
            cols.insert(1, 'District_Name')  # Insert after District_ID
        
        # Add representative name if it exists
        if 'Representative_Name' in gdf.columns:
            # Insert after District_Name (or after District_ID if District_Name doesn't exist)
            insert_index = 2 if 'District_Name' in cols else 1
            cols.insert(insert_index, 'Representative_Name')
        
        gdf = gdf[cols].rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName',
            'totalRequests': 'totalRequestsFromDistrict',
            'closedRequests': 'closedRequestsFromDistrict', 
            'avgResponseHours': 'averageResponseTimeHours',
            'medianResponseHours': 'medianResponseTimeHours',
            'Performance_Quartile': 'performanceQuartile',
            'Quartile_Label': 'quartileLabel'
        })
        
        # Ensure numeric fields are explicitly typed before writing to GeoPackage
        gdf['districtId'] = gdf['districtId'].astype('int64')
        gdf['totalRequestsFromDistrict'] = gdf['totalRequestsFromDistrict'].astype('int64')
        gdf['closedRequestsFromDistrict'] = gdf['closedRequestsFromDistrict'].astype('int64')
        gdf['averageResponseTimeHours'] = gdf['averageResponseTimeHours'].astype('float64')
        gdf['medianResponseTimeHours'] = gdf['medianResponseTimeHours'].astype('float64')
        
        # Get summary stats before writing
        quartile_counts = gdf['performanceQuartile'].value_counts().sort_index().to_dict()
        
        # Get Q1 and Q4 stats for summary
        q1_districts = gdf[gdf['performanceQuartile'] == 'Q1']
        q4_districts = gdf[gdf['performanceQuartile'] == 'Q4']
        
        q1_count = len(q1_districts)
        q4_count = len(q4_districts)
        # Use median instead of mean for summary stats (more robust)
        q1_median_hours = q1_districts['medianResponseTimeHours'].median() if q1_count > 0 else 0
        q4_median_hours = q4_districts['medianResponseTimeHours'].median() if q4_count > 0 else 0
        
        # Create request points layer
        points_gdf = self.createRequestPointsLayer(df)
        
        # Create GeoPackage in temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            # Write polygon layer (District Performance Quartiles)
            gdf.to_file(outputPath, driver='GPKG', layer='District Performance Quartiles')
            
            # Append point layer (Service Requests) to the same GeoPackage
            # This creates a multi-layer GeoPackage
            points_gdf.to_file(outputPath, driver='GPKG', layer='Service Requests', mode='a')
            
            # Upload to S3
            s3Urls = self.uploadGeoPackageToS3(outputPath)
        
        return {
            's3Urls': s3Urls,
            'mainUrl': s3Urls.get('mainUrl', ''),
            'summary': {
                'districts': len(gdf),
                'request_points': len(points_gdf),
                'quartile_counts': quartile_counts,
                'q1_count': q1_count,
                'q1_median_hours': float(q1_median_hours),
                'q4_count': q4_count,
                'q4_median_hours': float(q4_median_hours)
            }
        }


def main():
    """Main function to generate performance quartiles GeoPackage."""
    try:
        generator = PerformanceQuartilesGenerator()
        result = generator.createShapefile()
        
        print(f"Performance Quartiles GeoPackage created and uploaded to S3!")
        print(f"\nS3 Location: s3://{generator.bucketName}/{generator.s3ShapefilePrefix}/")
        print(f"\nPublic URL:")
        print(f"  GeoPackage: {result['mainUrl']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Districts: {summary['districts']}")
        print(f"  Request Points: {summary['request_points']:,}")
        print(f"  Quartile Distribution:")
        for quartile, count in sorted(summary['quartile_counts'].items()):
            print(f"    {quartile}: {count} districts")
        print(f"\n  Top 25% (Q1): {summary['q1_count']} districts (median {summary['q1_median_hours']:.2f} hrs)")
        print(f"  Needs Improvement (Q4): {summary['q4_count']} districts (median {summary['q4_median_hours']:.2f} hrs)")
        print(f"  (Quartiles based on Median Response Time - fully dynamic, adapts to data)")
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Performance Quartiles' (polygons)")
        print(f"     2. 'Service Requests' (points - {summary['request_points']:,} requests)")
        print(f"   Use quartile column for color-coding districts (4 groups)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

