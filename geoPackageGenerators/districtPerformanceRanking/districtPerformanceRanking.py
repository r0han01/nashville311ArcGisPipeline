#!/usr/bin/env python3
"""
District Performance Ranking Shapefile Generator

Creates a shapefile ranking Nashville council districts by response time performance.
Uses Parquet data from S3 and district boundaries to generate polygon-based performance analysis.
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

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


class DistrictPerformanceRankingGenerator:
    """Generator for district performance ranking shapefiles."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/districtPerformanceRanking'
        self.outputName = 'districtPerformanceRanking.gpkg'
        
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
        """Calculate performance metrics by district."""
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
        
        # Calculate rankings based on Median Response Time (more robust, less skewed by outliers)
        # Sort by median response hours (ascending: lower = better performance)
        metrics = metrics.sort_values('medianResponseHours', ascending=True).reset_index(drop=True)
        metrics['Performance_Rank_By_Median'] = metrics.index + 1
        
        # Calculate percentiles based on median ranking
        n = len(metrics)
        if n > 1:
            metrics['Percentile_Rank'] = ((n - metrics['Performance_Rank_By_Median']) / (n - 1) * 100).round().astype('Int64')
        else:
            metrics['Percentile_Rank'] = 100
        
        return metrics
    
    def loadDistrictBoundaries(self) -> gpd.GeoDataFrame:
        """Load and transform district boundaries from S3."""
        import tempfile
        
        # Download shapefile components from S3 to temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            # Download all shapefile components
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
            
            # Load the shapefile
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
        Create the district performance ranking shapefile and upload to S3.
        
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
                'Performance_Rank_By_Median', 'Percentile_Rank', 'geometry']
        
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
            'Performance_Rank_By_Median': 'performanceRankBasedOnMedianResponseTime',
            'Percentile_Rank': 'percentileRank'
        })
        
        # Ensure numeric fields are explicitly typed before writing to GeoPackage
        gdf['performanceRankBasedOnMedianResponseTime'] = gdf['performanceRankBasedOnMedianResponseTime'].astype('int64')
        gdf['percentileRank'] = gdf['percentileRank'].astype('Int64').fillna(0).astype('int64')
        gdf['districtId'] = gdf['districtId'].astype('int64')
        gdf['totalRequestsFromDistrict'] = gdf['totalRequestsFromDistrict'].astype('int64')
        gdf['closedRequestsFromDistrict'] = gdf['closedRequestsFromDistrict'].astype('int64')
        gdf['averageResponseTimeHours'] = gdf['averageResponseTimeHours'].astype('float64')
        gdf['medianResponseTimeHours'] = gdf['medianResponseTimeHours'].astype('float64')
        
        # Get summary stats before writing (using median-based ranking)
        bestDistrict = gdf.loc[gdf['performanceRankBasedOnMedianResponseTime'] == 1, 'districtId'].iloc[0]
        bestMedianHours = gdf.loc[gdf['performanceRankBasedOnMedianResponseTime'] == 1, 'medianResponseTimeHours'].iloc[0]
        worstDistrict = gdf.loc[gdf['performanceRankBasedOnMedianResponseTime'] == gdf['performanceRankBasedOnMedianResponseTime'].max(), 'districtId'].iloc[0]
        worstMedianHours = gdf.loc[gdf['performanceRankBasedOnMedianResponseTime'] == gdf['performanceRankBasedOnMedianResponseTime'].max(), 'medianResponseTimeHours'].iloc[0]
        
        # Create GeoPackage in temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            # Write GeoPackage to temp directory
            # Use descriptive layer name for better display in ArcGIS Pro
            gdf.to_file(outputPath, driver='GPKG', layer='District Performance Ranking')
            
            # Upload to S3
            s3Urls = self.uploadGeoPackageToS3(outputPath)
        
        return {
            's3Urls': s3Urls,
            'mainUrl': s3Urls.get('mainUrl', ''),
            'summary': {
                'districts': len(gdf),
                'bestDistrict': int(bestDistrict),
                'bestMedianHours': float(bestMedianHours),
                'worstDistrict': int(worstDistrict),
                'worstMedianHours': float(worstMedianHours)
            }
        }


def main():
    """Main function to generate district performance ranking shapefile."""
    try:
        generator = DistrictPerformanceRankingGenerator()
        result = generator.createShapefile()
        
        print(f"District Performance Ranking GeoPackage created and uploaded to S3!")
        print(f"\nS3 Location: s3://{generator.bucketName}/{generator.s3ShapefilePrefix}/")
        print(f"\nPublic URL:")
        print(f"  GeoPackage: {result['mainUrl']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Districts: {summary['districts']}")
        print(f"  Best performer: District {summary['bestDistrict']} ({summary['bestMedianHours']:.2f} hrs median)")
        print(f"  Worst performer: District {summary['worstDistrict']} ({summary['worstMedianHours']:.2f} hrs median)")
        print(f"  (Ranking based on Median Response Time - robust to outliers)")
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Single file - no missing file issues!")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
