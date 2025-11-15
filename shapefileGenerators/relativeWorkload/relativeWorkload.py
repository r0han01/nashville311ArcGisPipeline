#!/usr/bin/env python3
"""
Relative Workload Shapefile Generator

Creates a GeoPackage ranking Nashville council districts by relative workload
(requests per square mile/km). Uses Parquet data from S3 and district boundaries 
to generate polygon-based workload analysis. All calculations are dynamic and 
adapt to the data.
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


class RelativeWorkloadGenerator:
    """Generator for district relative workload shapefiles."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/relativeWorkload'
        self.outputName = 'relativeWorkload.gpkg'
        
        # Conversion constants (standard conversion factors - not hardcoded thresholds)
        self.SQ_M_TO_SQ_KM = 1_000_000
        self.SQ_KM_TO_SQ_MI = 0.386102
        
    def loadDataFromS3(self, fileName: Optional[str] = None) -> pd.DataFrame:
        """Load latest Parquet data from S3."""
        if fileName is None:
            # Find latest parquet file dynamically
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
    
    def calculateWorkloadMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """
        Calculate workload metrics by district.
        
        Args:
            df: DataFrame with request data
            boundaries: GeoDataFrame with district boundaries (includes area calculations)
            
        Returns:
            DataFrame with workload metrics
        """
        # Ensure required columns exist
        requiredCols = ['Council_District']
        for col in requiredCols:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Count requests per district
        requestCounts = df.groupby('districtInt', dropna=True).size().rename('totalRequests').to_frame()
        requestCounts = requestCounts.reset_index().rename(columns={'districtInt': 'District_ID'})
        
        # Get area data from boundaries (already calculated in loadDistrictBoundaries)
        # Check which area columns exist
        area_cols = ['District_ID']
        if 'Area_Square_Miles' in boundaries.columns:
            area_cols.append('Area_Square_Miles')
        if 'Area_Square_Kilometers' in boundaries.columns:
            area_cols.append('Area_Square_Kilometers')
        
        areaData = boundaries[area_cols].copy()
        
        # Merge request counts with area data
        metrics = requestCounts.merge(areaData, on='District_ID', how='inner')
        
        # Ensure area columns exist (in case they weren't in boundaries)
        if 'Area_Square_Miles' not in metrics.columns:
            raise ValueError("Area_Square_Miles column not found in boundaries after calculation")
        if 'Area_Square_Kilometers' not in metrics.columns:
            raise ValueError("Area_Square_Kilometers column not found in boundaries after calculation")
        
        # Calculate requests per square mile/km (dynamic - adapts to data)
        metrics['requests_per_sq_mile'] = metrics['totalRequests'] / metrics['Area_Square_Miles']
        metrics['requests_per_sq_km'] = metrics['totalRequests'] / metrics['Area_Square_Kilometers']
        
        # Calculate city average workload (dynamic - based on actual data)
        totalRequests = metrics['totalRequests'].sum()
        totalAreaSqMi = metrics['Area_Square_Miles'].sum()
        totalAreaSqKm = metrics['Area_Square_Kilometers'].sum()
        
        cityAvgRequestsPerSqMi = totalRequests / totalAreaSqMi if totalAreaSqMi > 0 else 0
        cityAvgRequestsPerSqKm = totalRequests / totalAreaSqKm if totalAreaSqKm > 0 else 0
        
        # Calculate ratio to city average (dynamic - 1.0 = average)
        metrics['workload_ratio_sq_mi'] = (
            metrics['requests_per_sq_mile'] / cityAvgRequestsPerSqMi 
            if cityAvgRequestsPerSqMi > 0 else 0
        )
        metrics['workload_ratio_sq_km'] = (
            metrics['requests_per_sq_km'] / cityAvgRequestsPerSqKm 
            if cityAvgRequestsPerSqKm > 0 else 0
        )
        
        # Add city average to each row for reference
        metrics['city_avg_requests_per_sq_mi'] = cityAvgRequestsPerSqMi
        metrics['city_avg_requests_per_sq_km'] = cityAvgRequestsPerSqKm
        
        # Rank districts by workload (1 = highest workload, dynamic ranking)
        metrics = metrics.sort_values('requests_per_sq_mile', ascending=False).reset_index(drop=True)
        metrics['workload_rank'] = metrics.index + 1
        
        return metrics
    
    def loadDistrictBoundaries(self) -> gpd.GeoDataFrame:
        """
        Load and transform district boundaries from S3.
        Calculates area for each district polygon.
        """
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
            # Create district name from ID dynamically
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
        
        # Transform CRS to EPSG:4326 first (WGS84)
        if boundaries.crs is None:
            boundaries = boundaries.set_crs('EPSG:2274', allow_override=True)
        boundaries = boundaries.to_crs('EPSG:4326')
        
        # Calculate area: Transform to projected CRS for accurate area calculation
        # Use Web Mercator (EPSG:3857) which measures in meters
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        
        # Calculate area in square meters (dynamic - from actual polygon shapes)
        boundaries_proj['area_sq_m'] = boundaries_proj.geometry.area
        
        # Convert to square kilometers and square miles (using standard conversion factors)
        boundaries_proj['Area_Square_Kilometers'] = boundaries_proj['area_sq_m'] / self.SQ_M_TO_SQ_KM
        boundaries_proj['Area_Square_Miles'] = boundaries_proj['Area_Square_Kilometers'] * self.SQ_KM_TO_SQ_MI
        
        # Transform back to EPSG:4326 for output (keep area columns)
        boundaries = boundaries_proj.to_crs('EPSG:4326')
        
        # Drop intermediate area_sq_m column (we have sq_km and sq_mi)
        if 'area_sq_m' in boundaries.columns:
            boundaries = boundaries.drop(columns=['area_sq_m'])
        
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
        
        # Rename columns to Title Case for better readability in ArcGIS Pro
        rename_map = {
            'Request__': 'Request ID',
            'Request_Type': 'Request Type',
            'Subrequest_Type': 'Subrequest Type',
            'Status': 'Status',
            'Address': 'Address',
            'City': 'City',
            'Council_District': 'Council District',
            'ZIP': 'ZIP Code',
            'Date_Time_Opened': 'Date Time Opened',
            'Date_Time_Closed': 'Date Time Closed',
            'Response_Hours': 'Response Time Hours'
        }
        
        # Only rename columns that exist
        rename_map = {k: v for k, v in rename_map.items() if k in points_gdf.columns}
        points_gdf = points_gdf.rename(columns=rename_map)
        
        # Convert timestamp columns to readable dates if they exist
        if 'Date Time Opened' in points_gdf.columns:
            points_gdf['Date Time Opened'] = pd.to_datetime(
                points_gdf['Date Time Opened'], unit='ms', errors='coerce'
            )
        if 'Date Time Closed' in points_gdf.columns:
            points_gdf['Date Time Closed'] = pd.to_datetime(
                points_gdf['Date Time Closed'], unit='ms', errors='coerce'
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
        Create the relative workload GeoPackage and upload to S3.
        
        Returns:
            Dictionary with S3 URLs and summary information
        """
        # Load data and boundaries
        df = self.loadDataFromS3(fileName)
        boundaries = self.loadDistrictBoundaries()
        
        # Calculate workload metrics
        metrics = self.calculateWorkloadMetrics(df, boundaries)
        
        # Merge boundaries with metrics
        gdf = boundaries.merge(metrics, on='District_ID', how='inner')
        
        # Select and rename columns with descriptive, readable names
        # Build column list dynamically based on what exists
        cols = ['District_ID', 'geometry']
        
        # Add district name if it exists
        if 'District_Name' in gdf.columns:
            cols.append('District_Name')
        
        # Add representative name if it exists
        if 'Representative_Name' in gdf.columns:
            cols.append('Representative_Name')
        
        # Add metrics columns (check if they exist)
        metric_cols = [
            'totalRequests', 'Area_Square_Miles', 'Area_Square_Kilometers',
            'requests_per_sq_mile', 'requests_per_sq_km',
            'city_avg_requests_per_sq_mi', 'city_avg_requests_per_sq_km',
            'workload_ratio_sq_mi', 'workload_ratio_sq_km', 'workload_rank'
        ]
        
        for col in metric_cols:
            if col in gdf.columns:
                cols.append(col)
        
        # Get summary stats BEFORE renaming (using original column names)
        # Calculate from metrics which has the correct merged data
        total_requests = metrics['totalRequests'].sum()
        total_area_sq_mi = metrics['Area_Square_Miles'].sum()
        city_avg = metrics['city_avg_requests_per_sq_mi'].iloc[0] if len(metrics) > 0 else 0
        
        # Get highest and lowest workload districts (before renaming)
        if 'workload_rank' in gdf.columns:
            highest_workload = gdf[gdf['workload_rank'] == 1].iloc[0]
            lowest_workload = gdf[gdf['workload_rank'] == len(gdf)].iloc[0]
            highest_district_id = int(highest_workload['District_ID'])
            highest_workload_value = float(highest_workload['requests_per_sq_mile'])
            lowest_district_id = int(lowest_workload['District_ID'])
            lowest_workload_value = float(lowest_workload['requests_per_sq_mile'])
        else:
            highest_district_id = 0
            highest_workload_value = 0.0
            lowest_district_id = 0
            lowest_workload_value = 0.0
        
        # Select only existing columns and rename
        gdf = gdf[cols].rename(columns={
            'District_ID': 'District ID',
            'District_Name': 'District Name',
            'Representative_Name': 'Representative Name',
            'totalRequests': 'Total Requests',
            'Area_Square_Miles': 'District Area Square Miles',
            'Area_Square_Kilometers': 'District Area Square Kilometers',
            'requests_per_sq_mile': 'Requests Per Square Mile',
            'requests_per_sq_km': 'Requests Per Square Kilometer',
            'city_avg_requests_per_sq_mi': 'City Average Requests Per Square Mile',
            'city_avg_requests_per_sq_km': 'City Average Requests Per Square Kilometer',
            'workload_ratio_sq_mi': 'Workload Ratio To City Average',
            'workload_ratio_sq_km': 'Workload Ratio To City Average Sq Km',
            'workload_rank': 'Workload Rank'
        })
        
        # Create request points layer
        points_gdf = self.createRequestPointsLayer(df)
        
        # Create GeoPackage in temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            # Write polygon layer (District Relative Workload)
            gdf.to_file(outputPath, driver='GPKG', layer='District Relative Workload')
            
            # Append point layer (Service Requests) to the same GeoPackage
            points_gdf.to_file(outputPath, driver='GPKG', layer='Service Requests', mode='a')
            
            # Upload to S3
            s3Urls = self.uploadGeoPackageToS3(outputPath)
        
        return {
            's3Urls': s3Urls,
            'mainUrl': s3Urls.get('mainUrl', ''),
            'summary': {
                'districts': len(gdf),
                'request_points': len(points_gdf),
                'total_requests': int(total_requests),
                'total_area_sq_mi': float(total_area_sq_mi),
                'city_avg_requests_per_sq_mi': float(city_avg),
                'highest_workload_district': highest_district_id,
                'highest_workload_value': highest_workload_value,
                'lowest_workload_district': lowest_district_id,
                'lowest_workload_value': lowest_workload_value
            }
        }


def main():
    """Main function to generate relative workload GeoPackage."""
    try:
        generator = RelativeWorkloadGenerator()
        result = generator.createShapefile()
        
        print(f"Relative Workload GeoPackage created and uploaded to S3!")
        print(f"\nS3 Location: s3://{generator.bucketName}/{generator.s3ShapefilePrefix}/")
        print(f"\nPublic URL:")
        print(f"  GeoPackage: {result['mainUrl']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Districts: {summary['districts']}")
        print(f"  Request Points: {summary['request_points']:,}")
        print(f"  Total Requests: {summary['total_requests']:,}")
        print(f"  Total Area: {summary['total_area_sq_mi']:.2f} sq miles")
        print(f"  City Average: {summary['city_avg_requests_per_sq_mi']:.2f} requests/sq mile")
        print(f"\n  Highest Workload: District {summary['highest_workload_district']} ({summary['highest_workload_value']:.2f} requests/sq mile)")
        print(f"  Lowest Workload: District {summary['lowest_workload_district']} ({summary['lowest_workload_value']:.2f} requests/sq mile)")
        print(f"  (Workload based on requests per square mile - fully dynamic, adapts to data)")
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Relative Workload' (polygons)")
        print(f"     2. 'Service Requests' (points - {summary['request_points']:,} requests)")
        print(f"   Use workload rank for color-coding districts")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

