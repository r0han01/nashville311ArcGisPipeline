#!/usr/bin/env python3
"""
Service Efficiency GeoPackage Generator

Creates a GeoPackage ranking Nashville council districts by service efficiency
(fastest response relative to workload). Combines response time and workload
metrics to identify districts that respond quickly despite high workload.
All calculations are dynamic and adapt to the data.
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


class ServiceEfficiencyGenerator:
    """Generator for district service efficiency GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/serviceEfficiency'
        self.outputName = 'serviceEfficiency.gpkg'
        
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
    
    def calculateResponseTimeMetrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate response time metrics by district.
        
        Args:
            df: DataFrame with request data
            
        Returns:
            DataFrame with response time metrics
        """
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
        
        # Calculate city average response time (dynamic - based on actual data)
        cityAvgResponseHours = metrics['medianResponseHours'].mean() if len(metrics) > 0 else 0
        
        # Calculate response time ratio to city average (dynamic - 1.0 = average)
        metrics['response_time_ratio'] = (
            metrics['medianResponseHours'] / cityAvgResponseHours 
            if cityAvgResponseHours > 0 else 0
        )
        
        # Add city average to each row for reference
        metrics['city_avg_response_hours'] = cityAvgResponseHours
        
        return metrics
    
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
        area_cols = ['District_ID']
        if 'Area_Square_Miles' in boundaries.columns:
            area_cols.append('Area_Square_Miles')
        if 'Area_Square_Kilometers' in boundaries.columns:
            area_cols.append('Area_Square_Kilometers')
        
        areaData = boundaries[area_cols].copy()
        
        # Merge request counts with area data
        metrics = requestCounts.merge(areaData, on='District_ID', how='inner')
        
        # Ensure area columns exist
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
        
        cityAvgRequestsPerSqMi = totalRequests / totalAreaSqMi if totalAreaSqMi > 0 else 0
        
        # Calculate workload ratio to city average (dynamic - 1.0 = average)
        metrics['workload_ratio_sq_mi'] = (
            metrics['requests_per_sq_mile'] / cityAvgRequestsPerSqMi 
            if cityAvgRequestsPerSqMi > 0 else 0
        )
        
        # Add city average to each row for reference
        metrics['city_avg_requests_per_sq_mi'] = cityAvgRequestsPerSqMi
        
        return metrics
    
    def calculateEfficiencyMetrics(self, responseMetrics: pd.DataFrame, workloadMetrics: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate efficiency metrics by combining response time and workload.
        
        Efficiency = Workload Ratio / Response Time Ratio
        Higher = More Efficient (responds fast despite high workload)
        
        Args:
            responseMetrics: DataFrame with response time metrics
            workloadMetrics: DataFrame with workload metrics
            
        Returns:
            DataFrame with efficiency metrics
        """
        # Merge response time and workload metrics
        metrics = responseMetrics.merge(
            workloadMetrics[['District_ID', 'workload_ratio_sq_mi', 'city_avg_requests_per_sq_mi']],
            on='District_ID',
            how='inner'
        )
        
        # Calculate efficiency score (dynamic - adapts to data)
        # Efficiency = Workload Ratio / Response Time Ratio
        # Higher = More Efficient
        metrics['efficiency_score'] = (
            metrics['workload_ratio_sq_mi'] / metrics['response_time_ratio']
            if (metrics['response_time_ratio'] > 0).any() else 0
        )
        
        # Replace infinite values with 0 (when response_time_ratio is 0)
        metrics['efficiency_score'] = metrics['efficiency_score'].replace([float('inf'), float('-inf')], 0)
        
        # Calculate city average efficiency (dynamic - based on actual data)
        cityAvgEfficiency = metrics['efficiency_score'].mean() if len(metrics) > 0 else 1.0
        
        # Calculate efficiency ratio to city average (dynamic - 1.0 = average)
        metrics['efficiency_ratio'] = (
            metrics['efficiency_score'] / cityAvgEfficiency 
            if cityAvgEfficiency > 0 else 0
        )
        
        # Rank districts by efficiency (1 = most efficient, dynamic ranking)
        metrics = metrics.sort_values('efficiency_score', ascending=False).reset_index(drop=True)
        metrics['efficiency_rank'] = metrics.index + 1
        
        # Calculate efficiency percentile (dynamic - 0-100, relative position)
        n = len(metrics)
        if n > 1:
            metrics['efficiency_percentile'] = ((n - metrics['efficiency_rank']) / (n - 1) * 100).round().astype('Int64')
        else:
            metrics['efficiency_percentile'] = 100
        
        # Calculate efficiency quartiles (fully dynamic - adapts to data distribution)
        n = len(metrics)
        if n >= 4:
            # Use qcut for equal-sized quartiles
            metrics['efficiency_quartile'] = pd.qcut(
                metrics['efficiency_score'],
                q=4,
                labels=['Q1', 'Q2', 'Q3', 'Q4'],
                duplicates='drop'  # Handle cases where multiple districts have same efficiency
            )
        else:
            # If fewer than 4 districts, assign all to Q2 (middle quartile)
            metrics['efficiency_quartile'] = 'Q2'
        
        # Map quartiles to descriptive labels (dynamic - based on quartile assignment)
        quartileLabels = {
            'Q1': 'Top 25% Most Efficient',
            'Q2': 'Above Average Efficiency',
            'Q3': 'Below Average Efficiency',
            'Q4': 'Bottom 25% Least Efficient'
        }
        metrics['efficiency_quartile_label'] = metrics['efficiency_quartile'].map(quartileLabels)
        
        return metrics
    
    def loadDistrictBoundaries(self) -> gpd.GeoDataFrame:
        """
        Load and transform district boundaries from S3.
        Calculates area for each district polygon.
        """
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
        Create the service efficiency GeoPackage and upload to S3.
        
        Returns:
            Dictionary with S3 URLs and summary information
        """
        # Load data and boundaries
        df = self.loadDataFromS3(fileName)
        boundaries = self.loadDistrictBoundaries()
        
        # Calculate response time and workload metrics
        responseMetrics = self.calculateResponseTimeMetrics(df)
        workloadMetrics = self.calculateWorkloadMetrics(df, boundaries)
        
        # Calculate efficiency metrics (combines response time and workload)
        metrics = self.calculateEfficiencyMetrics(responseMetrics, workloadMetrics)
        
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
            'medianResponseHours', 'response_time_ratio', 'city_avg_response_hours',
            'requests_per_sq_mile', 'workload_ratio_sq_mi', 'city_avg_requests_per_sq_mi',
            'efficiency_score', 'efficiency_ratio', 'efficiency_rank', 
            'efficiency_percentile', 'efficiency_quartile', 'efficiency_quartile_label'
        ]
        
        for col in metric_cols:
            if col in gdf.columns:
                cols.append(col)
        
        # Get summary stats BEFORE renaming (using original column names)
        # Calculate from metrics which has the correct merged data
        total_requests = metrics['totalRequests'].sum() if 'totalRequests' in metrics.columns else 0
        city_avg_efficiency = metrics['efficiency_score'].mean() if len(metrics) > 0 else 0
        
        # Get highest and lowest efficiency districts (before renaming)
        if 'efficiency_rank' in metrics.columns:
            highest_efficiency = metrics[metrics['efficiency_rank'] == 1].iloc[0]
            lowest_efficiency = metrics[metrics['efficiency_rank'] == len(metrics)].iloc[0]
            highest_district_id = int(highest_efficiency['District_ID'])
            highest_efficiency_value = float(highest_efficiency['efficiency_score'])
            lowest_district_id = int(lowest_efficiency['District_ID'])
            lowest_efficiency_value = float(lowest_efficiency['efficiency_score'])
        else:
            highest_district_id = 0
            highest_efficiency_value = 0.0
            lowest_district_id = 0
            lowest_efficiency_value = 0.0
        
        # Select only existing columns and rename
        gdf = gdf[cols].rename(columns={
            'District_ID': 'District ID',
            'District_Name': 'District Name',
            'Representative_Name': 'Representative Name',
            'medianResponseHours': 'Median Response Time Hours',
            'response_time_ratio': 'Response Time Ratio To City Average',
            'city_avg_response_hours': 'City Average Response Time Hours',
            'requests_per_sq_mile': 'Requests Per Square Mile',
            'workload_ratio_sq_mi': 'Workload Ratio To City Average',
            'city_avg_requests_per_sq_mi': 'City Average Requests Per Square Mile',
            'efficiency_score': 'Efficiency Score',
            'efficiency_ratio': 'Efficiency Ratio To City Average',
            'efficiency_rank': 'Efficiency Rank',
            'efficiency_percentile': 'Efficiency Percentile',
            'efficiency_quartile': 'Efficiency Quartile',
            'efficiency_quartile_label': 'Efficiency Quartile Label'
        })
        
        # Create request points layer
        points_gdf = self.createRequestPointsLayer(df)
        
        # Create GeoPackage in temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            # Write polygon layer (District Service Efficiency)
            gdf.to_file(outputPath, driver='GPKG', layer='District Service Efficiency')
            
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
                'city_avg_efficiency': float(city_avg_efficiency),
                'highest_efficiency_district': highest_district_id,
                'highest_efficiency_value': highest_efficiency_value,
                'lowest_efficiency_district': lowest_district_id,
                'lowest_efficiency_value': lowest_efficiency_value
            }
        }


def main():
    """Main function to generate service efficiency GeoPackage."""
    try:
        generator = ServiceEfficiencyGenerator()
        result = generator.createShapefile()
        
        print(f"Service Efficiency GeoPackage created and uploaded to S3!")
        print(f"\nS3 Location: s3://{generator.bucketName}/{generator.s3ShapefilePrefix}/")
        print(f"\nPublic URL:")
        print(f"  GeoPackage: {result['mainUrl']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Districts: {summary['districts']}")
        print(f"  Request Points: {summary['request_points']:,}")
        print(f"  Total Requests: {summary['total_requests']:,}")
        print(f"  City Average Efficiency: {summary['city_avg_efficiency']:.2f}")
        print(f"\n  Most Efficient: District {summary['highest_efficiency_district']} (efficiency score: {summary['highest_efficiency_value']:.2f})")
        print(f"  Least Efficient: District {summary['lowest_efficiency_district']} (efficiency score: {summary['lowest_efficiency_value']:.2f})")
        print(f"  (Efficiency = Workload Ratio / Response Time Ratio - fully dynamic, adapts to data)")
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Service Efficiency' (polygons)")
        print(f"     2. 'Service Requests' (points - {summary['request_points']:,} requests)")
        print(f"   Use efficiency rank for color-coding districts")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

