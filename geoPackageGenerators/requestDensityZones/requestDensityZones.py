#!/usr/bin/env python3
"""
Request Density Zones Shapefile Generator

Creates a GeoPackage with hexagonal density zones showing request density
across Nashville. Uses pure metrics (rank, percentile) with no categorization.
All calculations are dynamic and adapt to the data.
"""

import sys
import os
import io
import boto3
import pandas as pd
import geopandas as gpd
import tempfile
import numpy as np
from datetime import datetime
from typing import Optional, Dict, Any
from shapely.geometry import Point, Polygon
from shapely.ops import transform
import pyproj

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


class RequestDensityZonesGenerator:
    """Generator for request density zones shapefiles."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.s3ShapefilePrefix = 'gpkg-public/requestDensityZones'
        self.outputName = 'requestDensityZones.gpkg'
        
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
    
    def createHexagon(self, x: float, y: float, radius: float) -> Polygon:
        """
        Create a single hexagon centered at (x, y) with given radius.
        
        Args:
            x: Center longitude
            y: Center latitude
            radius: Radius of hexagon (in degrees)
            
        Returns:
            Polygon representing the hexagon
        """
        # Hexagon vertices calculation
        angles = [i * np.pi / 3 for i in range(6)]
        vertices = []
        for angle in angles:
            vx = x + radius * np.cos(angle)
            vy = y + radius * np.sin(angle)
            vertices.append((vx, vy))
        
        # Close the polygon
        vertices.append(vertices[0])
        return Polygon(vertices)
    
    def createHexagonalGrid(self, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Create a hexagonal grid covering all request points.
        Grid size is calculated dynamically based on data extent and point count.
        
        Args:
            points_gdf: GeoDataFrame with request points
            
        Returns:
            GeoDataFrame with hexagonal zones
        """
        # Calculate data extent dynamically
        bounds = points_gdf.total_bounds  # minx, miny, maxx, maxy
        minx, miny, maxx, maxy = bounds
        
        extent_width = maxx - minx
        extent_height = maxy - miny
        extent_area = extent_width * extent_height
        
        # Calculate target number of zones dynamically
        total_points = len(points_gdf)
        # Adaptive: More points = more zones (smaller hexagons for better granularity)
        # Create more zones for better detail (aim for ~300-500 zones with data)
        target_zones = max(300, min(800, int(total_points / 100)))
        
        # Calculate optimal hexagon size (radius in degrees)
        # Hexagon area formula: area = (3 * sqrt(3) / 2) * radius^2
        # Approximate: in degrees, use simple approximation
        hex_area = extent_area / target_zones
        # Approximate radius (for small areas, degrees are roughly linear)
        # Use more accurate calculation for projected CRS later
        radius_approx = np.sqrt(hex_area / 2.598)  # 2.598 = 3*sqrt(3)/2
        
        # Transform to projected CRS for accurate hexagon creation
        points_proj = points_gdf.to_crs('EPSG:3857')  # Web Mercator
        bounds_proj = points_proj.total_bounds
        
        # Recalculate in meters
        extent_width_m = bounds_proj[2] - bounds_proj[0]
        extent_height_m = bounds_proj[3] - bounds_proj[1]
        extent_area_m2 = extent_width_m * extent_height_m
        
        hex_area_m2 = extent_area_m2 / target_zones
        radius_m = np.sqrt(hex_area_m2 / 2.598076211)
        
        # Create hexagonal grid in projected CRS
        hexagons = []
        hex_id = 1
        
        # Starting position (slightly outside bounds to ensure coverage)
        start_x = bounds_proj[0] - radius_m
        start_y = bounds_proj[1] - radius_m
        
        # Grid spacing
        hex_width = radius_m * np.sqrt(3)
        hex_height = radius_m * 1.5
        
        y = start_y
        while y < bounds_proj[3] + radius_m:
            x = start_x
            # Offset every other row
            offset = (hex_id % 2) * (hex_width / 2)
            x += offset
            
            while x < bounds_proj[2] + radius_m:
                # Create hexagon in projected CRS
                hex_poly = self.createHexagon(x, y, radius_m)
                
                # Check if hexagon intersects with any points (optional optimization)
                hexagons.append({
                    'Zone_ID': hex_id,
                    'geometry': hex_poly
                })
                
                hex_id += 1
                x += hex_width
            
            y += hex_height
        
        # Create GeoDataFrame
        hex_grid = gpd.GeoDataFrame(hexagons, crs='EPSG:3857')
        
        # Transform back to EPSG:4326
        hex_grid = hex_grid.to_crs('EPSG:4326')
        
        # Calculate area in projected CRS (more accurate)
        hex_grid_proj = hex_grid.to_crs('EPSG:3857')
        hex_grid_proj['area_sq_m'] = hex_grid_proj.geometry.area
        hex_grid_proj['Area_Square_Kilometers'] = hex_grid_proj['area_sq_m'] / self.SQ_M_TO_SQ_KM
        hex_grid_proj['Area_Square_Miles'] = hex_grid_proj['Area_Square_Kilometers'] * self.SQ_KM_TO_SQ_MI
        
        # Transform back to EPSG:4326 (keep area columns)
        hex_grid = hex_grid_proj.to_crs('EPSG:4326')
        
        # Drop intermediate column
        if 'area_sq_m' in hex_grid.columns:
            hex_grid = hex_grid.drop(columns=['area_sq_m'])
        
        return hex_grid
    
    def calculateDensityMetrics(self, hex_grid: gpd.GeoDataFrame, points_gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Calculate density metrics for each hexagon.
        Pure metrics: density, rank, percentile (no categories).
        
        Args:
            hex_grid: GeoDataFrame with hexagonal zones
            points_gdf: GeoDataFrame with request points
            
        Returns:
            GeoDataFrame with density metrics
        """
        # Spatial join: assign points to hexagons
        points_in_zones = gpd.sjoin(points_gdf, hex_grid, how='inner', predicate='within')
        
        # Count requests per zone
        request_counts = points_in_zones.groupby('Zone_ID').size().reset_index(name='totalRequests')
        
        # Merge with hex grid
        zones = hex_grid.merge(request_counts, on='Zone_ID', how='left').fillna({'totalRequests': 0})
        
        # Calculate density per zone (dynamic - adapts to data)
        zones['requests_per_sq_mile'] = zones['totalRequests'] / zones['Area_Square_Miles']
        zones['requests_per_sq_km'] = zones['totalRequests'] / zones['Area_Square_Kilometers']
        
        # Replace infinite values with 0 (if area is 0)
        zones['requests_per_sq_mile'] = zones['requests_per_sq_mile'].replace([float('inf'), float('-inf')], 0)
        zones['requests_per_sq_km'] = zones['requests_per_sq_km'].replace([float('inf'), float('-inf')], 0)
        
        # Rank zones by density (1 = highest density, fully dynamic ranking)
        zones = zones.sort_values('requests_per_sq_mile', ascending=False).reset_index(drop=True)
        zones['density_rank'] = zones.index + 1
        
        # Calculate density percentile (0-100, relative position, fully dynamic)
        n = len(zones)
        if n > 1:
            zones['density_percentile'] = ((n - zones['density_rank']) / (n - 1) * 100).round().astype('Int64')
        else:
            zones['density_percentile'] = 100
        
        return zones
    
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
        Create the request density zones GeoPackage and upload to S3.
        
        Returns:
            Dictionary with S3 URLs and summary information
        """
        # Load data
        df = self.loadDataFromS3(fileName)
        
        # Create request points layer
        points_gdf = self.createRequestPointsLayer(df)
        
        # Create hexagonal grid dynamically
        hex_grid = self.createHexagonalGrid(points_gdf)
        
        # Calculate density metrics (pure metrics - no categories)
        zones = self.calculateDensityMetrics(hex_grid, points_gdf)
        
        # Filter out empty zones (only keep zones with requests)
        # This makes the visualization more meaningful
        zones_with_requests = zones[zones['totalRequests'] > 0].copy()
        
        # Recalculate rank and percentile after filtering (relative to zones with data)
        zones_with_requests = zones_with_requests.sort_values('requests_per_sq_mile', ascending=False).reset_index(drop=True)
        zones_with_requests['density_rank'] = zones_with_requests.index + 1
        
        # Recalculate percentile (0-100, relative to zones with data)
        n = len(zones_with_requests)
        if n > 1:
            zones_with_requests['density_percentile'] = ((n - zones_with_requests['density_rank']) / (n - 1) * 100).round().astype('Int64')
        else:
            zones_with_requests['density_percentile'] = 100
        
        # Select and rename columns with descriptive names
        cols = [
            'Zone_ID', 'totalRequests', 'Area_Square_Miles', 'Area_Square_Kilometers',
            'requests_per_sq_mile', 'requests_per_sq_km',
            'density_rank', 'density_percentile', 'geometry'
        ]
        
        zones_final = zones_with_requests[cols].rename(columns={
            'Zone_ID': 'Zone ID',
            'totalRequests': 'Total Requests',
            'Area_Square_Miles': 'Zone Area Square Miles',
            'Area_Square_Kilometers': 'Zone Area Square Kilometers',
            'requests_per_sq_mile': 'Requests Per Square Mile',
            'requests_per_sq_km': 'Requests Per Square Kilometer',
            'density_rank': 'Density Rank',
            'density_percentile': 'Density Percentile'
        })
        
        # Get summary stats before writing
        total_zones = len(zones_final)
        total_requests = zones_final['Total Requests'].sum()
        
        # Get highest and lowest density zones
        if len(zones_final) > 0:
            highest_density = zones_final[zones_final['Density Rank'] == 1].iloc[0]
            lowest_density = zones_final[zones_final['Density Rank'] == len(zones_final)].iloc[0]
            highest_density_value = float(highest_density['Requests Per Square Mile'])
            lowest_density_value = float(lowest_density['Requests Per Square Mile'])
        else:
            highest_density_value = 0.0
            lowest_density_value = 0.0
        
        # Create GeoPackage in temporary directory
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            # Write polygon layer (Request Density Zones)
            zones_final.to_file(outputPath, driver='GPKG', layer='Request Density Zones')
            
            # Append point layer (Service Requests) to the same GeoPackage
            points_gdf.to_file(outputPath, driver='GPKG', layer='Service Requests', mode='a')
            
            # Upload to S3
            s3Urls = self.uploadGeoPackageToS3(outputPath)
        
        return {
            's3Urls': s3Urls,
            'mainUrl': s3Urls.get('mainUrl', ''),
            'summary': {
                'total_zones': total_zones,
                'zones_with_requests': zones_with_requests,
                'request_points': len(points_gdf),
                'total_requests': int(total_requests),
                'highest_density_value': highest_density_value,
                'lowest_density_value': lowest_density_value
            }
        }


def main():
    """Main function to generate request density zones GeoPackage."""
    try:
        generator = RequestDensityZonesGenerator()
        result = generator.createShapefile()
        
        print(f"Request Density Zones GeoPackage created and uploaded to S3!")
        print(f"\nS3 Location: s3://{generator.bucketName}/{generator.s3ShapefilePrefix}/")
        print(f"\nPublic URL:")
        print(f"  GeoPackage: {result['mainUrl']}")
        
        print(f"\nSummary:")
        summary = result['summary']
        print(f"  Zones with Requests: {summary['total_zones']} (empty zones filtered out)")
        print(f"  Request Points: {summary['request_points']:,}")
        print(f"  Total Requests: {summary['total_requests']:,}")
        print(f"\n  Highest Density: {summary['highest_density_value']:.2f} requests/sq mile (Rank 1)")
        print(f"  Lowest Density: {summary['lowest_density_value']:.2f} requests/sq mile (Rank {summary['total_zones']})")
        print(f"  (Pure metrics approach - only zones with requests included, no empty zones)")
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'Request Density Zones' (polygons - pure metrics)")
        print(f"     2. 'Service Requests' (points - {summary['request_points']:,} requests)")
        print(f"   Use density rank or percentile for color-coding zones")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

