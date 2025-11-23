#!/usr/bin/env python3
"""
Temporal Patterns GeoPackage Generator

Creates a GeoPackage analyzing weekday vs weekend request patterns across
Nashville council districts. Focuses on WHEN requests happen (weekdays vs weekends).

All calculations are data-driven with no hardcoded thresholds.
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
from shapely.geometry import Point

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


def calculate_silhouette_score_1d(data, k):
    """Calculate silhouette score for 1D data using k-means-like clustering."""
    data = np.array([x for x in data if not pd.isna(x)])
    
    if len(data) < k or len(data) < 2:
        return -1.0
    
    # Use quantiles to create initial clusters
    quantiles = np.linspace(0, 100, k + 1)
    breaks = np.percentile(data, quantiles)
    
    # Assign clusters based on breaks
    labels = np.digitize(data, breaks[1:], right=False)
    labels = np.clip(labels, 0, k - 1)
    
    # Calculate silhouette score
    if len(np.unique(labels)) < 2:
        return -1.0
    
    # For 1D data, use distance to cluster centers
    silhouette_scores = []
    for i, value in enumerate(data):
        cluster = labels[i]
        cluster_values = data[labels == cluster]
        
        # Average distance to points in same cluster (a_i)
        if len(cluster_values) > 1:
            a_i = np.mean(np.abs(cluster_values - value))
        else:
            a_i = 0.0
        
        # Average distance to points in nearest other cluster (b_i)
        b_i = float('inf')
        for other_cluster in range(k):
            if other_cluster != cluster:
                other_cluster_values = data[labels == other_cluster]
                if len(other_cluster_values) > 0:
                    avg_dist = np.mean(np.abs(other_cluster_values - value))
                    b_i = min(b_i, avg_dist)
        
        if max(a_i, b_i) > 0:
            s_i = (b_i - a_i) / max(a_i, b_i)
        else:
            s_i = 0.0
        
        silhouette_scores.append(s_i)
    
    return np.mean(silhouette_scores) if silhouette_scores else -1.0


def calculate_data_driven_k_range(n):
    """
    Calculate k_range based on sample size (fully data-driven).
    
    Formula: k_max = min(5, max(3, int(sqrt(n)), int(n/7)))
    """
    k_min = 3
    k_max = min(5, max(3, int(np.sqrt(n)), int(n / 7)))
    k_max = max(k_min, k_max)
    return list(range(k_min, k_max + 1))


def find_optimal_k_silhouette(data, k_range=None):
    """Find optimal number of classes using silhouette score."""
    data = np.array([x for x in data if not pd.isna(x)])
    n = len(data)
    
    if k_range is None:
        k_range = calculate_data_driven_k_range(n)
    
    if n < min(k_range):
        return min(k_range)
    
    scores = {}
    for k in k_range:
        if n >= k:
            score = calculate_silhouette_score_1d(data, k)
            scores[k] = score
    
    if not scores:
        return min(k_range)
    
    optimal_k = max(scores, key=scores.get)
    
    # If scores are very close (within 0.05), prefer k=4 for interpretability
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        if abs(best_score - second_score) < 0.05:
            if 4 in scores:
                return 4
            k_values = sorted(scores.keys())
            return k_values[len(k_values) // 2]
    
    return optimal_k


class TemporalPatternsGenerator:
    """Generator for district temporal patterns GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/temporalPatterns'
        self.outputName = 'temporalPatterns.gpkg'
    
    def loadDataFromS3(self, fileName: Optional[str] = None) -> pd.DataFrame:
        """Load latest Parquet data from S3."""
        if fileName is None:
            response = self.s3Client.list_objects_v2(
                Bucket=self.bucketName, 
                Prefix='processed-data/'
            )
            
            if 'Contents' not in response or not response['Contents']:
                raise ValueError("No parquet files found in processed-data/")
            
            latest = max(response['Contents'], key=lambda x: x['LastModified'])
            fileName = latest['Key'].split('/')[-1]
        
        response = self.s3Client.get_object(
            Bucket=self.bucketName,
            Key=f'processed-data/{fileName}'
        )
        
        return pd.read_parquet(io.BytesIO(response['Body'].read()))
    
    def loadDistrictBoundaries(self) -> gpd.GeoDataFrame:
        """Load and transform district boundaries from S3."""
        with tempfile.TemporaryDirectory() as tempDir:
            shapefileExtensions = ['.shp', '.shx', '.dbf', '.prj', '.cpg', '.shp.xml']
            baseName = '2022_Council_Districts'
            
            for ext in shapefileExtensions:
                s3Key = f'{self.boundaryS3Key.replace(".shp", ext)}'
                localPath = os.path.join(tempDir, f'{baseName}{ext}')
                try:
                    self.s3Client.download_file(self.bucketName, s3Key, localPath)
                except Exception as e:
                    if ext in ['.cpg', '.shp.xml']:
                        continue
                    raise
            
            boundaries = gpd.read_file(os.path.join(tempDir, f'{baseName}.shp'))
            
            # Standardize district ID and name columns
            districtIdCol = None
            districtNameCol = None
            repNameCol = None
            
            for col in boundaries.columns:
                colLower = col.lower()
                if 'district' in colLower and ('id' in colLower or 'num' in colLower):
                    districtIdCol = col
                elif 'district' in colLower and 'name' in colLower:
                    districtNameCol = col
                elif 'represent' in colLower or 'council' in colLower:
                    repNameCol = col
            
            if districtIdCol:
                boundaries['District_ID'] = boundaries[districtIdCol]
            else:
                boundaries['District_ID'] = range(1, len(boundaries) + 1)
            
            if districtNameCol:
                boundaries['District_Name'] = boundaries[districtNameCol]
            else:
                boundaries['District_Name'] = boundaries['District_ID'].astype(str)
            
            if repNameCol:
                boundaries['Representative_Name'] = boundaries[repNameCol]
            else:
                boundaries['Representative_Name'] = ''
            
            return boundaries[['District_ID', 'District_Name', 'Representative_Name', 'geometry']]
    
    def calculateTemporalMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate weekday/weekend temporal metrics by district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract day of week (0=Monday, 6=Sunday)
        df['dayOfWeek'] = df['openedDt'].dt.dayofweek
        df['dayName'] = df['openedDt'].dt.day_name()
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        temporal_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id]
            
            if len(district_data) == 0:
                temporal_metrics.append({
                    'District_ID': district_id,
                    'totalRequests': 0,
                    'weekdayRequestCount': 0,
                    'weekendRequestCount': 0,
                    'mondayCount': 0,
                    'tuesdayCount': 0,
                    'wednesdayCount': 0,
                    'thursdayCount': 0,
                    'fridayCount': 0,
                    'saturdayCount': 0,
                    'sundayCount': 0
                })
                continue
            
            # Count by day of week
            day_counts = district_data['dayName'].value_counts()
            
            mondayCount = day_counts.get('Monday', 0)
            tuesdayCount = day_counts.get('Tuesday', 0)
            wednesdayCount = day_counts.get('Wednesday', 0)
            thursdayCount = day_counts.get('Thursday', 0)
            fridayCount = day_counts.get('Friday', 0)
            saturdayCount = day_counts.get('Saturday', 0)
            sundayCount = day_counts.get('Sunday', 0)
            
            # Weekday vs weekend
            weekdayCount = mondayCount + tuesdayCount + wednesdayCount + thursdayCount + fridayCount
            weekendCount = saturdayCount + sundayCount
            totalCount = len(district_data)
            
            temporal_metrics.append({
                'District_ID': district_id,
                'totalRequests': totalCount,
                'weekdayRequestCount': weekdayCount,
                'weekendRequestCount': weekendCount,
                'mondayCount': mondayCount,
                'tuesdayCount': tuesdayCount,
                'wednesdayCount': wednesdayCount,
                'thursdayCount': thursdayCount,
                'fridayCount': fridayCount,
                'saturdayCount': saturdayCount,
                'sundayCount': sundayCount
            })
        
        metrics = pd.DataFrame(temporal_metrics)
        
        # Calculate percentages and ratios
        metrics['weekdayPercent'] = (metrics['weekdayRequestCount'] / metrics['totalRequests'] * 100).round(2)
        metrics['weekendPercent'] = (metrics['weekendRequestCount'] / metrics['totalRequests'] * 100).round(2)
        metrics['weekdayWeekendRatio'] = (metrics['weekdayRequestCount'] / metrics['weekendRequestCount']).replace([np.inf, -np.inf], np.nan).round(2)
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('weekdayPercent', ascending=False).reset_index(drop=True)
            metrics['temporalPatternPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['temporalPatternQuartile'] = pd.qcut(
                    metrics['weekdayPercent'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = highest weekday percent
                    duplicates='drop'
                )
            else:
                metrics['temporalPatternQuartile'] = 'Q2'
        else:
            metrics['temporalPatternPercentile'] = 100
            metrics['temporalPatternQuartile'] = 'Q1'
        
        # Pattern type using quartiles (data-driven)
        weekday_percent_values = metrics['weekdayPercent'].dropna().values
        if len(weekday_percent_values) >= 3:
            Q1, Q2, Q3 = np.percentile(weekday_percent_values, [25, 50, 75])
            
            def classify_pattern(value):
                if pd.isna(value):
                    return 'Unknown'
                if value >= Q3:
                    return 'Weekday Dominant'
                elif value <= Q1:
                    return 'Weekend Dominant'
                else:
                    return 'Balanced'
            
            metrics['temporalPatternType'] = metrics['weekdayPercent'].apply(classify_pattern)
        else:
            metrics['temporalPatternType'] = 'Balanced'
        
        # Calculate consistency (coefficient of variation across 7 days)
        consistency_metrics = []
        for idx, row in metrics.iterrows():
            day_counts = [
                row['mondayCount'],
                row['tuesdayCount'],
                row['wednesdayCount'],
                row['thursdayCount'],
                row['fridayCount'],
                row['saturdayCount'],
                row['sundayCount']
            ]
            
            if sum(day_counts) > 0:
                mean_count = np.mean(day_counts)
                std_count = np.std(day_counts)
                if mean_count > 0:
                    cv = std_count / mean_count
                    consistency = 1 - cv  # Higher = more consistent
                    consistency = max(0, min(1, consistency))  # Clamp to 0-1
                else:
                    consistency = 1.0
            else:
                consistency = 1.0
            
            consistency_metrics.append(consistency)
        
        metrics['weekdayWeekendConsistency'] = consistency_metrics
        
        # Calculate consistency percentile and quartile
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('weekdayWeekendConsistency', ascending=False).reset_index(drop=True)
            metrics['consistencyPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['consistencyQuartile'] = pd.qcut(
                    metrics['weekdayWeekendConsistency'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = most consistent
                    duplicates='drop'
                )
            else:
                metrics['consistencyQuartile'] = 'Q2'
        else:
            metrics['consistencyPercentile'] = 100
            metrics['consistencyQuartile'] = 'Q1'
        
        # Stability category using Silhouette + Quantiles (data-driven)
        consistency_values = metrics['weekdayWeekendConsistency'].dropna().values
        if len(consistency_values) >= 3:
            optimal_k = find_optimal_k_silhouette(consistency_values, k_range=None)
            
            # For consistency, higher values = better, so we reverse the order
            sorted_desc = np.sort(consistency_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]
            
            # Create labels
            labels = []
            for i in range(optimal_k):
                if optimal_k == 3:
                    if i == 0:
                        labels.append('Top 33% Consistency')
                    elif i == 1:
                        labels.append('Middle 33% Consistency')
                    else:
                        labels.append('Bottom 33% Consistency')
                elif optimal_k == 4:
                    if i == 0:
                        labels.append('Top 25% Consistency')
                    elif i == 1:
                        labels.append('Upper-Middle 25% Consistency')
                    elif i == 2:
                        labels.append('Lower-Middle 25% Consistency')
                    else:
                        labels.append('Bottom 25% Consistency')
                else:  # k == 5
                    if i == 0:
                        labels.append('Top 20% Consistency')
                    elif i == optimal_k - 1:
                        labels.append('Bottom 20% Consistency')
                    else:
                        labels.append(f'Middle-{i} Consistency')
            
            # Create classification function
            def classify_consistency(value):
                if pd.isna(value):
                    return 'Unknown'
                for i in range(optimal_k):
                    if i == optimal_k - 1:
                        if value <= breaks[i]:
                            return labels[i]
                    else:
                        if value <= breaks[i] and value > breaks[i + 1]:
                            return labels[i]
                return labels[0]
            
            metrics['temporalStabilityCategory'] = metrics['weekdayWeekendConsistency'].apply(classify_consistency)
        else:
            metrics['temporalStabilityCategory'] = 'Moderate Consistency'
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with temporal metrics."""
        points_df = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        
        # Convert timestamp to datetime
        points_df['openedDt'] = pd.to_datetime(points_df['Date_Time_Opened'], unit='ms', errors='coerce')
        points_df = points_df[points_df['openedDt'].notna()].copy()
        
        # Extract day of week
        points_df['dayOfWeek'] = points_df['openedDt'].dt.dayofweek
        points_df['dayName'] = points_df['openedDt'].dt.day_name()
        
        # Create geometry
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df['Longitude'], points_df['Latitude']),
            crs='EPSG:4326'
        )
        
        # Convert district to numeric
        points_gdf['districtInt'] = pd.to_numeric(points_gdf['Council_District'], errors='coerce').astype('Int64')
        
        # Select and rename columns to camelCase (preserve geometry)
        columns_to_keep = ['geometry', 'districtInt', 'Request__', 'Request_Type', 'Status', 'dayName']
        points_gdf = points_gdf[columns_to_keep].copy()
        points_gdf = points_gdf.rename(columns={
            'districtInt': 'districtId',
            'Request__': 'requestId',
            'Request_Type': 'requestType',
            'Status': 'status',
            'dayName': 'dayOfWeek'
        })
        
        # Ensure it's still a GeoDataFrame
        points_gdf = gpd.GeoDataFrame(points_gdf, geometry='geometry', crs='EPSG:4326')
        
        # Merge district metrics if provided
        if metrics is not None:
            metrics_merge = metrics[['District_ID', 'temporalPatternType', 'temporalPatternPercentile', 
                                    'temporalPatternQuartile', 'weekdayPercent', 'weekendPercent']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'temporalPatternType': 'districtTemporalPatternType',
                'temporalPatternPercentile': 'districtTemporalPatternPercentile',
                'temporalPatternQuartile': 'districtTemporalPatternQuartile',
                'weekdayPercent': 'districtWeekdayPercent',
                'weekendPercent': 'districtWeekendPercent'
            })
            
            # Preserve geometry during merge
            geometry = points_gdf.geometry
            points_gdf = points_gdf.merge(metrics_merge, on='districtId', how='left')
            points_gdf = gpd.GeoDataFrame(points_gdf, geometry=geometry, crs='EPSG:4326')
        
        return points_gdf
    
    def uploadGeoPackageToS3(self, localPath: str) -> str:
        """Upload GeoPackage to S3."""
        s3Key = f'{self.s3ShapefilePrefix}/{self.outputName}'
        self.s3Client.upload_file(localPath, self.bucketName, s3Key)
        return f'https://{self.bucketName}.s3.amazonaws.com/{s3Key}'
    
    def createShapefile(self):
        """Orchestrate the entire GeoPackage creation process."""
        print("Loading data from S3...")
        df = self.loadDataFromS3()
        print(f"Loaded {len(df):,} service requests")
        
        print("Loading district boundaries...")
        boundaries = self.loadDistrictBoundaries()
        print(f"Loaded {len(boundaries)} districts")
        
        print("Calculating temporal metrics...")
        metrics = self.calculateTemporalMetrics(df, boundaries)
        
        # Merge metrics with boundaries
        boundaries = boundaries.merge(metrics, on='District_ID', how='left')
        
        # Convert to camelCase
        boundaries = boundaries.rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName'
        })
        
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, metrics)
        
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkgPath = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            boundaries.to_file(gpkgPath, layer='District Temporal Patterns', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Temporal Patterns GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Temporal Patterns' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = TemporalPatternsGenerator()
    generator.createShapefile()

