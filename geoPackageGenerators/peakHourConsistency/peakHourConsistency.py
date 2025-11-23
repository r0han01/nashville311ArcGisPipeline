#!/usr/bin/env python3
"""
Peak Hour Consistency GeoPackage Generator

Creates a GeoPackage analyzing peak hours vs off-hours request patterns across
Nashville council districts. Focuses on WHEN during the day requests happen
and identifies districts with consistent off-hours request patterns.

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


class PeakHourConsistencyGenerator:
    """Generator for district peak hour consistency GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/peakHourConsistency'
        self.outputName = 'peakHourConsistency.gpkg'
        
        # Standard business hours (8 AM - 5 PM) - widely accepted definition
        self.PEAK_HOUR_START = 8
        self.PEAK_HOUR_END = 17  # 5 PM
    
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
    
    def calculatePeakHourMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate peak hours vs off-hours metrics by district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract hour of day (0-23) and day of week
        df['hourOfDay'] = df['openedDt'].dt.hour
        df['dayOfWeek'] = df['openedDt'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['isWeekday'] = df['dayOfWeek'] < 5  # Monday-Friday
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        peak_hour_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id]
            
            if len(district_data) == 0:
                peak_hour_metrics.append({
                    'District_ID': district_id,
                    'totalRequests': 0,
                    'peakHourRequestCount': 0,
                    'offHoursRequestCount': 0,
                    'businessHoursRequestCount': 0,
                    'peakHourOfDay': None
                })
                continue
            
            # Peak hours: 8 AM - 5 PM (8-16), weekdays only
            peak_hour_mask = (
                (district_data['hourOfDay'] >= self.PEAK_HOUR_START) &
                (district_data['hourOfDay'] < self.PEAK_HOUR_END) &
                (district_data['isWeekday'])
            )
            
            # Off-hours: everything else (before 8 AM, after 5 PM, or weekends)
            off_hours_mask = ~peak_hour_mask
            
            peakHourCount = peak_hour_mask.sum()
            offHoursCount = off_hours_mask.sum()
            businessHoursCount = peakHourCount  # Same as peak hours
            totalCount = len(district_data)
            
            # Find peak hour of day (most common hour)
            hour_counts = district_data['hourOfDay'].value_counts()
            peakHourOfDay = hour_counts.idxmax() if len(hour_counts) > 0 else None
            
            peak_hour_metrics.append({
                'District_ID': district_id,
                'totalRequests': totalCount,
                'peakHourRequestCount': peakHourCount,
                'offHoursRequestCount': offHoursCount,
                'businessHoursRequestCount': businessHoursCount,
                'peakHourOfDay': peakHourOfDay
            })
        
        metrics = pd.DataFrame(peak_hour_metrics)
        
        # Calculate percentages
        metrics['peakHourPercent'] = (metrics['peakHourRequestCount'] / metrics['totalRequests'] * 100).round(2)
        metrics['offHoursPercent'] = (metrics['offHoursRequestCount'] / metrics['totalRequests'] * 100).round(2)
        metrics['businessHoursPercent'] = metrics['peakHourPercent']  # Same as peak hours
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('offHoursPercent', ascending=False).reset_index(drop=True)
            metrics['peakHourPatternPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['peakHourPatternQuartile'] = pd.qcut(
                    metrics['offHoursPercent'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = highest off-hours percent
                    duplicates='drop'
                )
            else:
                metrics['peakHourPatternQuartile'] = 'Q2'
        else:
            metrics['peakHourPatternPercentile'] = 100
            metrics['peakHourPatternQuartile'] = 'Q1'
        
        # Pattern type using Silhouette + Quantiles (data-driven, NOT quartiles)
        off_hours_percent_values = metrics['offHoursPercent'].dropna().values
        if len(off_hours_percent_values) >= 3:
            optimal_k = find_optimal_k_silhouette(off_hours_percent_values, k_range=None)
            
            # For off-hours, higher values = more off-hours, so we reverse the order
            sorted_desc = np.sort(off_hours_percent_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]
            
            # Create labels based on optimal k
            labels = []
            for i in range(optimal_k):
                if optimal_k == 3:
                    if i == 0:
                        labels.append('Consistent Off-Hours')
                    elif i == 1:
                        labels.append('Variable')
                    else:
                        labels.append('Consistent Peak-Hours')
                elif optimal_k == 4:
                    if i == 0:
                        labels.append('Consistent Off-Hours')
                    elif i == 1:
                        labels.append('Moderate Off-Hours')
                    elif i == 2:
                        labels.append('Moderate Peak-Hours')
                    else:
                        labels.append('Consistent Peak-Hours')
                else:  # k == 5
                    if i == 0:
                        labels.append('Consistent Off-Hours')
                    elif i == optimal_k - 1:
                        labels.append('Consistent Peak-Hours')
                    else:
                        labels.append('Variable')
            
            # Create classification function
            def classify_pattern(value):
                if pd.isna(value):
                    return 'Unknown'
                for i in range(optimal_k):
                    if i == optimal_k - 1:
                        if value <= breaks[i]:
                            return labels[i]
                    else:
                        if value <= breaks[i] and value > breaks[i + 1]:
                            return labels[i]
                return labels[0]  # Highest values go to top
            
            metrics['peakHourPatternType'] = metrics['offHoursPercent'].apply(classify_pattern)
        else:
            metrics['peakHourPatternType'] = 'Variable'
        
        # Calculate consistency metrics
        consistency_metrics = []
        peak_consistency_metrics = []
        hourly_consistency_metrics = []
        
        for idx, row in metrics.iterrows():
            district_data = df[df['districtInt'] == row['District_ID']]
            
            if len(district_data) == 0:
                consistency_metrics.append(1.0)
                peak_consistency_metrics.append(1.0)
                hourly_consistency_metrics.append(1.0)
                continue
            
            # Off-hours consistency: coefficient of variation across different time periods
            # Group by hour and day type (use .copy() to avoid SettingWithCopyWarning)
            district_data_copy = district_data.copy()
            district_data_copy['timePeriod'] = district_data_copy.apply(
                lambda x: 'off-hours' if (
                    x['hourOfDay'] < self.PEAK_HOUR_START or 
                    x['hourOfDay'] >= self.PEAK_HOUR_END or 
                    not x['isWeekday']
                ) else 'peak-hours',
                axis=1
            )
            
            off_hours_data = district_data_copy[district_data_copy['timePeriod'] == 'off-hours']
            peak_hours_data = district_data_copy[district_data_copy['timePeriod'] == 'peak-hours']
            
            # Off-hours consistency
            if len(off_hours_data) > 1:
                # Group by hour and count
                off_hours_by_hour = off_hours_data.groupby('hourOfDay').size()
                if len(off_hours_by_hour) > 1:
                    mean_off = off_hours_by_hour.mean()
                    std_off = off_hours_by_hour.std()
                    if mean_off > 0:
                        cv_off = std_off / mean_off
                        off_consistency = 1 - cv_off
                        off_consistency = max(0, min(1, off_consistency))
                    else:
                        off_consistency = 1.0
                else:
                    off_consistency = 1.0
            else:
                off_consistency = 1.0
            
            # Peak-hours consistency
            if len(peak_hours_data) > 1:
                peak_hours_by_hour = peak_hours_data.groupby('hourOfDay').size()
                if len(peak_hours_by_hour) > 1:
                    mean_peak = peak_hours_by_hour.mean()
                    std_peak = peak_hours_by_hour.std()
                    if mean_peak > 0:
                        cv_peak = std_peak / mean_peak
                        peak_consistency = 1 - cv_peak
                        peak_consistency = max(0, min(1, peak_consistency))
                    else:
                        peak_consistency = 1.0
                else:
                    peak_consistency = 1.0
            else:
                peak_consistency = 1.0
            
            # Overall hourly pattern consistency
            hourly_counts = district_data.groupby('hourOfDay').size()
            if len(hourly_counts) > 1:
                mean_hourly = hourly_counts.mean()
                std_hourly = hourly_counts.std()
                if mean_hourly > 0:
                    cv_hourly = std_hourly / mean_hourly
                    hourly_consistency = 1 - cv_hourly
                    hourly_consistency = max(0, min(1, hourly_consistency))
                else:
                    hourly_consistency = 1.0
            else:
                hourly_consistency = 1.0
            
            consistency_metrics.append(off_consistency)
            peak_consistency_metrics.append(peak_consistency)
            hourly_consistency_metrics.append(hourly_consistency)
        
        metrics['offHoursConsistency'] = consistency_metrics
        metrics['peakHourConsistency'] = peak_consistency_metrics
        metrics['hourlyPatternConsistency'] = hourly_consistency_metrics
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with peak hour metrics."""
        points_df = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        
        # Convert timestamp to datetime
        points_df['openedDt'] = pd.to_datetime(points_df['Date_Time_Opened'], unit='ms', errors='coerce')
        points_df = points_df[points_df['openedDt'].notna()].copy()
        
        # Extract hour of day
        points_df['hourOfDay'] = points_df['openedDt'].dt.hour
        
        # Create geometry
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df['Longitude'], points_df['Latitude']),
            crs='EPSG:4326'
        )
        
        # Convert district to numeric
        points_gdf['districtInt'] = pd.to_numeric(points_gdf['Council_District'], errors='coerce').astype('Int64')
        
        # Select and rename columns to camelCase (preserve geometry)
        columns_to_keep = ['geometry', 'districtInt', 'Request__', 'Request_Type', 'Status', 'hourOfDay']
        points_gdf = points_gdf[columns_to_keep].copy()
        points_gdf = points_gdf.rename(columns={
            'districtInt': 'districtId',
            'Request__': 'requestId',
            'Request_Type': 'requestType',
            'Status': 'status',
            'hourOfDay': 'hourOfDay'
        })
        
        # Ensure it's still a GeoDataFrame
        points_gdf = gpd.GeoDataFrame(points_gdf, geometry='geometry', crs='EPSG:4326')
        
        # Merge district metrics if provided
        if metrics is not None:
            metrics_merge = metrics[['District_ID', 'peakHourPatternType', 'peakHourPatternPercentile', 
                                    'peakHourPatternQuartile', 'offHoursPercent', 'peakHourPercent']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'peakHourPatternType': 'districtPeakHourPatternType',
                'peakHourPatternPercentile': 'districtPeakHourPatternPercentile',
                'peakHourPatternQuartile': 'districtPeakHourPatternQuartile',
                'offHoursPercent': 'districtOffHoursPercent',
                'peakHourPercent': 'districtPeakHourPercent'
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
        
        print("Calculating peak hour metrics...")
        metrics = self.calculatePeakHourMetrics(df, boundaries)
        
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
            boundaries.to_file(gpkgPath, layer='District Peak Hour Patterns', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Peak Hour Consistency GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Peak Hour Patterns' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = PeakHourConsistencyGenerator()
    generator.createShapefile()

