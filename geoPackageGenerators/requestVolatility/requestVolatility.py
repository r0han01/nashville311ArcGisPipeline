#!/usr/bin/env python3
"""
Request Volatility GeoPackage Generator

Creates a GeoPackage analyzing request volatility/stability patterns across
Nashville council districts. Focuses on identifying districts with stable vs
volatile request patterns for resource planning and capacity management.

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


def calculate_coefficient_of_variation(values):
    """Calculate coefficient of variation (CV = std/mean)."""
    values = np.array([x for x in values if not pd.isna(x)])
    if len(values) < 2:
        return 0.0
    mean = np.mean(values)
    if mean == 0:
        return 0.0
    std = np.std(values)
    return std / mean


class RequestVolatilityGenerator:
    """Generator for district request volatility GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/requestVolatility'
        self.outputName = 'requestVolatility.gpkg'
    
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
    
    def calculateVolatilityMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate volatility/stability metrics by district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract temporal components
        df['date'] = df['openedDt'].dt.date
        df['week'] = df['openedDt'].dt.to_period('W').astype(str)
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        volatility_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                volatility_metrics.append({
                    'District_ID': district_id,
                    'totalRequests': 0,
                    'averageDailyRequests': 0.0,
                    'medianDailyRequests': 0.0,
                    'dailyVolatilityCoefficient': 0.0,
                    'weeklyVolatilityCoefficient': 0.0,
                    'monthlyVolatilityCoefficient': 0.0,
                    'overallTemporalVolatility': 0.0,
                    'volumeVolatilityCoefficient': 0.0,
                    'volumeStabilityIndex': 1.0,
                    'peakToTroughRatio': 1.0,
                    'requestTypeDiversity': 0,
                    'requestTypeConcentration': 0.0,
                    'requestTypeVolatility': 0.0,
                    'patternConsistencyScore': 1.0
                })
                continue
            
            total_requests = len(district_data)
            
            # Daily volatility
            daily_counts = district_data.groupby('date').size()
            average_daily = daily_counts.mean() if len(daily_counts) > 0 else 0.0
            median_daily = daily_counts.median() if len(daily_counts) > 0 else 0.0
            daily_cv = calculate_coefficient_of_variation(daily_counts.values) if len(daily_counts) > 1 else 0.0
            
            # Weekly volatility
            weekly_counts = district_data.groupby('week').size()
            weekly_cv = calculate_coefficient_of_variation(weekly_counts.values) if len(weekly_counts) > 1 else 0.0
            
            # Monthly volatility
            monthly_counts = district_data.groupby('month').size()
            monthly_cv = calculate_coefficient_of_variation(monthly_counts.values) if len(monthly_counts) > 1 else 0.0
            
            # Overall temporal volatility (weighted average)
            temporal_volatilities = [daily_cv, weekly_cv, monthly_cv]
            temporal_volatilities = [v for v in temporal_volatilities if v > 0]
            overall_temporal_volatility = np.mean(temporal_volatilities) if temporal_volatilities else 0.0
            
            # Volume volatility (using daily counts)
            volume_cv = daily_cv  # Same as daily volatility
            volume_stability_index = 1.0 / (1.0 + volume_cv) if volume_cv > 0 else 1.0  # Inverse, normalized
            
            # Peak to trough ratio
            if len(daily_counts) > 0:
                peak_day = daily_counts.max()
                trough_day = daily_counts.min()
                peak_to_trough = peak_day / trough_day if trough_day > 0 else 1.0
            else:
                peak_to_trough = 1.0
            
            # Request type volatility
            request_type_counts = district_data['Request_Type'].value_counts()
            request_type_diversity = len(request_type_counts)
            
            # Request type concentration (Herfindahl index)
            if total_requests > 0:
                proportions = request_type_counts / total_requests
                concentration = np.sum(proportions ** 2)  # Herfindahl index
            else:
                concentration = 0.0
            
            # Request type volatility over time (CV of type proportions by day)
            type_volatility = 0.0
            if len(daily_counts) > 1:
                daily_type_proportions = []
                for date in daily_counts.index:
                    day_data = district_data[district_data['date'] == date]
                    if len(day_data) > 0:
                        day_type_counts = day_data['Request_Type'].value_counts()
                        if len(day_type_counts) > 0:
                            # Use proportion of most common type as proxy
                            most_common_prop = day_type_counts.iloc[0] / len(day_data)
                            daily_type_proportions.append(most_common_prop)
                
                if len(daily_type_proportions) > 1:
                    type_volatility = calculate_coefficient_of_variation(daily_type_proportions)
            
            # Pattern consistency score (inverse of overall volatility, normalized)
            pattern_consistency = 1.0 / (1.0 + overall_temporal_volatility) if overall_temporal_volatility > 0 else 1.0
            
            volatility_metrics.append({
                'District_ID': district_id,
                'totalRequests': total_requests,
                'averageDailyRequests': round(average_daily, 2),
                'medianDailyRequests': round(median_daily, 2),
                'dailyVolatilityCoefficient': round(daily_cv, 4),
                'weeklyVolatilityCoefficient': round(weekly_cv, 4),
                'monthlyVolatilityCoefficient': round(monthly_cv, 4),
                'overallTemporalVolatility': round(overall_temporal_volatility, 4),
                'volumeVolatilityCoefficient': round(volume_cv, 4),
                'volumeStabilityIndex': round(volume_stability_index, 4),
                'peakToTroughRatio': round(peak_to_trough, 2),
                'requestTypeDiversity': request_type_diversity,
                'requestTypeConcentration': round(concentration, 4),
                'requestTypeVolatility': round(type_volatility, 4),
                'patternConsistencyScore': round(pattern_consistency, 4)
            })
        
        metrics = pd.DataFrame(volatility_metrics)
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            # Use overallTemporalVolatility for ranking (higher = more volatile)
            metrics = metrics.sort_values('overallTemporalVolatility', ascending=False).reset_index(drop=True)
            metrics['volatilityPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['volatilityQuartile'] = pd.qcut(
                    metrics['overallTemporalVolatility'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = most volatile
                    duplicates='drop'
                )
            else:
                metrics['volatilityQuartile'] = 'Q2'
        else:
            metrics['volatilityPercentile'] = 100
            metrics['volatilityQuartile'] = 'Q1'
        
        # Volatility category using Silhouette + Quantiles (data-driven)
        volatility_values = metrics['overallTemporalVolatility'].dropna().values
        if len(volatility_values) >= 3:
            optimal_k = find_optimal_k_silhouette(volatility_values, k_range=None)
            
            # Higher volatility = more volatile, so we sort descending
            sorted_desc = np.sort(volatility_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]
            
            # Create labels based on optimal k
            labels = []
            for i in range(optimal_k):
                if optimal_k == 3:
                    if i == 0:
                        labels.append('Highly Volatile')
                    elif i == 1:
                        labels.append('Moderate Volatility')
                    else:
                        labels.append('Stable')
                elif optimal_k == 4:
                    if i == 0:
                        labels.append('Highly Volatile')
                    elif i == 1:
                        labels.append('Moderate-High Volatility')
                    elif i == 2:
                        labels.append('Moderate-Low Volatility')
                    else:
                        labels.append('Stable')
                else:  # k == 5
                    if i == 0:
                        labels.append('Highly Volatile')
                    elif i == optimal_k - 1:
                        labels.append('Stable')
                    else:
                        labels.append('Moderate Volatility')
            
            # Create classification function
            def classify_volatility(value):
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
            
            metrics['volatilityCategory'] = metrics['overallTemporalVolatility'].apply(classify_volatility)
        else:
            metrics['volatilityCategory'] = 'Moderate Volatility'
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with volatility metrics."""
        points_df = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        
        # Create geometry
        points_gdf = gpd.GeoDataFrame(
            points_df,
            geometry=gpd.points_from_xy(points_df['Longitude'], points_df['Latitude']),
            crs='EPSG:4326'
        )
        
        # Convert district to numeric
        points_gdf['districtInt'] = pd.to_numeric(points_gdf['Council_District'], errors='coerce').astype('Int64')
        
        # Select and rename columns to camelCase (preserve geometry)
        columns_to_keep = ['geometry', 'districtInt', 'Request__', 'Request_Type', 'Status']
        points_gdf = points_gdf[columns_to_keep].copy()
        points_gdf = points_gdf.rename(columns={
            'districtInt': 'districtId',
            'Request__': 'requestId',
            'Request_Type': 'requestType',
            'Status': 'status'
        })
        
        # Ensure it's still a GeoDataFrame
        points_gdf = gpd.GeoDataFrame(points_gdf, geometry='geometry', crs='EPSG:4326')
        
        # Merge district metrics if provided
        if metrics is not None:
            metrics_merge = metrics[['District_ID', 'volatilityCategory', 'volatilityPercentile', 
                                    'volatilityQuartile', 'overallTemporalVolatility', 
                                    'volumeStabilityIndex', 'patternConsistencyScore']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'volatilityCategory': 'districtVolatilityCategory',
                'volatilityPercentile': 'districtVolatilityPercentile',
                'volatilityQuartile': 'districtVolatilityQuartile',
                'overallTemporalVolatility': 'districtOverallTemporalVolatility',
                'volumeStabilityIndex': 'districtVolumeStabilityIndex',
                'patternConsistencyScore': 'districtPatternConsistencyScore'
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
        
        print("Calculating volatility metrics...")
        metrics = self.calculateVolatilityMetrics(df, boundaries)
        
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
            boundaries.to_file(gpkgPath, layer='District Request Volatility', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Request Volatility GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Request Volatility' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = RequestVolatilityGenerator()
    generator.createShapefile()

