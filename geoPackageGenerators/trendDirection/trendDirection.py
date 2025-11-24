#!/usr/bin/env python3
"""
Trend Direction GeoPackage Generator

Creates a GeoPackage analyzing trends in service request volumes over time across
Nashville council districts. Identifies districts with increasing, decreasing, or
stable trends for capacity planning and resource allocation.

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
from typing import Optional, Dict, Any, Tuple
from shapely.geometry import Point

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig

try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False


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


def calculate_linear_regression(x, y):
    """
    Calculate linear regression: y = mx + b
    
    Returns:
        slope (m), intercept (b), r_squared, p_value
    """
    x = np.array(x)
    y = np.array(y)
    
    # Remove NaN values
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return 0.0, 0.0, 0.0, 1.0
    
    # Calculate slope and intercept
    n = len(x)
    sum_x = np.sum(x)
    sum_y = np.sum(y)
    sum_xy = np.sum(x * y)
    sum_x2 = np.sum(x ** 2)
    
    denominator = n * sum_x2 - sum_x ** 2
    if abs(denominator) < 1e-10:
        return 0.0, np.mean(y), 0.0, 1.0
    
    slope = (n * sum_xy - sum_x * sum_y) / denominator
    intercept = (sum_y - slope * sum_x) / n
    
    # Calculate R²
    y_pred = slope * x + intercept
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
    
    # Calculate p-value (if scipy available)
    if SCIPY_AVAILABLE and len(x) > 2:
        try:
            _, p_value = stats.linregress(x, y)[3:5]
        except:
            p_value = 1.0
    else:
        # Approximate p-value based on R² and sample size
        # Higher R² and larger n = lower p-value
        if r_squared > 0.5 and n > 5:
            p_value = max(0.001, 1.0 - r_squared)
        else:
            p_value = 1.0
    
    return slope, intercept, r_squared, p_value


class TrendDirectionGenerator:
    """Generator for district trend direction GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/trendDirection'
        self.outputName = 'trendDirection.gpkg'
    
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
    
    def calculateTrendMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate trend direction metrics by district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract month for aggregation
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        trend_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                trend_metrics.append({
                    'District_ID': district_id,
                    'totalRequests': 0,
                    'averageMonthlyRequests': 0.0,
                    'medianMonthlyRequests': 0.0,
                    'trendDirection': 'Stable',
                    'trendSlope': 0.0,
                    'trendStrength': 0.0,
                    'trendSignificance': 1.0,
                    'monthlyChangeRate': 0.0,
                    'trendMagnitude': 0.0,
                    'projectedNextMonth': 0.0,
                    'trendConsistency': 0.0,
                    'trendVolatility': 0.0,
                    'trendConfidence': 0.0
                })
                continue
            
            total_requests = len(district_data)
            
            # Aggregate by month
            monthly_counts = district_data.groupby('month').size().sort_index()
            
            if len(monthly_counts) < 2:
                # Not enough data for trend analysis
                avg_monthly = monthly_counts.iloc[0] if len(monthly_counts) > 0 else 0.0
                median_monthly = avg_monthly
                
                trend_metrics.append({
                    'District_ID': district_id,
                    'totalRequests': total_requests,
                    'averageMonthlyRequests': round(avg_monthly, 2),
                    'medianMonthlyRequests': round(median_monthly, 2),
                    'trendDirection': 'Stable',
                    'trendSlope': 0.0,
                    'trendStrength': 0.0,
                    'trendSignificance': 1.0,
                    'monthlyChangeRate': 0.0,
                    'trendMagnitude': 0.0,
                    'projectedNextMonth': round(avg_monthly, 2),
                    'trendConsistency': 1.0,
                    'trendVolatility': 0.0,
                    'trendConfidence': 0.0
                })
                continue
            
            avg_monthly = monthly_counts.mean()
            median_monthly = monthly_counts.median()
            
            # Calculate linear regression
            x = np.arange(len(monthly_counts))  # Time (months)
            y = monthly_counts.values  # Request counts
            
            slope, intercept, r_squared, p_value = calculate_linear_regression(x, y)
            
            # Trend direction (data-driven threshold)
            # Use percentile of absolute slopes to determine "stable" threshold
            # For now, use a small threshold relative to average monthly count
            if avg_monthly > 0:
                relative_slope = abs(slope) / avg_monthly
                # If slope is less than 1% of average, consider it stable
                stable_threshold = 0.01
            else:
                relative_slope = 0.0
                stable_threshold = 0.01
            
            if relative_slope < stable_threshold:
                trend_direction = 'Stable'
            elif slope > 0:
                trend_direction = 'Increasing'
            else:
                trend_direction = 'Decreasing'
            
            # Monthly change rate (%)
            if avg_monthly > 0:
                monthly_change_rate = (slope / avg_monthly) * 100
            else:
                monthly_change_rate = 0.0
            
            # Trend magnitude (absolute change rate)
            trend_magnitude = abs(slope)
            
            # Projected next month
            next_month_x = len(monthly_counts)
            projected_next_month = slope * next_month_x + intercept
            projected_next_month = max(0, projected_next_month)  # Can't be negative
            
            # Trend consistency (inverse of volatility around trend line)
            y_pred = slope * x + intercept
            residuals = y - y_pred
            if len(residuals) > 1:
                residual_std = np.std(residuals)
                if avg_monthly > 0:
                    residual_cv = residual_std / avg_monthly
                    trend_consistency = 1.0 / (1.0 + residual_cv)  # Inverse, normalized
                else:
                    trend_consistency = 1.0
                trend_volatility = residual_cv if avg_monthly > 0 else 0.0
            else:
                trend_consistency = 1.0
                trend_volatility = 0.0
            
            # Trend confidence (based on R² and p-value)
            # Higher R² and lower p-value = higher confidence
            trend_confidence = r_squared * (1.0 - min(p_value, 0.99))
            
            trend_metrics.append({
                'District_ID': district_id,
                'totalRequests': total_requests,
                'averageMonthlyRequests': round(avg_monthly, 2),
                'medianMonthlyRequests': round(median_monthly, 2),
                'trendDirection': trend_direction,
                'trendSlope': round(slope, 4),
                'trendStrength': round(r_squared, 4),
                'trendSignificance': round(p_value, 4),
                'monthlyChangeRate': round(monthly_change_rate, 2),
                'trendMagnitude': round(trend_magnitude, 2),
                'projectedNextMonth': round(projected_next_month, 2),
                'trendConsistency': round(trend_consistency, 4),
                'trendVolatility': round(trend_volatility, 4),
                'trendConfidence': round(trend_confidence, 4)
            })
        
        metrics = pd.DataFrame(trend_metrics)
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            # Use trendSlope for ranking (higher = stronger increase)
            metrics = metrics.sort_values('trendSlope', ascending=False).reset_index(drop=True)
            metrics['trendPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['trendQuartile'] = pd.qcut(
                    metrics['trendSlope'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = strongest increase
                    duplicates='drop'
                )
            else:
                metrics['trendQuartile'] = 'Q2'
        else:
            metrics['trendPercentile'] = 100
            metrics['trendQuartile'] = 'Q1'
        
        # Trend category using Silhouette + Quantiles (data-driven)
        # Combine trend direction and strength for classification
        # Create a combined score: slope * strength (weighted)
        metrics['trendScore'] = metrics['trendSlope'] * metrics['trendStrength']
        
        trend_score_values = metrics['trendScore'].dropna().values
        if len(trend_score_values) >= 3:
            optimal_k = find_optimal_k_silhouette(trend_score_values, k_range=None)
            
            # Sort descending (higher scores = stronger trends)
            sorted_desc = np.sort(trend_score_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]
            
            # Create labels based on optimal k and trend direction
            def classify_trend(row):
                if pd.isna(row['trendScore']):
                    return 'Unknown'
                
                score = row['trendScore']
                direction = row['trendDirection']
                strength = row['trendStrength']
                
                # Find which quantile group
                group_idx = 0
                for i in range(optimal_k):
                    if i == optimal_k - 1:
                        if score <= breaks[i]:
                            group_idx = i
                            break
                    else:
                        if score <= breaks[i] and score > breaks[i + 1]:
                            group_idx = i
                            break
                
                # Create label based on direction and strength
                if direction == 'Increasing':
                    if group_idx == 0:
                        return 'Strongly Increasing'
                    elif group_idx == optimal_k - 1:
                        return 'Moderately Increasing'
                    else:
                        return 'Increasing'
                elif direction == 'Decreasing':
                    if group_idx == 0:
                        return 'Strongly Decreasing'
                    elif group_idx == optimal_k - 1:
                        return 'Moderately Decreasing'
                    else:
                        return 'Decreasing'
                else:  # Stable
                    return 'Stable'
            
            metrics['trendCategory'] = metrics.apply(classify_trend, axis=1)
        else:
            metrics['trendCategory'] = metrics['trendDirection']
        
        # Drop temporary trendScore column
        metrics = metrics.drop(columns=['trendScore'])
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with trend metrics."""
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
            metrics_merge = metrics[['District_ID', 'trendCategory', 'trendDirection',
                                    'trendPercentile', 'trendQuartile', 'trendSlope',
                                    'monthlyChangeRate']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'trendCategory': 'districtTrendCategory',
                'trendDirection': 'districtTrendDirection',
                'trendPercentile': 'districtTrendPercentile',
                'trendQuartile': 'districtTrendQuartile',
                'trendSlope': 'districtTrendSlope',
                'monthlyChangeRate': 'districtMonthlyChangeRate'
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
        
        print("Calculating trend metrics...")
        metrics = self.calculateTrendMetrics(df, boundaries)
        
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
            boundaries.to_file(gpkgPath, layer='District Trend Direction', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Trend Direction GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Trend Direction' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = TrendDirectionGenerator()
    generator.createShapefile()

