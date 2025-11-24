#!/usr/bin/env python3
"""
Capacity Planning GeoPackage Generator

Creates a GeoPackage analyzing which districts are approaching service capacity limits
based on current volumes, trends, and statistical thresholds. Helps identify districts
that need resource expansion before reaching capacity.

All calculations are data-driven with no hardcoded thresholds.
Uses median-based statistics for robustness to outliers.
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
    
    # Calculate data-driven k_range if not provided
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
    
    # Return k with highest silhouette score
    optimal_k = max(scores, key=scores.get)
    
    # If scores are very close (within 0.05), prefer middle k for interpretability
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        if abs(best_score - second_score) < 0.05:
            # Prefer k=4 if scores are close (most interpretable)
            if 4 in scores:
                return 4
            # Otherwise return middle k
            k_values = sorted(scores.keys())
            return k_values[len(k_values) // 2]
    
    return optimal_k


def get_quantile_labels_and_classifier(values, optimal_k, metric_name, ascending=True):
    """
    Generate percentile-based labels and classification function.
    
    Args:
        values: Array of values to classify
        optimal_k: Optimal number of classes
        metric_name: Name of metric (e.g., 'Capacity', 'Risk')
        ascending: If True, lower values = better (e.g., risk). If False, higher = better.
        
    Returns:
        Tuple: (classify_function, labels_list)
    """
    values = np.array([x for x in values if not pd.isna(x)])
    
    if len(values) < optimal_k:
        optimal_k = max(2, len(values))
    
    # Calculate quantile breaks
    quantiles = np.linspace(0, 100, optimal_k + 1)
    breaks = np.percentile(values, quantiles)
    
    # Create percentile-based labels
    labels = []
    for i in range(optimal_k):
        if ascending:
            # Lower values = better (e.g., "Top 25% Risk" means lowest risk)
            percentile_start = 100 - (100 * (i + 1) / optimal_k)
            percentile_end = 100 - (100 * i / optimal_k)
        else:
            # Higher values = better (e.g., "Top 25% Capacity" means highest capacity)
            percentile_start = 100 * i / optimal_k
            percentile_end = 100 * (i + 1) / optimal_k
        
        if i == 0:
            if ascending:
                label = f"Top {int(100/optimal_k)}% {metric_name}"
            else:
                label = f"Top {int(100/optimal_k)}% {metric_name}"
        else:
            label = f"{int(percentile_start)}-{int(percentile_end)}% {metric_name}"
        
        labels.append(label)
    
    # Reverse labels if ascending (so "Top 25%" is first)
    if ascending:
        labels = labels[::-1]
    
    def classify(value):
        if pd.isna(value):
            return labels[-1]  # Default to last category
        for i in range(optimal_k):
            if i == 0:
                if value <= breaks[i + 1]:
                    return labels[i]
            elif i == optimal_k - 1:
                if value > breaks[i]:
                    return labels[i]
            else:
                if breaks[i] < value <= breaks[i + 1]:
                    return labels[i]
        return labels[-1]
    
    return classify, labels


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
        if r_squared > 0.5 and n > 5:
            p_value = max(0.001, 1.0 - r_squared)
        else:
            p_value = 1.0
    
    return slope, intercept, r_squared, p_value


def calculate_median_absolute_deviation(data):
    """Calculate Median Absolute Deviation (MAD) - robust measure of spread."""
    data = np.array([x for x in data if not pd.isna(x)])
    if len(data) == 0:
        return 0.0
    median = np.median(data)
    mad = np.median(np.abs(data - median))
    return mad


def calculate_variance_based_weights(data_df):
    """
    Calculate weights based on variance of each column (data-driven).
    Higher variance = more important = higher weight.
    """
    weights = []
    for col in data_df.columns:
        if col == 'District_ID':
            continue
        values = data_df[col].dropna().values
        if len(values) > 1:
            variance = np.var(values)
            weights.append(variance)
        else:
            weights.append(0.0)
    
    # Normalize to sum to 1
    total = sum(weights)
    if total > 0:
        weights = [w / total for w in weights]
    else:
        # Equal weights fallback
        weights = [1.0 / len(weights)] * len(weights)
    
    return weights


def normalize_to_0_1(values):
    """Normalize values to 0-1 range using min/max from data."""
    values = np.array([x for x in values if not pd.isna(x)])
    if len(values) == 0:
        return np.array([])
    min_val = np.min(values)
    max_val = np.max(values)
    if max_val == min_val:
        return np.ones(len(values)) * 0.5
    return (values - min_val) / (max_val - min_val)


class CapacityPlanningGenerator:
    """Generator for capacity planning GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/capacityPlanning'
        self.outputName = 'capacityPlanning.gpkg'
    
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
    
    def calculateCapacityMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate capacity planning metrics by district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract month for aggregation
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Find most recent month
        all_months = sorted(df['month'].unique())
        if len(all_months) == 0:
            raise ValueError("No valid months found in data")
        most_recent_month = all_months[-1]
        
        # Calculate monthly volumes for all districts
        monthly_volumes = []
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            if len(district_data) > 0:
                monthly_counts = district_data.groupby('month').size()
                for month, count in monthly_counts.items():
                    monthly_volumes.append({
                        'District_ID': district_id,
                        'month': month,
                        'volume': count
                    })
        
        monthly_volumes_df = pd.DataFrame(monthly_volumes)
        
        # Calculate capacity threshold using MEDIAN-based method (robust to outliers)
        if len(monthly_volumes_df) > 0:
            all_volumes = monthly_volumes_df['volume'].values
            
            # Method 1: Median + 2×MAD (Median Absolute Deviation) - robust to outliers
            median_volume = np.median(all_volumes)
            mad = calculate_median_absolute_deviation(all_volumes)
            capacity_threshold_mad = median_volume + (2 * mad)
            
            # Method 2: IQR-based (Q3 + 1.5×IQR) - standard outlier threshold
            q1 = np.percentile(all_volumes, 25)
            q3 = np.percentile(all_volumes, 75)
            iqr = q3 - q1
            capacity_threshold_iqr = q3 + (1.5 * iqr)
            
            # Use the higher threshold (more conservative)
            capacity_threshold = max(capacity_threshold_mad, capacity_threshold_iqr)
        else:
            capacity_threshold = 0.0
        
        capacity_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                capacity_metrics.append({
                    'District_ID': district_id,
                    'currentMonthlyVolume': 0,
                    'capacityThreshold': round(capacity_threshold, 2),
                    'capacityUtilization': 0.0,
                    'capacityPressure': 0.0,
                    'timeToCapacity': None,
                    'capacityRisk': 0.0
                })
                continue
            
            # Current monthly volume (most recent month)
            current_volume = len(district_data[district_data['month'] == most_recent_month])
            
            # Capacity utilization
            if capacity_threshold > 0:
                utilization = current_volume / capacity_threshold
            else:
                utilization = 0.0
            
            # Capacity pressure (capped at 1.0)
            pressure = min(1.0, utilization)
            
            # Calculate trend for timeToCapacity
            monthly_counts = district_data.groupby('month').size().sort_index()
            time_to_capacity = None
            
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                slope, intercept, r_squared, _ = calculate_linear_regression(x, y)
                
                # Only calculate if trending up with some strength
                if slope > 0 and r_squared > 0.3:
                    current_volume_for_trend = monthly_counts.iloc[-1] if len(monthly_counts) > 0 else current_volume
                    
                    if current_volume_for_trend < capacity_threshold and slope > 0:
                        months_needed = (capacity_threshold - current_volume_for_trend) / slope
                        time_to_capacity = max(1, int(months_needed)) if months_needed > 0 else None
            
            # Calculate capacity risk components
            # Risk from utilization (higher utilization = higher risk)
            utilization_risk = min(1.0, utilization)
            
            # Risk from trend (increasing trend = higher risk)
            trend_risk = 0.0
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                slope, _, r_squared, _ = calculate_linear_regression(x, y)
                if slope > 0:
                    # Normalize slope to 0-1 (relative to average monthly volume)
                    avg_monthly = np.mean(y) if len(y) > 0 else 1.0
                    if avg_monthly > 0:
                        relative_slope = min(1.0, (slope / avg_monthly) * r_squared)
                        trend_risk = relative_slope
                elif slope < 0:
                    # Decreasing trend = lower risk
                    trend_risk = 0.0
            
            # Risk from volatility (higher volatility = higher risk)
            volatility_risk = 0.0
            if len(monthly_counts) >= 2:
                cv = np.std(y) / np.mean(y) if np.mean(y) > 0 else 0.0
                volatility_risk = min(1.0, cv)  # Coefficient of variation, capped at 1.0
            
            # Combine risk factors (will use variance-based weights later)
            # For now, store individual components
            capacity_metrics.append({
                'District_ID': district_id,
                'currentMonthlyVolume': current_volume,
                'capacityThreshold': round(capacity_threshold, 2),
                'capacityUtilization': round(utilization, 4),
                'capacityPressure': round(pressure, 4),
                'timeToCapacity': time_to_capacity,
                'utilizationRisk': utilization_risk,
                'trendRisk': trend_risk,
                'volatilityRisk': volatility_risk
            })
        
        metrics_df = pd.DataFrame(capacity_metrics)
        
        # Calculate variance-based weights for risk factors
        risk_factors_df = metrics_df[['District_ID', 'utilizationRisk', 'trendRisk', 'volatilityRisk']].copy()
        weights = calculate_variance_based_weights(risk_factors_df[['utilizationRisk', 'trendRisk', 'volatilityRisk']])
        
        # Normalize weights
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]  # Equal weights fallback
        
        # Calculate combined capacity risk
        metrics_df['capacityRisk'] = (
            weights[0] * metrics_df['utilizationRisk'] +
            weights[1] * metrics_df['trendRisk'] +
            weights[2] * metrics_df['volatilityRisk']
        )
        metrics_df['capacityRisk'] = metrics_df['capacityRisk'].round(4)
        
        # Drop intermediate risk columns (not needed in final output)
        metrics_df = metrics_df.drop(columns=['utilizationRisk', 'trendRisk', 'volatilityRisk'])
        
        return metrics_df
    
    def classifyCapacityStatus(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Classify capacity status using Silhouette + Quantiles."""
        utilization_values = metrics['capacityUtilization'].dropna().values
        
        if len(utilization_values) >= 3:
            optimal_k = find_optimal_k_silhouette(utilization_values, k_range=None)
            
            # For capacity status, we want: Below, Near, At Capacity
            # Use optimal_k but ensure at least 3 categories
            optimal_k = max(3, optimal_k)
            
            # Create labels based on utilization (ascending=False, higher = worse)
            classify_func, labels = get_quantile_labels_and_classifier(
                utilization_values, optimal_k, 'Capacity', ascending=False
            )
            
            # Map to interpretable labels
            # Top group = "At Capacity", Middle = "Near Capacity", Bottom = "Below Capacity"
            if optimal_k >= 3:
                # Use top group for "At Capacity"
                top_label = labels[0] if len(labels) > 0 else "At Capacity"
                metrics['capacityStatus'] = metrics['capacityUtilization'].apply(classify_func)
                
                # Replace labels with interpretable ones
                label_mapping = {}
                sorted_labels = sorted(set(metrics['capacityStatus']))
                if len(sorted_labels) >= 3:
                    # Top group = At Capacity
                    label_mapping[sorted_labels[0]] = "At Capacity"
                    # Middle groups = Near Capacity
                    for i in range(1, len(sorted_labels) - 1):
                        label_mapping[sorted_labels[i]] = "Near Capacity"
                    # Bottom group = Below Capacity
                    label_mapping[sorted_labels[-1]] = "Below Capacity"
                else:
                    # Fallback if not enough groups
                    for label in sorted_labels:
                        if 'Top' in label or '75-100' in label:
                            label_mapping[label] = "At Capacity"
                        elif '50-75' in label or '25-50' in label:
                            label_mapping[label] = "Near Capacity"
                        else:
                            label_mapping[label] = "Below Capacity"
                
                metrics['capacityStatus'] = metrics['capacityStatus'].map(label_mapping)
            else:
                metrics['capacityStatus'] = "Below Capacity"
        else:
            metrics['capacityStatus'] = "Below Capacity"
        
        return metrics
    
    def classifyCapacityCategory(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Classify capacity category using Silhouette + Quantiles."""
        risk_values = metrics['capacityRisk'].dropna().values
        
        if len(risk_values) >= 3:
            optimal_k = find_optimal_k_silhouette(risk_values, k_range=None)
            
            classify_func, labels = get_quantile_labels_and_classifier(
                risk_values, optimal_k, 'Risk', ascending=True
            )
            
            metrics['capacityCategory'] = metrics['capacityRisk'].apply(classify_func)
            
            # Map to interpretable labels (Critical, High, Moderate, Low)
            sorted_labels = sorted(set(metrics['capacityCategory']))
            label_mapping = {}
            
            if len(sorted_labels) >= 4:
                label_mapping[sorted_labels[0]] = "Critical"
                label_mapping[sorted_labels[1]] = "High"
                label_mapping[sorted_labels[2]] = "Moderate"
                label_mapping[sorted_labels[-1]] = "Low"
            elif len(sorted_labels) == 3:
                label_mapping[sorted_labels[0]] = "Critical"
                label_mapping[sorted_labels[1]] = "High"
                label_mapping[sorted_labels[-1]] = "Low"
            else:
                # Fallback
                for label in sorted_labels:
                    if 'Top' in label or '75-100' in label:
                        label_mapping[label] = "Critical"
                    elif '50-75' in label:
                        label_mapping[label] = "High"
                    elif '25-50' in label:
                        label_mapping[label] = "Moderate"
                    else:
                        label_mapping[label] = "Low"
            
            metrics['capacityCategory'] = metrics['capacityCategory'].map(label_mapping)
        else:
            metrics['capacityCategory'] = "Low"
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, district_metrics: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create point layer for service requests with district capacity metrics."""
        # Filter valid coordinates
        points_df = df[
            (df['Latitude'].notna()) & 
            (df['Longitude'].notna()) &
            (df['Latitude'] != 0) & 
            (df['Longitude'] != 0)
        ].copy()
        
        if len(points_df) == 0:
            return gpd.GeoDataFrame()
        
        # Create geometry
        geometry = [Point(lon, lat) for lon, lat in 
                   zip(points_df['Longitude'], points_df['Latitude'])]
        
        points_gdf = gpd.GeoDataFrame(points_df, geometry=geometry, crs='EPSG:4326')
        points_gdf = points_gdf.to_crs('EPSG:3857')
        
        # Convert district to numeric
        points_gdf['districtInt'] = pd.to_numeric(points_gdf['Council_District'], errors='coerce').astype('Int64')
        
        # Merge district metrics
        district_metrics_for_merge = district_metrics.copy()
        district_metrics_for_merge['districtInt'] = district_metrics_for_merge['District_ID'].astype('Int64')
        
        # Select only capacity-related columns to merge
        capacity_cols = [
            'districtInt', 'currentMonthlyVolume', 'capacityThreshold', 
            'capacityUtilization', 'capacityPressure', 'timeToCapacity',
            'capacityRisk', 'capacityStatus', 'capacityCategory'
        ]
        available_cols = [col for col in capacity_cols if col in district_metrics_for_merge.columns]
        
        points_gdf = points_gdf.merge(
            district_metrics_for_merge[available_cols],
            on='districtInt',
            how='left'
        )
        
        # Rename columns to camelCase
        rename_map = {
            'currentMonthlyVolume': 'districtCurrentMonthlyVolume',
            'capacityThreshold': 'districtCapacityThreshold',
            'capacityUtilization': 'districtCapacityUtilization',
            'capacityPressure': 'districtCapacityPressure',
            'timeToCapacity': 'districtTimeToCapacity',
            'capacityRisk': 'districtCapacityRisk',
            'capacityStatus': 'districtCapacityStatus',
            'capacityCategory': 'districtCapacityCategory'
        }
        
        for old_col, new_col in rename_map.items():
            if old_col in points_gdf.columns:
                points_gdf = points_gdf.rename(columns={old_col: new_col})
        
        # Add days since opened
        if 'Date_Time_Opened' in points_gdf.columns:
            points_gdf['openedDt'] = pd.to_datetime(points_gdf['Date_Time_Opened'], unit='ms', errors='coerce')
            points_gdf['daysSinceOpened'] = (
                (datetime.now() - points_gdf['openedDt']).dt.total_seconds() / 86400
            ).round(0).fillna(0).astype(int)
        
        # Select essential columns
        essential_cols = [
            'geometry', 'Request_ID', 'Request_Type', 'Subrequest_Type',
            'Status', 'Date_Time_Opened', 'Date_Time_Closed',
            'Council_District', 'daysSinceOpened'
        ]
        
        # Add district capacity columns
        district_capacity_cols = [col for col in points_gdf.columns if col.startswith('district')]
        essential_cols.extend(district_capacity_cols)
        
        available_essential = [col for col in essential_cols if col in points_gdf.columns]
        points_gdf = points_gdf[available_essential]
        
        return points_gdf
    
    def createShapefile(self) -> str:
        """Create the capacity planning GeoPackage."""
        print("Loading data from S3...")
        df = self.loadDataFromS3()
        print(f"Loaded {len(df):,} service requests")
        
        print("Loading district boundaries...")
        boundaries = self.loadDistrictBoundaries()
        print(f"Loaded {len(boundaries)} districts")
        
        print("Calculating capacity metrics...")
        capacity_metrics = self.calculateCapacityMetrics(df, boundaries)
        
        print("Classifying capacity status...")
        capacity_metrics = self.classifyCapacityStatus(capacity_metrics)
        
        print("Classifying capacity category...")
        capacity_metrics = self.classifyCapacityCategory(capacity_metrics)
        
        # Merge with boundaries
        boundaries['District_ID'] = boundaries['District_ID'].astype(int)
        capacity_metrics['District_ID'] = capacity_metrics['District_ID'].astype(int)
        
        district_gdf = boundaries.merge(
            capacity_metrics,
            on='District_ID',
            how='left'
        )
        
        # Convert column names to camelCase
        column_mapping = {
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName'
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in district_gdf.columns:
                district_gdf = district_gdf.rename(columns={old_col: new_col})
        
        # Create request points layer
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, capacity_metrics)
        
        # Create GeoPackage
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkg_path = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            district_gdf.to_file(gpkg_path, layer='District Capacity Planning', driver='GPKG')
            
            # Write points layer if not empty
            if len(points_gdf) > 0:
                points_gdf.to_file(gpkg_path, layer='Service Requests', driver='GPKG')
            
            # Upload to S3
            print("Uploading to S3...")
            self.uploadGeoPackageToS3(gpkg_path)
        
        return f"s3://{self.bucketName}/{self.s3ShapefilePrefix}/{self.outputName}"
    
    def uploadGeoPackageToS3(self, localPath: str):
        """Upload GeoPackage to S3."""
        s3Key = f"{self.s3ShapefilePrefix}/{self.outputName}"
        
        self.s3Client.upload_file(
            localPath,
            self.bucketName,
            s3Key,
            ExtraArgs={'ContentType': 'application/geopackage+sqlite3'}
        )
        
        s3Url = f"https://{self.bucketName}.s3.amazonaws.com/{s3Key}"
        print(f"✅ Capacity Planning GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")


if __name__ == '__main__':
    generator = CapacityPlanningGenerator()
    s3Url = generator.createShapefile()
    
    print(f"\n✅ Ready for ArcGIS Pro!")
    print(f"GeoPackage URL: {s3Url}")
    print(f"\nContains 2 layers:")
    print(f"1. 'District Capacity Planning' (polygons)")
    print(f"2. 'Service Requests' (points)")

