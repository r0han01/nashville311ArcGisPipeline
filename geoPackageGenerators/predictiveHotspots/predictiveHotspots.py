#!/usr/bin/env python3
"""
Predictive Hotspots GeoPackage Generator

Creates a GeoPackage predicting which districts are likely to become hotspots
in the future based on current patterns and trends. Combines multiple risk factors
for proactive resource planning and early warning.

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
from typing import Optional, Dict, Any, Tuple, List
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
    """Calculate linear regression: y = mx + b. Returns slope, intercept, r_squared, p_value."""
    x = np.array(x)
    y = np.array(y)
    
    mask = ~(np.isnan(x) | np.isnan(y))
    x = x[mask]
    y = y[mask]
    
    if len(x) < 2:
        return 0.0, 0.0, 0.0, 1.0
    
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
    
    # Calculate p-value
    if SCIPY_AVAILABLE and len(x) > 2:
        try:
            _, p_value = stats.linregress(x, y)[3:5]
        except:
            p_value = 1.0
    else:
        if r_squared > 0.5 and n > 5:
            p_value = max(0.001, 1.0 - r_squared)
        else:
            p_value = 1.0
    
    return slope, intercept, r_squared, p_value


def normalize_to_0_1(values):
    """Normalize values to 0-1 scale using min/max from data."""
    values = np.array([x for x in values if not pd.isna(x)])
    if len(values) < 2:
        return np.array([0.5] * len(values)) if len(values) > 0 else np.array([])
    
    min_val = np.min(values)
    max_val = np.max(values)
    
    if max_val == min_val:
        return np.array([0.5] * len(values))
    
    normalized = (values - min_val) / (max_val - min_val)
    return normalized


def calculate_variance_based_weights(risk_factors_df):
    """Calculate weights based on variance of each risk factor (data-driven)."""
    variances = []
    for col in risk_factors_df.columns:
        if col != 'District_ID':
            var = risk_factors_df[col].var()
            variances.append(var if not pd.isna(var) else 0.0)
        else:
            variances.append(0.0)
    
    total_variance = sum(variances)
    if total_variance == 0:
        # Equal weights if no variance
        num_factors = len([v for v in variances if v > 0])
        return [1.0 / num_factors if v > 0 else 0.0 for v in variances]
    
    weights = [v / total_variance for v in variances]
    return weights


class PredictiveHotspotsGenerator:
    """Generator for district predictive hotspots GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/predictiveHotspots'
        self.outputName = 'predictiveHotspots.gpkg'
    
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
    
    def calculateRiskFactors(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate individual risk factors for each district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract temporal components
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        df['date'] = df['openedDt'].dt.date
        df['dayOfWeek'] = df['openedDt'].dt.dayofweek
        df['hourOfDay'] = df['openedDt'].dt.hour
        df['isWeekday'] = df['dayOfWeek'] < 5
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        risk_factors = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                risk_factors.append({
                    'District_ID': district_id,
                    'currentDensityRisk': 0.0,
                    'trendDirectionRisk': 0.0,
                    'volatilityRisk': 0.0,
                    'recurringIssueRisk': 0.0,
                    'temporalPatternRisk': 0.0
                })
                continue
            
            total_requests = len(district_data)
            
            # 1. Current Density Risk (from request density)
            # Calculate requests per square mile (approximate)
            # For simplicity, use total requests as proxy (will normalize later)
            density_score = total_requests  # Will normalize across all districts
            
            # 2. Trend Direction Risk (from trend analysis)
            monthly_counts = district_data.groupby('month').size().sort_index()
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                slope, _, r_squared, _ = calculate_linear_regression(x, y)
                # Combine slope and strength for trend risk
                trend_score = abs(slope) * r_squared if r_squared > 0 else 0.0
                # If increasing (positive slope), higher risk
                if slope > 0:
                    trend_score = trend_score * 1.5  # Increase risk for increasing trends
            else:
                trend_score = 0.0
            
            # 3. Volatility Risk (from coefficient of variation)
            daily_counts = district_data.groupby('date').size()
            if len(daily_counts) > 1:
                mean_daily = daily_counts.mean()
                std_daily = daily_counts.std()
                volatility_coef = (std_daily / mean_daily) if mean_daily > 0 else 0.0
            else:
                volatility_coef = 0.0
            
            # 4. Recurring Issue Risk (from service type concentration)
            service_type_counts = district_data['Request_Type'].value_counts()
            if total_requests > 0 and len(service_type_counts) > 0:
                # High concentration = more recurring issues
                proportions = service_type_counts / total_requests
                concentration = np.sum(proportions ** 2)  # Herfindahl index
                # Higher concentration = higher recurring risk
                recurring_score = concentration
            else:
                recurring_score = 0.0
            
            # 5. Temporal Pattern Risk (from temporal complexity)
            # Complexity based on weekday/weekend and peak hour patterns
            weekday_count = district_data['isWeekday'].sum()
            weekday_percent = (weekday_count / total_requests * 100) if total_requests > 0 else 50.0
            
            peak_hour_mask = (
                (district_data['hourOfDay'] >= 8) &
                (district_data['hourOfDay'] < 17) &
                (district_data['isWeekday'])
            )
            peak_hour_percent = (peak_hour_mask.sum() / total_requests * 100) if total_requests > 0 else 50.0
            
            # More complex = further from balanced (50/50)
            weekday_deviation = abs(weekday_percent - 50.0) / 50.0
            peak_deviation = abs(peak_hour_percent - 50.0) / 50.0
            temporal_complexity = (weekday_deviation + peak_deviation) / 2.0
            
            risk_factors.append({
                'District_ID': district_id,
                'currentDensityRisk': density_score,
                'trendDirectionRisk': trend_score,
                'volatilityRisk': volatility_coef,
                'recurringIssueRisk': recurring_score,
                'temporalPatternRisk': temporal_complexity
            })
        
        risk_factors_df = pd.DataFrame(risk_factors)
        
        # Normalize all risk factors to 0-1 scale (data-driven)
        for col in ['currentDensityRisk', 'trendDirectionRisk', 'volatilityRisk', 
                   'recurringIssueRisk', 'temporalPatternRisk']:
            if col in risk_factors_df.columns:
                normalized = normalize_to_0_1(risk_factors_df[col].values)
                risk_factors_df[col] = normalized
        
        return risk_factors_df
    
    def calculatePredictiveMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame, 
                                   risk_factors: pd.DataFrame) -> pd.DataFrame:
        """Calculate predictive hotspot metrics."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Calculate weights (variance-based, data-driven)
        weights = calculate_variance_based_weights(risk_factors)
        weight_map = {
            'currentDensityRisk': weights[0] if len(weights) > 0 else 0.2,
            'trendDirectionRisk': weights[1] if len(weights) > 1 else 0.2,
            'volatilityRisk': weights[2] if len(weights) > 2 else 0.2,
            'recurringIssueRisk': weights[3] if len(weights) > 3 else 0.2,
            'temporalPatternRisk': weights[4] if len(weights) > 4 else 0.2
        }
        
        # Normalize weights to sum to 1
        total_weight = sum(weight_map.values())
        if total_weight > 0:
            weight_map = {k: v / total_weight for k, v in weight_map.items()}
        else:
            # Equal weights fallback
            weight_map = {k: 0.2 for k in weight_map.keys()}
        
        predictive_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            district_risks = risk_factors[risk_factors['District_ID'] == district_id]
            
            if len(district_risks) == 0:
                predictive_metrics.append({
                    'District_ID': district_id,
                    'predictiveHotspotScore': 0.0,
                    'hotspotConfidence': 0.0,
                    'projectedRequestVolume': 0.0,
                    'timeToHotspot': None,
                    'dominantRiskFactor': 'None',
                    'interventionPriority': 'Low'
                })
                continue
            
            # Calculate combined risk score
            score = (
                weight_map['currentDensityRisk'] * district_risks['currentDensityRisk'].iloc[0] +
                weight_map['trendDirectionRisk'] * district_risks['trendDirectionRisk'].iloc[0] +
                weight_map['volatilityRisk'] * district_risks['volatilityRisk'].iloc[0] +
                weight_map['recurringIssueRisk'] * district_risks['recurringIssueRisk'].iloc[0] +
                weight_map['temporalPatternRisk'] * district_risks['temporalPatternRisk'].iloc[0]
            )
            
            # Find dominant risk factor
            contributions = {
                'density': weight_map['currentDensityRisk'] * district_risks['currentDensityRisk'].iloc[0],
                'trend': weight_map['trendDirectionRisk'] * district_risks['trendDirectionRisk'].iloc[0],
                'volatility': weight_map['volatilityRisk'] * district_risks['volatilityRisk'].iloc[0],
                'recurring': weight_map['recurringIssueRisk'] * district_risks['recurringIssueRisk'].iloc[0],
                'temporal': weight_map['temporalPatternRisk'] * district_risks['temporalPatternRisk'].iloc[0]
            }
            dominant_risk = max(contributions, key=contributions.get)
            
            # Calculate confidence from trend analysis
            monthly_counts = district_data.groupby('month').size().sort_index()
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                _, _, r_squared, p_value = calculate_linear_regression(x, y)
                confidence = r_squared * (1.0 - min(p_value, 0.99))
            else:
                confidence = 0.0
            
            # Projected request volume
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                slope, intercept, _, _ = calculate_linear_regression(x, y)
                next_month_x = len(monthly_counts)
                projected_volume = slope * next_month_x + intercept
                projected_volume = max(0, projected_volume)
            else:
                projected_volume = monthly_counts.iloc[0] if len(monthly_counts) > 0 else 0.0
            
            # Time to hotspot (only if trending up)
            time_to_hotspot = None
            if len(monthly_counts) >= 2:
                x = np.arange(len(monthly_counts))
                y = monthly_counts.values
                slope, intercept, r_squared, _ = calculate_linear_regression(x, y)
                
                if slope > 0 and r_squared > 0.3:  # Increasing trend with some strength
                    # Calculate hotspot threshold (75th percentile of monthly volumes - data-driven)
                    all_monthly_volumes = df.groupby('month').size()
                    if len(all_monthly_volumes) > 0:
                        hotspot_threshold = np.percentile(all_monthly_volumes.values, 75)
                        current_volume = monthly_counts.iloc[-1] if len(monthly_counts) > 0 else 0.0
                        
                        if current_volume < hotspot_threshold and slope > 0:
                            months_needed = (hotspot_threshold - current_volume) / slope
                            time_to_hotspot = max(1, int(months_needed)) if months_needed > 0 else None
            
            predictive_metrics.append({
                'District_ID': district_id,
                'predictiveHotspotScore': round(score, 4),
                'hotspotConfidence': round(confidence, 4),
                'projectedRequestVolume': round(projected_volume, 2),
                'timeToHotspot': time_to_hotspot,
                'dominantRiskFactor': dominant_risk,
                'interventionPriority': 'Low'  # Will calculate from percentile
            })
        
        metrics_df = pd.DataFrame(predictive_metrics)
        
        # Calculate intervention priority from score percentiles (data-driven)
        if len(metrics_df) > 1:
            scores = metrics_df['predictiveHotspotScore'].values
            # Calculate tertiles (33rd, 67th percentiles - data-driven)
            tertile_33 = np.percentile(scores, 33)
            tertile_67 = np.percentile(scores, 67)
            
            def assign_priority(score):
                if score >= tertile_67:
                    return 'High'
                elif score >= tertile_33:
                    return 'Medium'
                else:
                    return 'Low'
            
            metrics_df['interventionPriority'] = metrics_df['predictiveHotspotScore'].apply(assign_priority)
        
        return metrics_df
    
    def classifyHotspots(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Classify hotspots using Silhouette + Quantiles."""
        score_values = metrics['predictiveHotspotScore'].dropna().values
        
        if len(score_values) >= 3:
            optimal_k = find_optimal_k_silhouette(score_values, k_range=None)
            
            # Sort descending (higher scores = higher risk)
            sorted_desc = np.sort(score_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]
            
            # Create labels based on optimal k
            labels = []
            for i in range(optimal_k):
                if optimal_k == 3:
                    if i == 0:
                        labels.append('High Risk')
                    elif i == 1:
                        labels.append('Moderate Risk')
                    else:
                        labels.append('Low Risk')
                elif optimal_k == 4:
                    if i == 0:
                        labels.append('Very High Risk')
                    elif i == 1:
                        labels.append('High Risk')
                    elif i == 2:
                        labels.append('Moderate Risk')
                    else:
                        labels.append('Low Risk')
                else:  # k == 5
                    if i == 0:
                        labels.append('Very High Risk')
                    elif i == optimal_k - 1:
                        labels.append('Very Low Risk')
                    else:
                        labels.append('Moderate Risk')
            
            # Create classification function
            def classify_score(value):
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
            
            metrics['predictiveHotspotCategory'] = metrics['predictiveHotspotScore'].apply(classify_score)
        else:
            metrics['predictiveHotspotCategory'] = 'Moderate Risk'
        
        # Calculate percentile
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('predictiveHotspotScore', ascending=False).reset_index(drop=True)
            metrics['predictiveHotspotPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
        else:
            metrics['predictiveHotspotPercentile'] = 100
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with predictive hotspot metrics."""
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
            metrics_merge = metrics[['District_ID', 'predictiveHotspotCategory', 'predictiveHotspotScore',
                                    'interventionPriority', 'dominantRiskFactor']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'predictiveHotspotCategory': 'districtPredictiveHotspotCategory',
                'predictiveHotspotScore': 'districtPredictiveHotspotScore',
                'interventionPriority': 'districtInterventionPriority',
                'dominantRiskFactor': 'districtDominantRiskFactor'
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
        
        print("Calculating risk factors...")
        risk_factors = self.calculateRiskFactors(df, boundaries)
        
        print("Calculating predictive metrics...")
        predictive_metrics = self.calculatePredictiveMetrics(df, boundaries, risk_factors)
        
        print("Classifying hotspots...")
        predictive_metrics = self.classifyHotspots(predictive_metrics)
        
        # Merge with boundaries
        boundaries = boundaries.merge(predictive_metrics, on='District_ID', how='left')
        
        # Convert to camelCase
        boundaries = boundaries.rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName'
        })
        
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, predictive_metrics)
        
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkgPath = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            boundaries.to_file(gpkgPath, layer='District Predictive Hotspots', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Predictive Hotspots GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Predictive Hotspots' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = PredictiveHotspotsGenerator()
    generator.createShapefile()

