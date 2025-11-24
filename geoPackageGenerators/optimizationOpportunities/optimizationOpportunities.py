#!/usr/bin/env python3
"""
Optimization Opportunities GeoPackage Generator

Creates a GeoPackage identifying districts with the biggest improvement potential
and where optimization efforts would have the greatest impact. Combines multiple
performance gaps to prioritize optimization efforts.

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


class OptimizationOpportunitiesGenerator:
    """Generator for optimization opportunities GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/optimizationOpportunities'
        self.outputName = 'optimizationOpportunities.gpkg'
        
        # Conversion constants (standard conversion factors - not hardcoded thresholds)
        self.SQ_M_TO_SQ_KM = 1_000_000
        self.SQ_KM_TO_SQ_MI = 0.386102
    
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
    
    def calculateResponseTimeMetrics(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate response time metrics by district (median-based)."""
        # Calculate response times for closed requests
        closed = df[df['Status'] == 'Closed'].copy()
        closed = closed[closed['Date_Time_Opened'].notna() & closed['Date_Time_Closed'].notna()]
        
        closed['openedDt'] = pd.to_datetime(closed['Date_Time_Opened'], unit='ms', errors='coerce')
        closed['closedDt'] = pd.to_datetime(closed['Date_Time_Closed'], unit='ms', errors='coerce')
        closed = closed[closed['openedDt'].notna() & closed['closedDt'].notna()]
        
        closed['responseHours'] = (closed['closedDt'] - closed['openedDt']).dt.total_seconds() / 3600.0
        
        # Convert district to numeric
        closed['districtInt'] = pd.to_numeric(closed['Council_District'], errors='coerce').astype('Int64')
        
        # Aggregate by district using MEDIAN (robust to outliers)
        response_metrics = closed.groupby('districtInt')['responseHours'].agg(['median', 'count']).rename(columns={
            'median': 'medianResponseHours',
            'count': 'closedRequests'
        })
        
        response_metrics = response_metrics.reset_index().rename(columns={'districtInt': 'District_ID'})
        
        return response_metrics
    
    def calculateWorkloadMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate workload metrics by district."""
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Count requests per district
        requestCounts = df.groupby('districtInt', dropna=True).size().rename('totalRequests').to_frame()
        requestCounts = requestCounts.reset_index().rename(columns={'districtInt': 'District_ID'})
        
        # Calculate area if not already in boundaries
        if 'Area_Square_Miles' not in boundaries.columns:
            boundaries_projected = boundaries.to_crs('EPSG:3857')
            boundaries['Area_Square_Meters'] = boundaries_projected.geometry.area
            boundaries['Area_Square_Kilometers'] = boundaries['Area_Square_Meters'] / self.SQ_M_TO_SQ_KM
            boundaries['Area_Square_Miles'] = boundaries['Area_Square_Kilometers'] / self.SQ_KM_TO_SQ_MI
        
        # Merge with area data
        areaData = boundaries[['District_ID', 'Area_Square_Miles']].copy()
        workload_metrics = requestCounts.merge(areaData, on='District_ID', how='inner')
        
        # Calculate requests per square mile
        workload_metrics['requestsPerSqMile'] = (
            workload_metrics['totalRequests'] / workload_metrics['Area_Square_Miles']
        )
        
        return workload_metrics
    
    def calculateOptimizationGaps(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate optimization opportunity gaps for each district."""
        # Calculate response time metrics (median-based)
        response_metrics = self.calculateResponseTimeMetrics(df)
        
        # Calculate workload metrics
        workload_metrics = self.calculateWorkloadMetrics(df, boundaries)
        
        # Merge metrics
        all_districts = boundaries[['District_ID']].copy()
        metrics = all_districts.merge(response_metrics, on='District_ID', how='left')
        metrics = metrics.merge(workload_metrics, on='District_ID', how='left')
        
        # Fill missing values
        metrics = metrics.fillna({
            'medianResponseHours': 0.0,
            'closedRequests': 0,
            'totalRequests': 0,
            'requestsPerSqMile': 0.0
        })
        
        # Calculate efficiency (workload / response time ratio)
        # Higher efficiency = better (responds fast despite high workload)
        city_avg_response = metrics['medianResponseHours'].median() if len(metrics) > 0 else 1.0
        city_avg_workload = metrics['requestsPerSqMile'].median() if len(metrics) > 0 else 1.0
        
        if city_avg_response > 0 and city_avg_workload > 0:
            metrics['responseRatio'] = metrics['medianResponseHours'] / city_avg_response
            metrics['workloadRatio'] = metrics['requestsPerSqMile'] / city_avg_workload
            
            # Efficiency = Workload Ratio / Response Time Ratio
            metrics['efficiency'] = (
                metrics['workloadRatio'] / metrics['responseRatio']
            ).replace([float('inf'), float('-inf')], 0.0)
        else:
            metrics['responseRatio'] = 0.0
            metrics['workloadRatio'] = 0.0
            metrics['efficiency'] = 0.0
        
        # Find benchmarks (best performers) - data-driven
        valid_response = metrics[metrics['medianResponseHours'] > 0]
        valid_efficiency = metrics[metrics['efficiency'] > 0]
        
        if len(valid_response) > 0:
            best_response_median = valid_response['medianResponseHours'].min()  # Best = lowest
        else:
            best_response_median = 1.0
        
        if len(valid_efficiency) > 0:
            best_efficiency = valid_efficiency['efficiency'].max()  # Best = highest
        else:
            best_efficiency = 1.0
        
        # Calculate response time gap (how much slower vs. best)
        # Gap = (district - best) / best
        if best_response_median > 0:
            metrics['responseTimeGap'] = (
                (metrics['medianResponseHours'] - best_response_median) / best_response_median
            )
            # Set to 0 for districts with no data
            metrics.loc[metrics['medianResponseHours'] == 0, 'responseTimeGap'] = 0.0
        else:
            metrics['responseTimeGap'] = 0.0
        
        # Calculate efficiency gap (how much less efficient vs. best)
        # Gap = (best - district) / best
        if best_efficiency > 0:
            metrics['efficiencyGap'] = (
                (best_efficiency - metrics['efficiency']) / best_efficiency
            )
            # Set to 0 for districts with no data
            metrics.loc[metrics['efficiency'] == 0, 'efficiencyGap'] = 0.0
        else:
            metrics['efficiencyGap'] = 0.0
        
        # Calculate workload pressure (from capacity utilization)
        # Use median monthly volume as capacity indicator
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        df['month'] = df['openedDt'].dt.to_period('M').astype(str)
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        monthly_volumes = []
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            if len(district_data) > 0:
                monthly_counts = district_data.groupby('month').size()
                if len(monthly_counts) > 0:
                    monthly_volumes.append({
                        'District_ID': district_id,
                        'medianMonthlyVolume': monthly_counts.median()
                    })
        
        if len(monthly_volumes) > 0:
            monthly_df = pd.DataFrame(monthly_volumes)
            metrics = metrics.merge(monthly_df, on='District_ID', how='left')
            metrics['medianMonthlyVolume'] = metrics['medianMonthlyVolume'].fillna(0.0)
            
            # Calculate capacity threshold (median-based)
            all_volumes = monthly_df['medianMonthlyVolume'].values
            if len(all_volumes) > 0:
                median_volume = np.median(all_volumes)
                q3 = np.percentile(all_volumes, 75)
                iqr = q3 - np.percentile(all_volumes, 25)
                capacity_threshold = max(median_volume, q3 + 1.5 * iqr) if iqr > 0 else median_volume
                
                # Workload pressure = utilization (0-1, can exceed 1.0)
                metrics['workloadPressure'] = (
                    metrics['medianMonthlyVolume'] / capacity_threshold
                    if capacity_threshold > 0 else 0.0
                )
            else:
                metrics['workloadPressure'] = 0.0
        else:
            metrics['workloadPressure'] = 0.0
        
        # Calculate volatility impact (coefficient of variation)
        volatility_impact = []
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            if len(district_data) > 0:
                monthly_counts = district_data.groupby('month').size()
                if len(monthly_counts) >= 2:
                    cv = np.std(monthly_counts.values) / np.mean(monthly_counts.values) if np.mean(monthly_counts.values) > 0 else 0.0
                    # Higher volatility = harder to optimize = lower opportunity
                    # Impact = 1 - normalized_volatility (inverse relationship)
                    volatility_impact.append({
                        'District_ID': district_id,
                        'volatility': cv
                    })
                else:
                    volatility_impact.append({
                        'District_ID': district_id,
                        'volatility': 0.0
                    })
            else:
                volatility_impact.append({
                    'District_ID': district_id,
                    'volatility': 0.0
                })
        
        volatility_df = pd.DataFrame(volatility_impact)
        metrics = metrics.merge(volatility_df, on='District_ID', how='left')
        metrics['volatility'] = metrics['volatility'].fillna(0.0)
        
        # Normalize volatility to 0-1, then invert (lower volatility = higher opportunity)
        if len(metrics) > 0 and metrics['volatility'].max() > metrics['volatility'].min():
            normalized_volatility = normalize_to_0_1(metrics['volatility'].values)
            metrics['volatilityImpact'] = 1.0 - normalized_volatility  # Invert: lower volatility = higher impact
        else:
            metrics['volatilityImpact'] = 1.0  # Default to high impact if no variation
        
        # Normalize gaps to 0-1 for consistent scaling
        if len(metrics) > 0:
            metrics['responseTimeGap'] = normalize_to_0_1(metrics['responseTimeGap'].values)
            metrics['efficiencyGap'] = normalize_to_0_1(metrics['efficiencyGap'].values)
            metrics['workloadPressure'] = np.clip(normalize_to_0_1(metrics['workloadPressure'].values), 0, 1)
        
        return metrics
    
    def calculateOptimizationScores(self, gaps_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate combined optimization opportunity scores."""
        # Prepare data for weight calculation
        weight_data = gaps_df[['District_ID', 'responseTimeGap', 'efficiencyGap', 
                               'workloadPressure', 'volatilityImpact']].copy()
        
        # Calculate variance-based weights (data-driven)
        weights = calculate_variance_based_weights(
            weight_data[['responseTimeGap', 'efficiencyGap', 'workloadPressure', 'volatilityImpact']]
        )
        
        # Normalize weights to sum to 1
        total_weight = sum(weights)
        if total_weight > 0:
            weights = [w / total_weight for w in weights]
        else:
            weights = [0.25, 0.25, 0.25, 0.25]  # Equal weights fallback
        
        # Calculate optimization opportunity score (weighted sum)
        gaps_df['optimizationOpportunityScore'] = (
            weights[0] * gaps_df['responseTimeGap'] +
            weights[1] * gaps_df['efficiencyGap'] +
            weights[2] * gaps_df['workloadPressure'] +
            weights[3] * gaps_df['volatilityImpact']
        )
        
        # Calculate optimization impact (similar to opportunity, but emphasizes high workload)
        # Impact = opportunity score weighted by actual workload
        gaps_df['optimizationImpact'] = (
            gaps_df['optimizationOpportunityScore'] * 
            (0.5 + 0.5 * gaps_df['workloadPressure'])  # Boost impact for high workload districts
        )
        
        # Normalize to 0-1
        gaps_df['optimizationImpact'] = normalize_to_0_1(gaps_df['optimizationImpact'].values)
        
        return gaps_df
    
    def classifyOptimizationPriority(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Classify optimization priority using percentiles (data-driven)."""
        score_values = metrics['optimizationOpportunityScore'].dropna().values
        
        if len(score_values) >= 3:
            # Use tertiles (33rd, 67th percentiles) for High/Medium/Low
            tertile_33 = np.percentile(score_values, 33)
            tertile_67 = np.percentile(score_values, 67)
            
            def assign_priority(score):
                if pd.isna(score):
                    return 'Low'
                if score >= tertile_67:
                    return 'High'
                elif score >= tertile_33:
                    return 'Medium'
                else:
                    return 'Low'
            
            metrics['optimizationPriority'] = metrics['optimizationOpportunityScore'].apply(assign_priority)
        else:
            metrics['optimizationPriority'] = 'Low'
        
        return metrics
    
    def classifyOptimizationCategory(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Classify optimization category based on dominant gap."""
        def assign_category(row):
            if pd.isna(row['responseTimeGap']) or pd.isna(row['efficiencyGap']) or pd.isna(row['workloadPressure']):
                return 'Multi-Factor'
            
            gaps = {
                'Response Time': row['responseTimeGap'],
                'Efficiency': row['efficiencyGap'],
                'Workload': row['workloadPressure']
            }
            
            # Find dominant gap
            max_gap = max(gaps.values())
            dominant = [k for k, v in gaps.items() if v == max_gap][0]
            
            # Check if multiple gaps are similar (within 20% of max)
            similar_gaps = [k for k, v in gaps.items() if v >= 0.8 * max_gap]
            
            if len(similar_gaps) >= 2:
                return 'Multi-Factor'
            else:
                return dominant
        
        metrics['optimizationCategory'] = metrics.apply(assign_category, axis=1)
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, district_metrics: pd.DataFrame) -> gpd.GeoDataFrame:
        """Create point layer for service requests with district optimization metrics."""
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
        
        # Select only optimization-related columns to merge
        optimization_cols = [
            'districtInt', 'responseTimeGap', 'efficiencyGap', 'workloadPressure',
            'volatilityImpact', 'optimizationOpportunityScore', 'optimizationImpact',
            'optimizationPriority', 'optimizationCategory'
        ]
        available_cols = [col for col in optimization_cols if col in district_metrics_for_merge.columns]
        
        points_gdf = points_gdf.merge(
            district_metrics_for_merge[available_cols],
            on='districtInt',
            how='left'
        )
        
        # Rename columns to camelCase with district prefix
        rename_map = {
            'responseTimeGap': 'districtResponseTimeGap',
            'efficiencyGap': 'districtEfficiencyGap',
            'workloadPressure': 'districtWorkloadPressure',
            'volatilityImpact': 'districtVolatilityImpact',
            'optimizationOpportunityScore': 'districtOptimizationOpportunityScore',
            'optimizationImpact': 'districtOptimizationImpact',
            'optimizationPriority': 'districtOptimizationPriority',
            'optimizationCategory': 'districtOptimizationCategory'
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
        
        # Add district optimization columns
        district_optimization_cols = [col for col in points_gdf.columns if col.startswith('district')]
        essential_cols.extend(district_optimization_cols)
        
        available_essential = [col for col in essential_cols if col in points_gdf.columns]
        points_gdf = points_gdf[available_essential]
        
        return points_gdf
    
    def createShapefile(self) -> str:
        """Create the optimization opportunities GeoPackage."""
        print("Loading data from S3...")
        df = self.loadDataFromS3()
        print(f"Loaded {len(df):,} service requests")
        
        print("Loading district boundaries...")
        boundaries = self.loadDistrictBoundaries()
        print(f"Loaded {len(boundaries)} districts")
        
        print("Calculating optimization gaps...")
        optimization_metrics = self.calculateOptimizationGaps(df, boundaries)
        
        print("Calculating optimization scores...")
        optimization_metrics = self.calculateOptimizationScores(optimization_metrics)
        
        print("Classifying optimization priority...")
        optimization_metrics = self.classifyOptimizationPriority(optimization_metrics)
        
        print("Classifying optimization category...")
        optimization_metrics = self.classifyOptimizationCategory(optimization_metrics)
        
        # Select only essential columns for final output
        essential_cols = [
            'District_ID', 'responseTimeGap', 'efficiencyGap', 'workloadPressure',
            'volatilityImpact', 'optimizationOpportunityScore', 'optimizationImpact',
            'optimizationPriority', 'optimizationCategory'
        ]
        available_cols = [col for col in essential_cols if col in optimization_metrics.columns]
        optimization_metrics = optimization_metrics[available_cols]
        
        # Merge with boundaries
        boundaries['District_ID'] = boundaries['District_ID'].astype(int)
        optimization_metrics['District_ID'] = optimization_metrics['District_ID'].astype(int)
        
        district_gdf = boundaries.merge(
            optimization_metrics,
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
        points_gdf = self.createRequestPointsLayer(df, optimization_metrics)
        
        # Create GeoPackage
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkg_path = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            district_gdf.to_file(gpkg_path, layer='District Optimization Opportunities', driver='GPKG')
            
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
        print(f"✅ Optimization Opportunities GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")


if __name__ == '__main__':
    generator = OptimizationOpportunitiesGenerator()
    s3Url = generator.createShapefile()
    
    print(f"\n✅ Ready for ArcGIS Pro!")
    print(f"GeoPackage URL: {s3Url}")
    print(f"\nContains 2 layers:")
    print(f"1. 'District Optimization Opportunities' (polygons)")
    print(f"2. 'Service Requests' (points)")

