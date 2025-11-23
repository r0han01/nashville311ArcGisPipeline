#!/usr/bin/env python3
"""
Geographic Patterns GeoPackage Generator

Creates a GeoPackage analyzing spatial distribution patterns of service requests
across Nashville council districts. Focuses on WHERE requests are located and
HOW they are distributed geographically (density, clustering, concentration, variability).

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
import math
from datetime import datetime
from typing import Optional, Dict, Any, Tuple
from shapely.geometry import Point
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    from libpysal.weights import Queen
    LIBPYSAL_AVAILABLE = True
except ImportError:
    LIBPYSAL_AVAILABLE = False

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


def calculate_silhouette_score_1d(data, k):
    """
    Calculate silhouette score for 1D data using k-means-like clustering.
    Simplified version for 1D data classification.
    
    Args:
        data: Array of numeric values
        k: Number of clusters/classes
        
    Returns:
        Silhouette score (higher is better, range: -1 to 1)
    """
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
    This ensures:
    - Minimum 3 classes (interpretable, not binary)
    - Maximum 5 classes (interpretable for maps)
    - Adapts to dataset size (sqrt(n) and n/7)
    - Ensures minimum ~7 observations per class (n/7)
    
    Args:
        n: Sample size
        
    Returns:
        List of k values to test
    """
    k_min = 3  # Minimum interpretable (domain knowledge: not binary)
    k_max = min(5, max(3, int(np.sqrt(n)), int(n / 7)))  # Statistical + domain constraints
    k_max = max(k_min, k_max)  # Ensure k_max >= k_min
    return list(range(k_min, k_max + 1))


def find_optimal_k_silhouette(data, k_range=None):
    """
    Find optimal number of classes using silhouette score.
    
    Args:
        data: Array of numeric values
        k_range: Optional list of k values to test. If None, calculates data-driven range.
        
    Returns:
        Optimal k value
    """
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


def classify_with_quantiles(data, k, labels=None, metric_name=''):
    """
    Classify data into k quantile-based categories with percentile labels.
    
    Args:
        data: Array of numeric values
        k: Number of classes
        labels: Optional custom labels (default: percentile-based)
        metric_name: Name of metric for better labels (e.g., 'Concentration', 'Consistency')
        
    Returns:
        Tuple: (classify_function, breaks, labels)
    """
    data = np.array([x for x in data if not pd.isna(x)])
    
    if len(data) < k:
        k = max(2, len(data))
    
    # Calculate quantile breaks
    quantiles = np.linspace(0, 100, k + 1)
    breaks = np.percentile(data, quantiles)
    
    # Create percentile-based labels if not provided
    if labels is None:
        labels = []
        for i in range(k):
            pct_low = int(quantiles[i])
            pct_high = int(quantiles[i + 1])
            
            # Create descriptive labels based on position
            if i == 0:
                # Top category
                if k == 3:
                    labels.append(f'Top 33% {metric_name}'.strip())
                elif k == 4:
                    labels.append(f'Top 25% {metric_name}'.strip())
                elif k == 5:
                    labels.append(f'Top 20% {metric_name}'.strip())
                else:
                    labels.append(f'Top {100 - pct_low}% {metric_name}'.strip())
            elif i == k - 1:
                # Bottom category
                if k == 3:
                    labels.append(f'Bottom 33% {metric_name}'.strip())
                elif k == 4:
                    labels.append(f'Bottom 25% {metric_name}'.strip())
                elif k == 5:
                    labels.append(f'Bottom 20% {metric_name}'.strip())
                else:
                    labels.append(f'Bottom {pct_high}% {metric_name}'.strip())
            else:
                # Middle categories
                if k == 3:
                    labels.append(f'Middle 33% {metric_name}'.strip())
                elif k == 4:
                    if i == 1:
                        labels.append(f'Upper-Middle 25% {metric_name}'.strip())
                    else:
                        labels.append(f'Lower-Middle 25% {metric_name}'.strip())
                elif k == 5:
                    labels.append(f'{pct_low}-{pct_high}th Percentile {metric_name}'.strip())
                else:
                    labels.append(f'{pct_low}-{pct_high}th Percentile {metric_name}'.strip())
    
    # Create classification function
    def classify(value):
        if pd.isna(value):
            return 'Unknown'
        
        # Handle edge cases
        if value < breaks[0]:
            return labels[-1]  # Bottom category
        if value >= breaks[-1]:
            return labels[0]  # Top category
        
        # Find which quantile this value belongs to
        for i in range(k):
            if i == k - 1:
                # Last category includes the upper bound
                if value >= breaks[i]:
                    return labels[i]
            else:
                if value >= breaks[i] and value < breaks[i + 1]:
                    return labels[i]
        
        # Fallback to last category
        return labels[-1]
    
    return classify, breaks, labels


def jenks_breaks(data, n_classes=4):
    """
    Calculate Jenks Natural Breaks for optimal data classification.
    Finds break points that minimize within-group variance.
    
    Args:
        data: Array of numeric values
        n_classes: Number of classes to create
        
    Returns:
        List of break points (including min and max)
    """
    data = np.array(sorted([x for x in data if not pd.isna(x)]))
    
    if len(data) < n_classes:
        # If not enough data points, return simple breaks
        return np.linspace(data.min(), data.max(), n_classes + 1).tolist()
    
    # Jenks Natural Breaks algorithm
    mat1 = np.zeros((len(data) + 1, n_classes + 1))
    mat2 = np.zeros((len(data) + 1, n_classes + 1))
    
    for i in range(1, n_classes + 1):
        mat1[1, i] = 1
        mat2[1, i] = 0
        for j in range(2, len(data) + 1):
            mat2[j, i] = float('inf')
    
    v = 0.0
    for l in range(2, len(data) + 1):
        s1 = 0.0
        s2 = 0.0
        w = 0.0
        for m in range(1, l + 1):
            i3 = l - m + 1
            val = data[i3 - 1]
            s2 += val * val
            s1 += val
            w += 1
            v = s2 - (s1 * s1) / w
            i4 = i3 - 1
            if i4 != 0:
                for j in range(2, n_classes + 1):
                    if mat2[l, j] >= (v + mat2[i4, j - 1]):
                        mat1[l, j] = i3
                        mat2[l, j] = v + mat2[i4, j - 1]
        mat1[l, 1] = 1
        mat2[l, 1] = v
    
    # Extract break points
    k = len(data)
    kclass = []
    for j in range(n_classes, 0, -1):
        kclass.append(int(mat1[k, j]) - 1)
        k = int(mat1[k, j] - 1)
    
    kclass.reverse()
    breaks = [data[0]]
    for i in range(1, len(kclass)):
        breaks.append(data[kclass[i]])
    breaks.append(data[-1])
    
    return sorted(list(set(breaks)))


class GeographicPatternsGenerator:
    """Generator for district geographic patterns GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/geographicPatterns'
        self.outputName = 'geographicPatterns.gpkg'
        
        # Conversion constants
        self.SQ_M_TO_SQ_KM = 1_000_000
        self.SQ_KM_TO_SQ_MI = 0.386102
        self.M_TO_MI = 0.000621371
        
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
                s3Key = f'boundaries/nashvilleCouncilDistricts/{baseName}{ext}'
                localPath = os.path.join(tempDir, f'{baseName}{ext}')
                
                try:
                    self.s3Client.download_file(self.bucketName, s3Key, localPath)
                except Exception as e:
                    if ext == '.shp.xml':
                        continue
                    raise FileNotFoundError(f"Could not download {s3Key}: {e}")
            
            shapefilePath = os.path.join(tempDir, f'{baseName}.shp')
            boundaries = gpd.read_file(shapefilePath)
        
        # Determine district column
        if 'DISTRICT' in boundaries.columns:
            distCol = 'DISTRICT'
        elif 'District' in boundaries.columns:
            distCol = 'District'
        else:
            raise ValueError("Could not find district column in boundaries")
        
        # Rename to standard
        boundaries = boundaries.rename(columns={distCol: 'District_ID'})
        
        # Convert to WGS84
        if boundaries.crs is None:
            boundaries.set_crs('EPSG:2263', inplace=True)  # Tennessee State Plane
        
        boundaries = boundaries.to_crs('EPSG:4326')
        
        # Convert to projected CRS for area calculations
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        
        # Calculate area in square miles
        boundaries['area_sq_m'] = boundaries_proj.geometry.area
        boundaries['area_sq_km'] = boundaries['area_sq_m'] / self.SQ_M_TO_SQ_KM
        boundaries['area_sq_mi'] = boundaries['area_sq_km'] * self.SQ_KM_TO_SQ_MI
        
        # Calculate centroids (in WGS84 - already in 4326)
        boundaries['centroid'] = boundaries.geometry.centroid
        boundaries['centroid_x'] = boundaries['centroid'].x
        boundaries['centroid_y'] = boundaries['centroid'].y
        
        boundaries = boundaries.drop(columns=['centroid'])
        
        return boundaries
    
    def calculateDensityMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate density metrics by district (data-driven)."""
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Count requests per district
        requestCounts = df.groupby('districtInt', dropna=True).size().rename('totalRequests').to_frame()
        requestCounts = requestCounts.reset_index().rename(columns={'districtInt': 'District_ID'})
        
        # Merge with boundaries to get area
        metrics = boundaries[['District_ID', 'area_sq_mi']].merge(
            requestCounts,
            on='District_ID',
            how='left'
        ).fillna({'totalRequests': 0})
        
        # Calculate requests per square mile
        metrics['requestsPerSquareMile'] = (
            metrics['totalRequests'] / metrics['area_sq_mi']
        ).replace([float('inf'), float('-inf')], 0).fillna(0)
        
        # Calculate city average (data-driven)
        cityAvg = metrics['requestsPerSquareMile'].mean() if len(metrics) > 0 else 0
        metrics['cityAverageRequestsPerSquareMile'] = cityAvg
        
        # Calculate density ratio to city average
        metrics['densityRatioToCityAverage'] = (
            metrics['requestsPerSquareMile'] / cityAvg
            if cityAvg > 0 else 0
        )
        
        # Calculate density percentile (0-100, data-driven)
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('requestsPerSquareMile', ascending=False).reset_index(drop=True)
            metrics['densityRank'] = metrics.index + 1
            metrics['densityPercentile'] = ((n - metrics['densityRank']) / (n - 1) * 100).round().astype('Int64')
        else:
            metrics['densityRank'] = 1
            metrics['densityPercentile'] = 100
        
        # Calculate density quartiles (data-driven)
        if n >= 4:
            metrics['densityQuartile'] = pd.qcut(
                metrics['requestsPerSquareMile'],
                q=4,
                labels=['Q4', 'Q3', 'Q2', 'Q1'],  # Q1 = highest density
                duplicates='drop'
            )
        else:
            metrics['densityQuartile'] = 'Q2'
        
        return metrics
    
    def calculateClusteringMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate spatial clustering metrics (data-driven spatial statistics)."""
        # Create points GeoDataFrame
        points = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        if len(points) == 0:
            # Return empty metrics
            boundaries['clusterType'] = 'Not Significant'
            boundaries['clusterZScore'] = 0.0
            boundaries['clusterPValue'] = 1.0
            boundaries['neighborAverageDensity'] = 0.0
            boundaries['neighborDensityRatio'] = 0.0
            return boundaries[['District_ID', 'clusterType', 'clusterZScore', 'clusterPValue', 
                              'neighborAverageDensity', 'neighborDensityRatio']]
        
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points['Longitude'], points['Latitude']),
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS for distance calculations
        points_proj = points_gdf.to_crs('EPSG:3857')
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        
        # Spatial join to assign points to districts
        points_with_district = gpd.sjoin(points_proj, boundaries_proj[['District_ID', 'geometry']], 
                                        how='left', predicate='within')
        
        # Count points per district
        district_counts = points_with_district.groupby('District_ID').size().reset_index(name='pointCount')
        
        # Merge with boundaries
        metrics = boundaries[['District_ID']].merge(district_counts, on='District_ID', how='left').fillna({'pointCount': 0})
        
        # Calculate neighbor relationships (spatial adjacency)
        if LIBPYSAL_AVAILABLE:
            w = Queen.from_dataframe(boundaries_proj)
        else:
            # Fallback: use geopandas spatial relationships
            w = None
        
        # Calculate neighbor average density
        neighbor_avg_density = []
        neighbor_density_ratios = []
        
        for idx, district_id in enumerate(metrics['District_ID']):
            if w is not None:
                neighbors = w.neighbors[idx]
            else:
                # Fallback: find neighbors using spatial intersection
                current_geom = boundaries_proj.iloc[idx].geometry
                neighbors = []
                for n_idx in range(len(boundaries_proj)):
                    if n_idx != idx:
                        neighbor_geom = boundaries_proj.iloc[n_idx].geometry
                        if current_geom.touches(neighbor_geom) or current_geom.intersects(neighbor_geom):
                            neighbors.append(n_idx)
            
            if neighbors:
                neighbor_counts = [metrics.loc[metrics['District_ID'] == boundaries_proj.iloc[n]['District_ID'], 'pointCount'].values[0] 
                                 if n < len(boundaries_proj) else 0 for n in neighbors]
                neighbor_avg = np.mean(neighbor_counts) if neighbor_counts else 0
                current_count = metrics.loc[metrics['District_ID'] == district_id, 'pointCount'].values[0]
                neighbor_avg_density.append(neighbor_avg)
                neighbor_density_ratios.append(current_count / neighbor_avg if neighbor_avg > 0 else 0)
            else:
                neighbor_avg_density.append(0)
                neighbor_density_ratios.append(0)
        
        metrics['neighborAverageDensity'] = neighbor_avg_density
        metrics['neighborDensityRatio'] = neighbor_density_ratios
        
        # Calculate Z-scores first (needed for statistical significance)
        mean_count = metrics['pointCount'].mean()
        std_count = metrics['pointCount'].std()
        metrics['clusterZScore'] = ((metrics['pointCount'] - mean_count) / std_count 
                                   if std_count > 0 else 0).fillna(0)
        
        # P-values (simplified - based on Z-score)
        if SCIPY_AVAILABLE:
            metrics['clusterPValue'] = 1 - stats.norm.cdf(abs(metrics['clusterZScore']))
        else:
            # Fallback: approximate p-value
            metrics['clusterPValue'] = np.exp(-abs(metrics['clusterZScore']) / 2)
        
        # Statistical significance thresholds (data-driven: standard 95% confidence)
        significance_threshold = 0.05  # Standard statistical threshold
        z_score_threshold = 1.96  # Standard for 95% confidence interval
        
        # Calculate city average for comparison
        city_avg_density = np.mean(neighbor_avg_density) if neighbor_avg_density else 0
        
        # Classify based on statistical significance (data-driven)
        def classify_cluster_statistical(row):
            if pd.isna(row['pointCount']) or pd.isna(row['neighborAverageDensity']) or pd.isna(row['clusterPValue']) or pd.isna(row['clusterZScore']):
                return 'Not Significant'
            
            # Check statistical significance
            is_significant = row['clusterPValue'] < significance_threshold and abs(row['clusterZScore']) >= z_score_threshold
            
            if not is_significant:
                return 'Not Significant'
            
            # If significant, classify based on density comparison
            current_density = row['pointCount']
            neighbor_density = row['neighborAverageDensity']
            
            if current_density > neighbor_density:
                if neighbor_density > city_avg_density:
                    return 'Significant High Cluster'  # High-High
                else:
                    return 'Significant High Outlier'  # High-Low
            else:
                if neighbor_density < city_avg_density:
                    return 'Significant Low Cluster'  # Low-Low
                else:
                    return 'Significant Low Outlier'  # Low-High
        
        metrics['clusterType'] = metrics.apply(classify_cluster_statistical, axis=1)
        
        return metrics[['District_ID', 'clusterType', 'clusterZScore', 'clusterPValue',
                       'neighborAverageDensity', 'neighborDensityRatio']]
    
    def calculateConcentrationMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate concentration metrics (spatial spread of requests)."""
        # Create points GeoDataFrame
        points = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        if len(points) == 0:
            boundaries['requestConcentrationIndex'] = 0.0
            boundaries['standardDistance'] = 0.0
            boundaries['concentrationPercentile'] = 0
            boundaries['concentrationQuartile'] = 'Q2'
            boundaries['concentrationCategory'] = 'Moderately Scattered'
            return boundaries[['District_ID', 'requestConcentrationIndex', 'standardDistance',
                             'concentrationPercentile', 'concentrationQuartile', 'concentrationCategory']]
        
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points['Longitude'], points['Latitude']),
            crs='EPSG:4326'
        )
        
        # Convert to projected CRS
        points_proj = points_gdf.to_crs('EPSG:3857')
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        
        # Spatial join
        points_with_district = gpd.sjoin(points_proj, boundaries_proj[['District_ID', 'geometry']], 
                                        how='left', predicate='within')
        
        concentration_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_points = points_with_district[points_with_district['District_ID'] == district_id]
            
            if len(district_points) == 0:
                concentration_metrics.append({
                    'District_ID': district_id,
                    'requestConcentrationIndex': 0.0,
                    'standardDistance': 0.0
                })
                continue
            
            # Get district centroid
            district_geom = boundaries_proj[boundaries_proj['District_ID'] == district_id].geometry.iloc[0]
            centroid = district_geom.centroid
            
            # Calculate distances from centroid
            distances = [point.distance(centroid) for point in district_points.geometry]
            
            # Standard distance (standard deviation of distances)
            if len(distances) > 1:
                std_distance = np.std(distances) * self.M_TO_MI  # Convert to miles
            else:
                std_distance = 0.0
            
            # Concentration index (0-1, where 1 = all points at centroid, 0 = evenly spread)
            max_distance = max(distances) if distances else 0
            if max_distance > 0:
                concentration_index = 1 - (std_distance / (max_distance * self.M_TO_MI))
                concentration_index = max(0, min(1, concentration_index))
            else:
                concentration_index = 1.0
            
            concentration_metrics.append({
                'District_ID': district_id,
                'requestConcentrationIndex': concentration_index,
                'standardDistance': std_distance
            })
        
        metrics = pd.DataFrame(concentration_metrics)
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('requestConcentrationIndex', ascending=False).reset_index(drop=True)
            metrics['concentrationPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['concentrationQuartile'] = pd.qcut(
                    metrics['requestConcentrationIndex'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = most concentrated
                    duplicates='drop'
                )
            else:
                metrics['concentrationQuartile'] = 'Q2'
        else:
            metrics['concentrationPercentile'] = 100
            metrics['concentrationQuartile'] = 'Q1'
        
        # Category labels using Silhouette + Quantiles (fully data-driven)
        concentration_values = metrics['requestConcentrationIndex'].dropna().values
        if len(concentration_values) >= 3:
            # Find optimal k using silhouette score (k_range calculated from data size)
            optimal_k = find_optimal_k_silhouette(concentration_values, k_range=None)
            
            # For concentration, higher values = better, so we reverse the order
            # Calculate quantiles on sorted descending data
            sorted_desc = np.sort(concentration_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]  # Sort descending for breaks
            
            # Create labels (Top = highest values)
            labels = []
            for i in range(optimal_k):
                if optimal_k == 3:
                    if i == 0:
                        labels.append('Top 33% Concentration')
                    elif i == 1:
                        labels.append('Middle 33% Concentration')
                    else:
                        labels.append('Bottom 33% Concentration')
                elif optimal_k == 4:
                    if i == 0:
                        labels.append('Top 25% Concentration')
                    elif i == 1:
                        labels.append('Upper-Middle 25% Concentration')
                    elif i == 2:
                        labels.append('Lower-Middle 25% Concentration')
                    else:
                        labels.append('Bottom 25% Concentration')
                else:  # k == 5
                    if i == 0:
                        labels.append('Top 20% Concentration')
                    elif i == optimal_k - 1:
                        labels.append('Bottom 20% Concentration')
                    else:
                        labels.append(f'Middle-{i} Concentration')
            
            # Create classification function (higher values = top category)
            def classify_concentration(value):
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
            
            metrics['concentrationCategory'] = metrics['requestConcentrationIndex'].apply(classify_concentration)
        else:
            # Fallback if not enough data
            metrics['concentrationCategory'] = 'Moderate Concentration'
        
        return metrics[['District_ID', 'requestConcentrationIndex', 'standardDistance',
                       'concentrationPercentile', 'concentrationQuartile', 'concentrationCategory']]
    
    def calculateVariabilityMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate variability metrics (consistency of request patterns)."""
        # Group by district and calculate temporal/spatial variance
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        variability_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id]
            
            if len(district_data) == 0:
                variability_metrics.append({
                    'District_ID': district_id,
                    'spatialVariability': 0.0,
                    'requestPatternConsistency': 1.0
                })
                continue
            
            # Spatial variability (coefficient of variation of point distribution)
            points = district_data[district_data['Latitude'].notna() & district_data['Longitude'].notna()]
            
            if len(points) > 1:
                # Calculate distances between all points
                coords = list(zip(points['Longitude'], points['Latitude']))
                distances = []
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        # Haversine distance
                        lat1, lon1 = math.radians(coords[i][1]), math.radians(coords[i][0])
                        lat2, lon2 = math.radians(coords[j][1]), math.radians(coords[j][0])
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        distance = 3959 * c  # Earth radius in miles
                        distances.append(distance)
                
                if distances:
                    mean_dist = np.mean(distances)
                    std_dist = np.std(distances)
                    spatial_variability = (std_dist / mean_dist) if mean_dist > 0 else 0
                else:
                    spatial_variability = 0
            else:
                spatial_variability = 0
            
            # Pattern consistency (inverse of variability, normalized 0-1)
            request_pattern_consistency = 1 / (1 + spatial_variability) if spatial_variability > 0 else 1.0
            
            variability_metrics.append({
                'District_ID': district_id,
                'spatialVariability': spatial_variability,
                'requestPatternConsistency': request_pattern_consistency
            })
        
        metrics = pd.DataFrame(variability_metrics)
        
        # Calculate percentiles and quartiles (data-driven)
        n = len(metrics)
        if n > 1:
            metrics = metrics.sort_values('requestPatternConsistency', ascending=False).reset_index(drop=True)
            metrics['variabilityPercentile'] = ((n - metrics.index) / (n - 1) * 100).round().astype('Int64')
            
            if n >= 4:
                metrics['variabilityQuartile'] = pd.qcut(
                    metrics['requestPatternConsistency'],
                    q=4,
                    labels=['Q1', 'Q2', 'Q3', 'Q4'],  # Q1 = most consistent
                    duplicates='drop'
                )
            else:
                metrics['variabilityQuartile'] = 'Q2'
        else:
            metrics['variabilityPercentile'] = 100
            metrics['variabilityQuartile'] = 'Q1'
        
        # Category labels using Silhouette + Quantiles (fully data-driven)
        consistency_values = metrics['requestPatternConsistency'].dropna().values
        if len(consistency_values) >= 3:
            # Find optimal k using silhouette score (k_range calculated from data size)
            optimal_k = find_optimal_k_silhouette(consistency_values, k_range=None)
            
            # For consistency, higher values = better, so we reverse the order
            # Calculate quantiles on sorted descending data
            sorted_desc = np.sort(consistency_values)[::-1]
            quantiles = np.linspace(0, 100, optimal_k + 1)
            breaks = np.percentile(sorted_desc, quantiles)
            breaks = np.sort(breaks)[::-1]  # Sort descending for breaks
            
            # Create labels (Top = highest values = most consistent)
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
            
            # Create classification function (higher values = top category)
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
                return labels[0]  # Highest values go to top
            
            metrics['variabilityCategory'] = metrics['requestPatternConsistency'].apply(classify_consistency)
        else:
            # Fallback if not enough data
            metrics['variabilityCategory'] = 'Moderate Consistency'
        
        return metrics[['District_ID', 'spatialVariability', 'requestPatternConsistency',
                       'variabilityPercentile', 'variabilityQuartile', 'variabilityCategory']]
    
    def calculateSpatialRelationships(self, boundaries: gpd.GeoDataFrame, density_metrics: pd.DataFrame) -> pd.DataFrame:
        """Calculate spatial relationship metrics."""
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        
        if LIBPYSAL_AVAILABLE:
            w = Queen.from_dataframe(boundaries_proj)
        else:
            w = None
        
        spatial_metrics = []
        
        for idx, district_id in enumerate(boundaries['District_ID']):
            if w is not None:
                neighbors = w.neighbors[idx]
            else:
                # Fallback: find neighbors using spatial intersection
                current_geom = boundaries_proj.iloc[idx].geometry
                neighbors = []
                for n_idx in range(len(boundaries_proj)):
                    if n_idx != idx:
                        neighbor_geom = boundaries_proj.iloc[n_idx].geometry
                        if current_geom.touches(neighbor_geom) or current_geom.intersects(neighbor_geom):
                            neighbors.append(n_idx)
            
            adjacent_count = len(neighbors)
            
            # Count high/low density neighbors
            high_density = 0
            low_density = 0
            
            if neighbors:
                for n_idx in neighbors:
                    if n_idx < len(boundaries_proj):
                        neighbor_id = boundaries_proj.iloc[n_idx]['District_ID']
                        neighbor_density = density_metrics[density_metrics['District_ID'] == neighbor_id]['densityQuartile'].values
                        if len(neighbor_density) > 0:
                            if neighbor_density[0] in ['Q1', 'Q2']:
                                high_density += 1
                            elif neighbor_density[0] in ['Q3', 'Q4']:
                                low_density += 1
            
            # Spatial isolation (distance to nearest high-density district)
            current_density = density_metrics[density_metrics['District_ID'] == district_id]['densityQuartile'].values
            is_high_density = len(current_density) > 0 and current_density[0] in ['Q1', 'Q2']
            
            if is_high_density:
                isolation_index = 0.0  # Not isolated if itself is high density
            else:
                # Find nearest high-density district
                min_distance = float('inf')
                current_geom = boundaries_proj.iloc[idx].geometry
                
                for n_idx in range(len(boundaries_proj)):
                    if n_idx != idx:
                        neighbor_id = boundaries_proj.iloc[n_idx]['District_ID']
                        neighbor_density = density_metrics[density_metrics['District_ID'] == neighbor_id]['densityQuartile'].values
                        if len(neighbor_density) > 0 and neighbor_density[0] in ['Q1', 'Q2']:
                            neighbor_geom = boundaries_proj.iloc[n_idx].geometry
                            distance = current_geom.distance(neighbor_geom) * self.M_TO_MI
                            min_distance = min(min_distance, distance)
                
                isolation_index = min_distance if min_distance != float('inf') else 0.0
            
            spatial_metrics.append({
                'District_ID': district_id,
                'adjacentDistrictCount': adjacent_count,
                'adjacentDistrictsHighDensity': high_density,
                'adjacentDistrictsLowDensity': low_density,
                'spatialIsolationIndex': isolation_index
            })
        
        return pd.DataFrame(spatial_metrics)
    
    def calculateRequestDistributionMetrics(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate request distribution metrics."""
        points = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        if len(points) == 0:
            boundaries['requestCentroidX'] = 0.0
            boundaries['requestCentroidY'] = 0.0
            boundaries['distanceFromDistrictCentroid'] = 0.0
            boundaries['requestSpread'] = 0.0
            return boundaries[['District_ID', 'requestCentroidX', 'requestCentroidY',
                              'distanceFromDistrictCentroid', 'requestSpread']]
        
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points['Longitude'], points['Latitude']),
            crs='EPSG:4326'
        )
        
        boundaries_proj = boundaries.to_crs('EPSG:3857')
        points_proj = points_gdf.to_crs('EPSG:3857')
        
        points_with_district = gpd.sjoin(points_proj, boundaries_proj[['District_ID', 'geometry']], 
                                        how='left', predicate='within')
        
        distribution_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_points = points_with_district[points_with_district['District_ID'] == district_id]
            district_boundary = boundaries_proj[boundaries_proj['District_ID'] == district_id]
            
            if len(district_points) == 0:
                distribution_metrics.append({
                    'District_ID': district_id,
                    'requestCentroidX': 0.0,
                    'requestCentroidY': 0.0,
                    'distanceFromDistrictCentroid': 0.0,
                    'requestSpread': 0.0
                })
                continue
            
            # Request centroid
            request_centroid = district_points.geometry.centroid.iloc[0]
            request_centroid_gdf = gpd.GeoDataFrame(geometry=[request_centroid], crs='EPSG:3857')
            request_centroid_4326 = request_centroid_gdf.to_crs('EPSG:4326').geometry.iloc[0]
            
            # District centroid
            district_centroid = district_boundary.geometry.centroid.iloc[0]
            district_centroid_gdf = gpd.GeoDataFrame(geometry=[district_centroid], crs='EPSG:3857')
            district_centroid_4326 = district_centroid_gdf.to_crs('EPSG:4326').geometry.iloc[0]
            
            # Distance between centroids
            lat1, lon1 = math.radians(request_centroid_4326.y), math.radians(request_centroid_4326.x)
            lat2, lon2 = math.radians(district_centroid_4326.y), math.radians(district_centroid_4326.x)
            dlat = lat2 - lat1
            dlon = lon2 - lon1
            a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
            c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
            distance = 3959 * c  # Miles
            
            # Request spread (max distance between requests)
            if len(district_points) > 1:
                coords = [(p.x, p.y) for p in district_points.geometry]
                max_dist = 0
                for i in range(len(coords)):
                    for j in range(i+1, len(coords)):
                        dist = np.sqrt((coords[i][0] - coords[j][0])**2 + (coords[i][1] - coords[j][1])**2) * self.M_TO_MI
                        max_dist = max(max_dist, dist)
            else:
                max_dist = 0.0
            
            distribution_metrics.append({
                'District_ID': district_id,
                'requestCentroidX': request_centroid_4326.x,
                'requestCentroidY': request_centroid_4326.y,
                'distanceFromDistrictCentroid': distance,
                'requestSpread': max_dist
            })
        
        return pd.DataFrame(distribution_metrics)
    
    def createCombinedPatternType(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Create combined geographic pattern type using median splits (data-driven)."""
        # Calculate medians from actual data (data-driven)
        density_median = metrics['requestsPerSquareMile'].median() if 'requestsPerSquareMile' in metrics.columns else 0
        concentration_median = metrics['requestConcentrationIndex'].median() if 'requestConcentrationIndex' in metrics.columns else 0.5
        consistency_median = metrics['requestPatternConsistency'].median() if 'requestPatternConsistency' in metrics.columns else 0.5
        
        def get_pattern_type_median(row):
            # Compare to medians (data-driven thresholds)
            density_val = row.get('requestsPerSquareMile', 0)
            concentration_val = row.get('requestConcentrationIndex', 0.5)
            consistency_val = row.get('requestPatternConsistency', 0.5)
            
            # Median splits (data-driven)
            density_desc = 'High Density' if (not pd.isna(density_val) and density_val >= density_median) else 'Low Density'
            concentration_desc = 'Concentrated' if (not pd.isna(concentration_val) and concentration_val >= concentration_median) else 'Scattered'
            consistency_desc = 'Consistent' if (not pd.isna(consistency_val) and consistency_val >= consistency_median) else 'Variable'
            
            return f"{density_desc} + {concentration_desc} + {consistency_desc}"
        
        metrics['geographicPatternType'] = metrics.apply(get_pattern_type_median, axis=1)
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, district_metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create service request points layer with district pattern metrics."""
        points = df[df['Latitude'].notna() & df['Longitude'].notna()].copy()
        
        if len(points) == 0:
            return gpd.GeoDataFrame(geometry=[], crs='EPSG:4326')
        
        points_gdf = gpd.GeoDataFrame(
            points,
            geometry=gpd.points_from_xy(points['Longitude'], points['Latitude']),
            crs='EPSG:4326'
        )
        
        # Rename base columns to camelCase
        rename_map = {
            'Request__': 'requestId',
            'Request_Type': 'requestType',
            'Subrequest_Type': 'subrequestType',
            'Status': 'status',
            'Address': 'address',
            'City': 'city',
            'Council_District': 'councilDistrict',
            'ZIP': 'zipCode',
            'Date_Time_Opened': 'dateTimeOpened',
            'Date_Time_Closed': 'dateTimeClosed',
            'Latitude': 'latitude',
            'Longitude': 'longitude'
        }
        
        rename_map = {k: v for k, v in rename_map.items() if k in points_gdf.columns}
        points_gdf = points_gdf.rename(columns=rename_map)
        
        # Add latitude and longitude from geometry if not already present
        if 'geometry' in points_gdf.columns:
            if 'latitude' not in points_gdf.columns:
                points_gdf['latitude'] = points_gdf.geometry.y
            if 'longitude' not in points_gdf.columns:
                points_gdf['longitude'] = points_gdf.geometry.x
        
        # Link district pattern metrics
        if district_metrics is not None:
            # Use districtId for merge (already in camelCase)
            points_gdf['districtInt'] = pd.to_numeric(points_gdf['councilDistrict'], errors='coerce').astype('Int64')
            
            # Prepare district metrics for merge - use districtId as key
            merge_cols = ['districtId', 'densityQuartile', 'clusterType', 'concentrationCategory', 
                         'variabilityCategory', 'districtCentroidX', 'districtCentroidY']
            available_merge_cols = [col for col in merge_cols if col in district_metrics.columns]
            
            district_subset = district_metrics[available_merge_cols].copy()
            
            # Rename columns for points layer
            rename_map = {
                'densityQuartile': 'districtDensityQuartile',
                'clusterType': 'districtClusterType',
                'concentrationCategory': 'districtConcentrationCategory',
                'variabilityCategory': 'districtVariabilityCategory'
            }
            
            district_subset = district_subset.rename(columns=rename_map)
            
            points_gdf = points_gdf.merge(
                district_subset,
                left_on='districtInt',
                right_on='districtId',
                how='left'
            )
            
            # Calculate distance from district centroid for each point
            if 'latitude' in points_gdf.columns and 'longitude' in points_gdf.columns:
                if 'districtCentroidX' in points_gdf.columns and 'districtCentroidY' in points_gdf.columns:
                    # Calculate distance using Haversine formula
                    def haversine_distance(row):
                        if pd.isna(row['latitude']) or pd.isna(row['longitude']) or pd.isna(row['districtCentroidY']) or pd.isna(row['districtCentroidX']):
                            return None
                        lat1, lon1 = math.radians(row['latitude']), math.radians(row['longitude'])
                        lat2, lon2 = math.radians(row['districtCentroidY']), math.radians(row['districtCentroidX'])
                        dlat = lat2 - lat1
                        dlon = lon2 - lon1
                        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
                        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
                        return 3959 * c  # Earth radius in miles
                    
                    points_gdf['distanceFromDistrictCentroid'] = points_gdf.apply(haversine_distance, axis=1)
            
            # Clean up merge columns
            cols_to_drop = ['districtId', 'districtInt', 'districtCentroidX', 'districtCentroidY']
            for col in cols_to_drop:
                if col in points_gdf.columns:
                    points_gdf = points_gdf.drop(columns=[col])
        
        return points_gdf
    
    def uploadGeoPackageToS3(self, localGpkgPath: str) -> Dict[str, str]:
        """Upload GeoPackage file to S3."""
        s3Key = f'{self.s3ShapefilePrefix}/{self.outputName}'
        
        self.s3Client.upload_file(
            localGpkgPath,
            self.bucketName,
            s3Key,
            ExtraArgs={'ContentType': 'application/geopackage+sqlite3'}
        )
        
        publicUrl = f'https://{self.bucketName}.s3.amazonaws.com/{s3Key}'
        
        return {
            'gpkg': publicUrl,
            'mainUrl': publicUrl
        }
    
    def createShapefile(self, fileName: Optional[str] = None) -> Dict[str, Any]:
        """Create the geographic patterns GeoPackage and upload to S3."""
        print("Loading data from S3...")
        df = self.loadDataFromS3(fileName)
        print(f"Loaded {len(df):,} service requests")
        
        print("Loading district boundaries...")
        boundaries = self.loadDistrictBoundaries()
        print(f"Loaded {len(boundaries)} districts")
        
        print("Calculating density metrics...")
        density_metrics = self.calculateDensityMetrics(df, boundaries)
        
        print("Calculating clustering metrics...")
        clustering_metrics = self.calculateClusteringMetrics(df, boundaries)
        
        print("Calculating concentration metrics...")
        concentration_metrics = self.calculateConcentrationMetrics(df, boundaries)
        
        print("Calculating variability metrics...")
        variability_metrics = self.calculateVariabilityMetrics(df, boundaries)
        
        print("Calculating spatial relationships...")
        spatial_metrics = self.calculateSpatialRelationships(boundaries, density_metrics)
        
        print("Calculating request distribution metrics...")
        distribution_metrics = self.calculateRequestDistributionMetrics(df, boundaries)
        
        # Merge all metrics
        print("Merging all metrics...")
        # Get available columns from boundaries
        boundary_cols = ['District_ID', 'area_sq_mi', 'centroid_x', 'centroid_y', 'geometry']
        if 'District_Name' in boundaries.columns:
            boundary_cols.append('District_Name')
        if 'Representative_Name' in boundaries.columns:
            boundary_cols.append('Representative_Name')
        if 'DISTRICT_NAME' in boundaries.columns:
            boundary_cols.append('DISTRICT_NAME')
        if 'REPRESENTATIVE' in boundaries.columns:
            boundary_cols.append('REPRESENTATIVE')
        
        all_metrics = boundaries[boundary_cols].copy()
        
        # Standardize column names
        if 'DISTRICT_NAME' in all_metrics.columns:
            all_metrics = all_metrics.rename(columns={'DISTRICT_NAME': 'District_Name'})
        if 'REPRESENTATIVE' in all_metrics.columns:
            all_metrics = all_metrics.rename(columns={'REPRESENTATIVE': 'Representative_Name'})
        if 'District_Name' not in all_metrics.columns:
            all_metrics['District_Name'] = 'District ' + all_metrics['District_ID'].astype(str)
        if 'Representative_Name' not in all_metrics.columns:
            all_metrics['Representative_Name'] = ''
        
        # Merge metrics (avoid duplicate columns by specifying suffixes or dropping duplicates)
        all_metrics = all_metrics.merge(density_metrics.drop(columns=['area_sq_mi'], errors='ignore'), on='District_ID', how='left')
        all_metrics = all_metrics.merge(clustering_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(concentration_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(variability_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(spatial_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(distribution_metrics, on='District_ID', how='left')
        
        # Remove any duplicate columns created from merge
        all_metrics = all_metrics.loc[:, ~all_metrics.columns.duplicated()]
        
        # Create combined pattern type
        all_metrics = self.createCombinedPatternType(all_metrics)
        
        # Rename to camelCase
        all_metrics = all_metrics.rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName',
            'area_sq_mi': 'districtAreaSquareMiles',
            'centroid_x': 'districtCentroidX',
            'centroid_y': 'districtCentroidY',
            'totalRequests': 'totalRequests',
            'requestsPerSquareMile': 'requestsPerSquareMile',
            'cityAverageRequestsPerSquareMile': 'cityAverageRequestsPerSquareMile',
            'densityRatioToCityAverage': 'densityRatioToCityAverage',
            'densityRank': 'densityRank',
            'densityPercentile': 'densityPercentile',
            'densityQuartile': 'densityQuartile',
            'clusterType': 'clusterType',
            'clusterZScore': 'clusterZScore',
            'clusterPValue': 'clusterPValue',
            'neighborAverageDensity': 'neighborAverageDensity',
            'neighborDensityRatio': 'neighborDensityRatio',
            'requestConcentrationIndex': 'requestConcentrationIndex',
            'standardDistance': 'standardDistance',
            'concentrationPercentile': 'concentrationPercentile',
            'concentrationQuartile': 'concentrationQuartile',
            'concentrationCategory': 'concentrationCategory',
            'spatialVariability': 'spatialVariability',
            'requestPatternConsistency': 'requestPatternConsistency',
            'variabilityPercentile': 'variabilityPercentile',
            'variabilityQuartile': 'variabilityQuartile',
            'variabilityCategory': 'variabilityCategory',
            'adjacentDistrictCount': 'adjacentDistrictCount',
            'adjacentDistrictsHighDensity': 'adjacentDistrictsHighDensity',
            'adjacentDistrictsLowDensity': 'adjacentDistrictsLowDensity',
            'spatialIsolationIndex': 'spatialIsolationIndex',
            'requestCentroidX': 'requestCentroidX',
            'requestCentroidY': 'requestCentroidY',
            'distanceFromDistrictCentroid': 'distanceFromDistrictCentroid',
            'requestSpread': 'requestSpread',
            'geographicPatternType': 'geographicPatternType'
        })
        
        # Create request points layer
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, all_metrics)
        
        # Create GeoPackage
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            outputPath = os.path.join(tempDir, self.outputName)
            
            all_metrics.to_file(outputPath, driver='GPKG', layer='District Geographic Patterns')
            points_gdf.to_file(outputPath, driver='GPKG', layer='Service Requests', mode='a')
            
            print("Uploading to S3...")
            s3Urls = self.uploadGeoPackageToS3(outputPath)
        
        print(f"\n✅ Geographic Patterns GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Urls['mainUrl']}")
        print(f"\nSummary:")
        print(f"  Districts: {len(all_metrics)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        
        return {
            's3Urls': s3Urls,
            'mainUrl': s3Urls.get('mainUrl', ''),
            'summary': {
                'districts': len(all_metrics),
                'request_points': len(points_gdf),
                'total_requests': len(df)
            }
        }


def main():
    """Main function to generate geographic patterns GeoPackage."""
    try:
        generator = GeographicPatternsGenerator()
        result = generator.createShapefile()
        
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {result['mainUrl']}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Geographic Patterns' (polygons)")
        print(f"     2. 'Service Requests' (points)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

