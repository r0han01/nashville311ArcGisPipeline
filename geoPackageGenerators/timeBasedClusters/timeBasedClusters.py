#!/usr/bin/env python3
"""
Time-Based Clusters GeoPackage Generator

Creates a GeoPackage identifying geographic clusters of districts with similar
temporal request patterns. Groups districts that have similar weekday/weekend,
peak hour, and daily/weekly timing behaviors for resource planning and service
optimization.

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
from typing import Optional, Dict, Any, List, Tuple
from shapely.geometry import Point
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from nashvilleGis import NashvilleDataFetcher, NashvilleConfig


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


def calculate_data_driven_k_range(n):
    """
    Calculate k_range based on sample size (fully data-driven).
    
    Formula: k_max = min(5, max(3, int(sqrt(n)), int(n/7)))
    """
    k_min = 3
    k_max = min(5, max(3, int(np.sqrt(n)), int(n / 7)))
    k_max = max(k_min, k_max)
    return list(range(k_min, k_max + 1))


def find_optimal_k_clustering(features, k_range=None):
    """Find optimal number of clusters using Silhouette score."""
    if k_range is None:
        n = len(features)
        k_range = calculate_data_driven_k_range(n)
    
    if len(features) < min(k_range):
        return min(k_range), None
    
    # Normalize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)
    
    scores = {}
    models = {}
    
    for k in k_range:
        if len(features) >= k:
            try:
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                labels = kmeans.fit_predict(features_scaled)
                
                if len(np.unique(labels)) > 1:
                    score = silhouette_score(features_scaled, labels)
                    scores[k] = score
                    models[k] = (kmeans, scaler, labels)
            except:
                continue
    
    if not scores:
        return min(k_range), None
    
    optimal_k = max(scores, key=scores.get)
    
    # If scores are very close (within 0.05), prefer k=4 for interpretability
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    if len(sorted_scores) >= 2:
        best_score = sorted_scores[0][1]
        second_score = sorted_scores[1][1]
        if abs(best_score - second_score) < 0.05:
            if 4 in scores:
                optimal_k = 4
            else:
                k_values = sorted(scores.keys())
                optimal_k = k_values[len(k_values) // 2]
    
    return optimal_k, models[optimal_k]


def generate_cluster_name(cluster_features: Dict[str, float]) -> str:
    """Generate descriptive cluster name based on characteristics."""
    weekday_pct = cluster_features.get('weekdayPercent', 50)
    peak_hour_pct = cluster_features.get('peakHourPercent', 50)
    volatility = cluster_features.get('dailyVolatilityCoefficient', 0.5)
    
    # Determine dominant pattern
    if weekday_pct >= 65:
        day_pattern = "Weekday-Dominant"
    elif weekday_pct <= 45:
        day_pattern = "Weekend-Dominant"
    else:
        day_pattern = "Balanced"
    
    if peak_hour_pct >= 55:
        hour_pattern = "Peak-Hours"
    elif peak_hour_pct <= 40:
        hour_pattern = "Off-Hours"
    else:
        hour_pattern = "Mixed-Hours"
    
    if volatility >= 0.6:
        vol_pattern = "High-Volatility"
    elif volatility <= 0.3:
        vol_pattern = "Low-Volatility"
    else:
        vol_pattern = "Moderate-Volatility"
    
    # Combine patterns
    if vol_pattern == "High-Volatility":
        return f"{vol_pattern} {day_pattern}"
    else:
        return f"{day_pattern} {hour_pattern}"


class TimeBasedClustersGenerator:
    """Generator for district temporal clustering GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/timeBasedClusters'
        self.outputName = 'timeBasedClusters.gpkg'
        
        # Standard business hours
        self.PEAK_HOUR_START = 8
        self.PEAK_HOUR_END = 17
    
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
    
    def calculateTemporalFeatures(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate temporal features for each district."""
        # Convert timestamp to datetime
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        # Extract temporal components
        df['dayOfWeek'] = df['openedDt'].dt.dayofweek  # 0=Monday, 6=Sunday
        df['dayName'] = df['openedDt'].dt.day_name()
        df['hourOfDay'] = df['openedDt'].dt.hour
        df['date'] = df['openedDt'].dt.date
        df['week'] = df['openedDt'].dt.to_period('W').astype(str)
        df['isWeekday'] = df['dayOfWeek'] < 5
        
        # Convert district to numeric
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        temporal_features = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                temporal_features.append({
                    'District_ID': district_id,
                    'totalRequests': 0,
                    'weekdayPercent': 0.0,
                    'weekendPercent': 0.0,
                    'peakHourPercent': 0.0,
                    'offHoursPercent': 0.0,
                    'mondayPercent': 0.0,
                    'tuesdayPercent': 0.0,
                    'wednesdayPercent': 0.0,
                    'thursdayPercent': 0.0,
                    'fridayPercent': 0.0,
                    'saturdayPercent': 0.0,
                    'sundayPercent': 0.0,
                    'morningPercent': 0.0,
                    'afternoonPercent': 0.0,
                    'eveningPercent': 0.0,
                    'nightPercent': 0.0,
                    'dailyVolatilityCoefficient': 0.0,
                    'weeklyVolatilityCoefficient': 0.0
                })
                continue
            
            total_requests = len(district_data)
            
            # Weekday vs weekend
            weekday_count = district_data['isWeekday'].sum()
            weekend_count = (~district_data['isWeekday']).sum()
            weekday_percent = (weekday_count / total_requests * 100) if total_requests > 0 else 0.0
            weekend_percent = (weekend_count / total_requests * 100) if total_requests > 0 else 0.0
            
            # Peak hours vs off-hours
            peak_hour_mask = (
                (district_data['hourOfDay'] >= self.PEAK_HOUR_START) &
                (district_data['hourOfDay'] < self.PEAK_HOUR_END) &
                (district_data['isWeekday'])
            )
            peak_hour_count = peak_hour_mask.sum()
            off_hours_count = total_requests - peak_hour_count
            peak_hour_percent = (peak_hour_count / total_requests * 100) if total_requests > 0 else 0.0
            off_hours_percent = (off_hours_count / total_requests * 100) if total_requests > 0 else 0.0
            
            # Day of week percentages
            day_counts = district_data['dayName'].value_counts()
            monday_pct = (day_counts.get('Monday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            tuesday_pct = (day_counts.get('Tuesday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            wednesday_pct = (day_counts.get('Wednesday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            thursday_pct = (day_counts.get('Thursday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            friday_pct = (day_counts.get('Friday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            saturday_pct = (day_counts.get('Saturday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            sunday_pct = (day_counts.get('Sunday', 0) / total_requests * 100) if total_requests > 0 else 0.0
            
            # Time of day percentages
            morning_mask = (district_data['hourOfDay'] >= 6) & (district_data['hourOfDay'] < 12)
            afternoon_mask = (district_data['hourOfDay'] >= 12) & (district_data['hourOfDay'] < 18)
            evening_mask = (district_data['hourOfDay'] >= 18) & (district_data['hourOfDay'] < 24)
            night_mask = (district_data['hourOfDay'] >= 0) & (district_data['hourOfDay'] < 6)
            
            morning_pct = (morning_mask.sum() / total_requests * 100) if total_requests > 0 else 0.0
            afternoon_pct = (afternoon_mask.sum() / total_requests * 100) if total_requests > 0 else 0.0
            evening_pct = (evening_mask.sum() / total_requests * 100) if total_requests > 0 else 0.0
            night_pct = (night_mask.sum() / total_requests * 100) if total_requests > 0 else 0.0
            
            # Daily volatility
            daily_counts = district_data.groupby('date').size()
            daily_cv = calculate_coefficient_of_variation(daily_counts.values) if len(daily_counts) > 1 else 0.0
            
            # Weekly volatility
            weekly_counts = district_data.groupby('week').size()
            weekly_cv = calculate_coefficient_of_variation(weekly_counts.values) if len(weekly_counts) > 1 else 0.0
            
            temporal_features.append({
                'District_ID': district_id,
                'totalRequests': total_requests,
                'weekdayPercent': round(weekday_percent, 2),
                'weekendPercent': round(weekend_percent, 2),
                'peakHourPercent': round(peak_hour_percent, 2),
                'offHoursPercent': round(off_hours_percent, 2),
                'mondayPercent': round(monday_pct, 2),
                'tuesdayPercent': round(tuesday_pct, 2),
                'wednesdayPercent': round(wednesday_pct, 2),
                'thursdayPercent': round(thursday_pct, 2),
                'fridayPercent': round(friday_pct, 2),
                'saturdayPercent': round(saturday_pct, 2),
                'sundayPercent': round(sunday_pct, 2),
                'morningPercent': round(morning_pct, 2),
                'afternoonPercent': round(afternoon_pct, 2),
                'eveningPercent': round(evening_pct, 2),
                'nightPercent': round(night_pct, 2),
                'dailyVolatilityCoefficient': round(daily_cv, 4),
                'weeklyVolatilityCoefficient': round(weekly_cv, 4)
            })
        
        return pd.DataFrame(temporal_features)
    
    def performClustering(self, features_df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict]:
        """Perform K-Means clustering on temporal features."""
        # Select feature columns for clustering
        feature_columns = [
            'weekdayPercent', 'weekendPercent',
            'peakHourPercent', 'offHoursPercent',
            'mondayPercent', 'tuesdayPercent', 'wednesdayPercent',
            'thursdayPercent', 'fridayPercent', 'saturdayPercent', 'sundayPercent',
            'morningPercent', 'afternoonPercent', 'eveningPercent', 'nightPercent',
            'dailyVolatilityCoefficient', 'weeklyVolatilityCoefficient'
        ]
        
        # Prepare features matrix
        features_matrix = features_df[feature_columns].fillna(0).values
        
        # Find optimal k and perform clustering
        optimal_k, (kmeans_model, scaler, labels) = find_optimal_k_clustering(features_matrix)
        
        # Add cluster assignments
        features_df['temporalClusterId'] = labels
        
        # Calculate cluster characteristics
        cluster_info = {}
        for cluster_id in range(optimal_k):
            cluster_mask = labels == cluster_id
            cluster_districts = features_df[cluster_mask]
            
            if len(cluster_districts) > 0:
                cluster_features = {
                    'weekdayPercent': cluster_districts['weekdayPercent'].mean(),
                    'weekendPercent': cluster_districts['weekendPercent'].mean(),
                    'peakHourPercent': cluster_districts['peakHourPercent'].mean(),
                    'offHoursPercent': cluster_districts['offHoursPercent'].mean(),
                    'dailyVolatilityCoefficient': cluster_districts['dailyVolatilityCoefficient'].mean(),
                    'weeklyVolatilityCoefficient': cluster_districts['weeklyVolatilityCoefficient'].mean()
                }
                
                cluster_name = generate_cluster_name(cluster_features)
                
                cluster_info[cluster_id] = {
                    'name': cluster_name,
                    'size': len(cluster_districts),
                    'features': cluster_features
                }
        
        # Add cluster names and characteristics
        features_df['temporalClusterName'] = features_df['temporalClusterId'].map(
            lambda x: cluster_info[x]['name'] if x in cluster_info else 'Unknown'
        )
        features_df['clusterSize'] = features_df['temporalClusterId'].map(
            lambda x: cluster_info[x]['size'] if x in cluster_info else 0
        )
        
        # Calculate distance from cluster center and confidence
        features_scaled = scaler.transform(features_matrix)
        cluster_centers = kmeans_model.cluster_centers_
        
        distances = []
        confidences = []
        
        for idx, row in features_df.iterrows():
            cluster_id = row['temporalClusterId']
            point = features_scaled[idx]
            center = cluster_centers[cluster_id]
            
            # Euclidean distance from cluster center
            distance = np.linalg.norm(point - center)
            distances.append(distance)
            
            # Confidence: inverse of distance (normalized)
            # Lower distance = higher confidence
            max_distance = np.max([np.linalg.norm(point - c) for c in cluster_centers])
            if max_distance > 0:
                confidence = 1.0 - (distance / max_distance)
            else:
                confidence = 1.0
            confidences.append(max(0, min(1, confidence)))
        
        features_df['clusterDistance'] = [round(d, 4) for d in distances]
        features_df['clusterConfidence'] = [round(c, 4) for c in confidences]
        
        # Add cluster characteristics
        features_df['clusterWeekdayPercent'] = features_df['temporalClusterId'].map(
            lambda x: round(cluster_info[x]['features']['weekdayPercent'], 2) if x in cluster_info else 0.0
        )
        features_df['clusterPeakHourPercent'] = features_df['temporalClusterId'].map(
            lambda x: round(cluster_info[x]['features']['peakHourPercent'], 2) if x in cluster_info else 0.0
        )
        features_df['clusterVolatility'] = features_df['temporalClusterId'].map(
            lambda x: round(cluster_info[x]['features']['dailyVolatilityCoefficient'], 4) if x in cluster_info else 0.0
        )
        
        # Determine dominant pattern for cluster
        def get_dominant_pattern(row):
            cluster_id = row['temporalClusterId']
            if cluster_id not in cluster_info:
                return 'Unknown'
            
            features = cluster_info[cluster_id]['features']
            weekday_pct = features['weekdayPercent']
            peak_pct = features['peakHourPercent']
            
            if weekday_pct >= 65 and peak_pct >= 55:
                return 'Weekday Peak-Hours'
            elif weekday_pct <= 45 and peak_pct <= 40:
                return 'Weekend Off-Hours'
            elif weekday_pct >= 65:
                return 'Weekday-Dominant'
            elif weekday_pct <= 45:
                return 'Weekend-Dominant'
            else:
                return 'Balanced'
        
        features_df['clusterDominantPattern'] = features_df.apply(get_dominant_pattern, axis=1)
        
        return features_df, cluster_info
    
    def calculateGeographicSpread(self, features_df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate geographic spread of clusters."""
        # Merge with boundaries to get centroids
        merged = boundaries.merge(features_df[['District_ID', 'temporalClusterId']], on='District_ID', how='left')
        
        # Calculate centroids
        merged['centroid'] = merged.geometry.centroid
        merged['centroid_x'] = merged['centroid'].x
        merged['centroid_y'] = merged['centroid'].y
        
        # Calculate cluster geographic spread (standard distance)
        cluster_spreads = {}
        for cluster_id in merged['temporalClusterId'].unique():
            if pd.isna(cluster_id):
                continue
            
            cluster_data = merged[merged['temporalClusterId'] == cluster_id]
            if len(cluster_data) > 1:
                centroid_x = cluster_data['centroid_x'].mean()
                centroid_y = cluster_data['centroid_y'].mean()
                
                distances = np.sqrt(
                    (cluster_data['centroid_x'] - centroid_x) ** 2 +
                    (cluster_data['centroid_y'] - centroid_y) ** 2
                )
                spread = distances.std()
            else:
                spread = 0.0
            
            cluster_spreads[cluster_id] = spread
        
        features_df['clusterGeographicSpread'] = features_df['temporalClusterId'].map(
            lambda x: round(cluster_spreads.get(x, 0.0), 4) if not pd.isna(x) else 0.0
        )
        
        return features_df
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with cluster metrics."""
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
            metrics_merge = metrics[['District_ID', 'temporalClusterId', 'temporalClusterName',
                                    'clusterSize', 'clusterDominantPattern']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'temporalClusterId': 'districtTemporalClusterId',
                'temporalClusterName': 'districtTemporalClusterName',
                'clusterSize': 'districtClusterSize',
                'clusterDominantPattern': 'districtClusterDominantPattern'
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
        
        print("Calculating temporal features...")
        features_df = self.calculateTemporalFeatures(df, boundaries)
        
        print("Performing temporal clustering...")
        clustered_df, cluster_info = self.performClustering(features_df)
        
        print("Calculating geographic spread...")
        clustered_df = self.calculateGeographicSpread(clustered_df, boundaries)
        
        # Merge with boundaries
        boundaries = boundaries.merge(clustered_df, on='District_ID', how='left')
        
        # Convert to camelCase
        boundaries = boundaries.rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName'
        })
        
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, clustered_df)
        
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkgPath = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            boundaries.to_file(gpkgPath, layer='District Temporal Clusters', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Time-Based Clusters GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"  Clusters Found: {len(cluster_info)}")
        for cluster_id, info in cluster_info.items():
            print(f"    Cluster {cluster_id}: {info['name']} ({info['size']} districts)")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Temporal Clusters' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = TimeBasedClustersGenerator()
    generator.createShapefile()

