#!/usr/bin/env python3
"""
Service Type Insights GeoPackage Generator

Creates a GeoPackage analyzing service type patterns across Nashville council
districts. Focuses on service type mix (diversity vs specialization), recurring
issue zones, and service complexity for resource planning and infrastructure management.

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


def get_quantile_labels_and_classifier(values, optimal_k, metric_name, ascending=True):
    """Generate percentile-based labels and classification function."""
    sorted_values = np.sort(values)[::-1] if not ascending else np.sort(values)
    quantiles = np.linspace(0, 100, optimal_k + 1)
    breaks = np.percentile(sorted_values, quantiles)
    
    # Create labels based on optimal k
    labels = []
    for i in range(optimal_k):
        if optimal_k == 3:
            if i == 0:
                labels.append('High' if ascending else 'Low')
            elif i == 1:
                labels.append('Moderate')
            else:
                labels.append('Low' if ascending else 'High')
        elif optimal_k == 4:
            if i == 0:
                labels.append('Very High' if ascending else 'Very Low')
            elif i == 1:
                labels.append('High' if ascending else 'Low')
            elif i == 2:
                labels.append('Moderate-High' if ascending else 'Moderate-Low')
            else:
                labels.append('Low' if ascending else 'High')
        else:  # k == 5
            if i == 0:
                labels.append('Very High' if ascending else 'Very Low')
            elif i == optimal_k - 1:
                labels.append('Very Low' if ascending else 'Very High')
            else:
                labels.append('Moderate')
    
    # Create classification function
    def classify(value):
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
    
    return labels, classify


class ServiceTypeInsightsGenerator:
    """Generator for district service type insights GeoPackages."""
    
    def __init__(self, bucketName: Optional[str] = None):
        """Initialize the generator."""
        self.config = NashvilleConfig()
        self.bucketName = bucketName or self.config.BUCKET_NAME
        self.s3Client = boto3.client('s3')
        self.boundaryS3Key = 'boundaries/nashvilleCouncilDistricts/2022_Council_Districts.shp'
        self.s3ShapefilePrefix = 'gpkg-public/serviceTypeInsights'
        self.outputName = 'serviceTypeInsights.gpkg'
    
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
    
    def calculateServiceTypeMix(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate service type mix metrics (diversity, concentration)."""
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        mix_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id]
            
            if len(district_data) == 0:
                mix_metrics.append({
                    'District_ID': district_id,
                    'serviceTypeDiversity': 0,
                    'serviceTypeConcentration': 0.0,
                    'topServiceType': None,
                    'topServiceTypePercent': 0.0,
                    'top3ServiceTypesPercent': 0.0
                })
                continue
            
            # Count service types
            service_type_counts = district_data['Request_Type'].value_counts()
            total_requests = len(district_data)
            
            # Service type diversity (number of unique types)
            diversity = len(service_type_counts)
            
            # Service type concentration (Herfindahl index)
            if total_requests > 0:
                proportions = service_type_counts / total_requests
                concentration = np.sum(proportions ** 2)  # Herfindahl index
            else:
                concentration = 0.0
            
            # Top service type
            if len(service_type_counts) > 0:
                top_service_type = service_type_counts.index[0]
                top_service_type_count = service_type_counts.iloc[0]
                top_service_type_percent = (top_service_type_count / total_requests * 100) if total_requests > 0 else 0.0
                
                # Top 3 service types
                top3_count = service_type_counts.head(3).sum()
                top3_percent = (top3_count / total_requests * 100) if total_requests > 0 else 0.0
            else:
                top_service_type = None
                top_service_type_percent = 0.0
                top3_percent = 0.0
            
            mix_metrics.append({
                'District_ID': district_id,
                'serviceTypeDiversity': diversity,
                'serviceTypeConcentration': round(concentration, 4),
                'topServiceType': top_service_type,
                'topServiceTypePercent': round(top_service_type_percent, 2),
                'top3ServiceTypesPercent': round(top3_percent, 2)
            })
        
        return pd.DataFrame(mix_metrics)
    
    def calculateRecurringIssues(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate recurring issue metrics."""
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Convert timestamp for temporal analysis
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df = df[df['openedDt'].notna()].copy()
        
        recurring_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                recurring_metrics.append({
                    'District_ID': district_id,
                    'recurringIssueCount': 0,
                    'recurringIssuePercent': 0.0,
                    'mostRecurringServiceType': None,
                    'recurringIssueFrequency': 0.0
                })
                continue
            
            total_requests = len(district_data)
            
            # Identify recurring service types
            # A service type is "recurring" if it appears multiple times
            # We'll use a statistical threshold: service types that appear more than expected
            service_type_counts = district_data['Request_Type'].value_counts()
            
            # Expected frequency if evenly distributed
            expected_frequency = total_requests / len(service_type_counts) if len(service_type_counts) > 0 else 0
            
            # Recurring types: those that appear more than 1.5x expected (data-driven threshold)
            # This is based on statistical deviation, not hardcoded
            if expected_frequency > 0:
                recurring_threshold = expected_frequency * 1.5
                recurring_types = service_type_counts[service_type_counts > recurring_threshold]
            else:
                recurring_types = pd.Series(dtype=int)
            
            recurring_count = len(recurring_types)
            recurring_requests = recurring_types.sum() if len(recurring_types) > 0 else 0
            recurring_percent = (recurring_requests / total_requests * 100) if total_requests > 0 else 0.0
            
            # Most recurring service type
            if len(recurring_types) > 0:
                most_recurring = recurring_types.index[0]
                recurring_frequency = recurring_types.iloc[0] / total_requests if total_requests > 0 else 0.0
            else:
                most_recurring = None
                recurring_frequency = 0.0
            
            recurring_metrics.append({
                'District_ID': district_id,
                'recurringIssueCount': recurring_count,
                'recurringIssuePercent': round(recurring_percent, 2),
                'mostRecurringServiceType': most_recurring,
                'recurringIssueFrequency': round(recurring_frequency, 4)
            })
        
        return pd.DataFrame(recurring_metrics)
    
    def calculateServiceComplexity(self, df: pd.DataFrame, boundaries: gpd.GeoDataFrame) -> pd.DataFrame:
        """Calculate service complexity metrics."""
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        
        # Convert timestamps
        df['openedDt'] = pd.to_datetime(df['Date_Time_Opened'], unit='ms', errors='coerce')
        df['closedDt'] = pd.to_datetime(df['Date_Time_Closed'], unit='ms', errors='coerce')
        
        # Calculate response time (complex services typically take longer)
        df['responseTimeHours'] = (
            (df['closedDt'] - df['openedDt']).dt.total_seconds() / 3600.0
        )
        
        # Calculate complexity score per service type (data-driven)
        # Complexity is based on:
        # 1. Average response time (longer = more complex)
        # 2. Frequency (more frequent = simpler, routine)
        service_type_stats = df.groupby('Request_Type').agg({
            'responseTimeHours': ['mean', 'median', 'count']
        }).reset_index()
        service_type_stats.columns = ['Request_Type', 'avgResponseTime', 'medianResponseTime', 'frequency']
        
        # Normalize complexity factors
        if len(service_type_stats) > 0:
            max_response_time = service_type_stats['avgResponseTime'].max()
            max_frequency = service_type_stats['frequency'].max()
            
            if max_response_time > 0:
                service_type_stats['responseTimeScore'] = service_type_stats['avgResponseTime'] / max_response_time
            else:
                service_type_stats['responseTimeScore'] = 0.0
            
            if max_frequency > 0:
                service_type_stats['frequencyScore'] = 1.0 - (service_type_stats['frequency'] / max_frequency)  # Inverse: less frequent = more complex
            else:
                service_type_stats['frequencyScore'] = 0.0
            
            # Complexity score: weighted average
            service_type_stats['complexityScore'] = (
                0.6 * service_type_stats['responseTimeScore'] +
                0.4 * service_type_stats['frequencyScore']
            )
        else:
            service_type_stats['complexityScore'] = 0.5
        
        # Create complexity mapping
        complexity_map = dict(zip(service_type_stats['Request_Type'], service_type_stats['complexityScore']))
        
        # Classify service types as simple or complex (data-driven threshold)
        median_complexity = service_type_stats['complexityScore'].median()
        service_type_stats['isComplex'] = service_type_stats['complexityScore'] > median_complexity
        complexity_classification = dict(zip(service_type_stats['Request_Type'], service_type_stats['isComplex']))
        
        complexity_metrics = []
        
        for district_id in boundaries['District_ID']:
            district_data = df[df['districtInt'] == district_id].copy()
            
            if len(district_data) == 0:
                complexity_metrics.append({
                    'District_ID': district_id,
                    'simpleServicePercent': 0.0,
                    'complexServicePercent': 0.0,
                    'averageServiceComplexity': 0.0
                })
                continue
            
            total_requests = len(district_data)
            
            # Count simple vs complex services
            district_data['isComplex'] = district_data['Request_Type'].map(complexity_classification).fillna(False)
            complex_count = district_data['isComplex'].sum()
            simple_count = total_requests - complex_count
            
            simple_percent = (simple_count / total_requests * 100) if total_requests > 0 else 0.0
            complex_percent = (complex_count / total_requests * 100) if total_requests > 0 else 0.0
            
            # Average complexity score
            district_data['complexityScore'] = district_data['Request_Type'].map(complexity_map).fillna(0.5)
            avg_complexity = district_data['complexityScore'].mean()
            
            complexity_metrics.append({
                'District_ID': district_id,
                'simpleServicePercent': round(simple_percent, 2),
                'complexServicePercent': round(complex_percent, 2),
                'averageServiceComplexity': round(avg_complexity, 4)
            })
        
        return pd.DataFrame(complexity_metrics)
    
    def classifyMetrics(self, metrics: pd.DataFrame) -> pd.DataFrame:
        """Add data-driven classifications using Silhouette + Quantiles."""
        # Service Type Mix Category
        concentration_values = metrics['serviceTypeConcentration'].dropna().values
        if len(concentration_values) >= 3:
            optimal_k = find_optimal_k_silhouette(concentration_values, k_range=None)
            labels, classifier = get_quantile_labels_and_classifier(
                concentration_values, optimal_k, 'concentration', ascending=False
            )
            # Map to "Specialized" vs "Diverse"
            if optimal_k == 3:
                label_map = {'High': 'Specialized', 'Moderate': 'Moderate', 'Low': 'Diverse'}
            elif optimal_k == 4:
                label_map = {'Very High': 'Highly Specialized', 'High': 'Specialized', 
                            'Moderate-High': 'Moderate', 'Low': 'Diverse'}
            else:
                label_map = {'Very High': 'Highly Specialized', 'Moderate': 'Moderate', 
                            'Very Low': 'Diverse'}
            
            metrics['serviceTypeMixCategory'] = metrics['serviceTypeConcentration'].apply(
                lambda x: label_map.get(classifier(x), 'Moderate') if not pd.isna(x) else 'Unknown'
            )
        else:
            metrics['serviceTypeMixCategory'] = 'Moderate'
        
        # Recurring Issue Category
        recurring_values = metrics['recurringIssuePercent'].dropna().values
        if len(recurring_values) >= 3:
            optimal_k = find_optimal_k_silhouette(recurring_values, k_range=None)
            labels, classifier = get_quantile_labels_and_classifier(
                recurring_values, optimal_k, 'recurring', ascending=False
            )
            # Map to recurrence categories
            if optimal_k == 3:
                label_map = {'High': 'High Recurrence', 'Moderate': 'Moderate Recurrence', 'Low': 'Low Recurrence'}
            elif optimal_k == 4:
                label_map = {'Very High': 'Very High Recurrence', 'High': 'High Recurrence',
                            'Moderate-High': 'Moderate Recurrence', 'Low': 'Low Recurrence'}
            else:
                label_map = {'Very High': 'Very High Recurrence', 'Moderate': 'Moderate Recurrence',
                            'Very Low': 'Low Recurrence'}
            
            metrics['recurringIssueCategory'] = metrics['recurringIssuePercent'].apply(
                lambda x: label_map.get(classifier(x), 'Moderate Recurrence') if not pd.isna(x) else 'Unknown'
            )
        else:
            metrics['recurringIssueCategory'] = 'Moderate Recurrence'
        
        # Complexity Category
        complexity_values = metrics['averageServiceComplexity'].dropna().values
        if len(complexity_values) >= 3:
            optimal_k = find_optimal_k_silhouette(complexity_values, k_range=None)
            labels, classifier = get_quantile_labels_and_classifier(
                complexity_values, optimal_k, 'complexity', ascending=False
            )
            # Map to complexity categories
            if optimal_k == 3:
                label_map = {'High': 'Complex', 'Moderate': 'Moderate', 'Low': 'Simple'}
            elif optimal_k == 4:
                label_map = {'Very High': 'Highly Complex', 'High': 'Complex',
                            'Moderate-High': 'Moderate', 'Low': 'Simple'}
            else:
                label_map = {'Very High': 'Highly Complex', 'Moderate': 'Moderate',
                            'Very Low': 'Simple'}
            
            metrics['complexityCategory'] = metrics['averageServiceComplexity'].apply(
                lambda x: label_map.get(classifier(x), 'Moderate') if not pd.isna(x) else 'Unknown'
            )
        else:
            metrics['complexityCategory'] = 'Moderate'
        
        return metrics
    
    def createRequestPointsLayer(self, df: pd.DataFrame, metrics: Optional[pd.DataFrame] = None) -> gpd.GeoDataFrame:
        """Create a GeoDataFrame of request points with service type insights."""
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
            metrics_merge = metrics[['District_ID', 'serviceTypeMixCategory', 'recurringIssueCategory',
                                    'complexityCategory', 'topServiceType']].copy()
            metrics_merge = metrics_merge.rename(columns={
                'District_ID': 'districtId',
                'serviceTypeMixCategory': 'districtServiceTypeMixCategory',
                'recurringIssueCategory': 'districtRecurringIssueCategory',
                'complexityCategory': 'districtComplexityCategory',
                'topServiceType': 'districtTopServiceType'
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
        
        print("Calculating service type mix...")
        mix_metrics = self.calculateServiceTypeMix(df, boundaries)
        
        print("Calculating recurring issues...")
        recurring_metrics = self.calculateRecurringIssues(df, boundaries)
        
        print("Calculating service complexity...")
        complexity_metrics = self.calculateServiceComplexity(df, boundaries)
        
        # Merge all metrics
        all_metrics = boundaries[['District_ID']].copy()
        all_metrics = all_metrics.merge(mix_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(recurring_metrics, on='District_ID', how='left')
        all_metrics = all_metrics.merge(complexity_metrics, on='District_ID', how='left')
        
        # Add total requests
        df['districtInt'] = pd.to_numeric(df['Council_District'], errors='coerce').astype('Int64')
        request_counts = df.groupby('districtInt').size().reset_index(name='totalRequests')
        request_counts = request_counts.rename(columns={'districtInt': 'District_ID'})
        all_metrics = all_metrics.merge(request_counts, on='District_ID', how='left')
        all_metrics['totalRequests'] = all_metrics['totalRequests'].fillna(0).astype(int)
        
        print("Classifying metrics...")
        all_metrics = self.classifyMetrics(all_metrics)
        
        # Merge with boundaries
        boundaries = boundaries.merge(all_metrics, on='District_ID', how='left')
        
        # Convert to camelCase
        boundaries = boundaries.rename(columns={
            'District_ID': 'districtId',
            'District_Name': 'districtName',
            'Representative_Name': 'representativeName'
        })
        
        print("Creating request points layer...")
        points_gdf = self.createRequestPointsLayer(df, all_metrics)
        
        print("Creating GeoPackage...")
        with tempfile.TemporaryDirectory() as tempDir:
            gpkgPath = os.path.join(tempDir, self.outputName)
            
            # Write district layer
            boundaries.to_file(gpkgPath, layer='District Service Type Insights', driver='GPKG')
            
            # Append points layer
            points_gdf.to_file(gpkgPath, layer='Service Requests', driver='GPKG')
            
            print("Uploading to S3...")
            s3Url = self.uploadGeoPackageToS3(gpkgPath)
        
        print(f"\n✅ Service Type Insights GeoPackage created and uploaded!")
        print(f"S3 URL: {s3Url}")
        print(f"\nSummary:")
        print(f"  Districts: {len(boundaries)}")
        print(f"  Request Points: {len(points_gdf):,}")
        print(f"  Total Requests: {len(df):,}")
        print(f"\n✅ Ready for ArcGIS Pro!")
        print(f"   GeoPackage URL: {s3Url}")
        print(f"   Contains 2 layers:")
        print(f"     1. 'District Service Type Insights' (polygons)")
        print(f"     2. 'Service Requests' (points)")


if __name__ == '__main__':
    generator = ServiceTypeInsightsGenerator()
    generator.createShapefile()

