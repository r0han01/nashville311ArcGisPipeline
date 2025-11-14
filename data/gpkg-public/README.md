# Public GeoPackage Files for ArcGIS Pro

This folder contains all public GeoPackage (`.gpkg`) files that you can download and load directly into ArcGIS Pro for visualization and analysis.

## What's Here

These GeoPackage files are generated from Nashville 311 service request data and are stored in AWS S3 for easy access. Each file contains spatial data with calculated metrics and attributes ready for mapping.

## Download URLs

All files are publicly accessible via S3 URLs. Use this format to download:

```
https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/[folder-name]/[filename].gpkg
```

**Example:**
```
https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/districtPerformanceRanking/districtPerformanceRanking.gpkg
```

## Folder Structure

Files are organized by analysis type in S3:

```
gpkg-public/
├── districtPerformanceRanking/
│   └── districtPerformanceRanking.gpkg
├── performanceQuartiles/
│   └── performanceQuartiles.gpkg
├── requestDensityZones/
│   └── requestDensityZones.gpkg
└── ... (more folders as they're generated)
```

## Planned GeoPackage Files

### **📊 Relative Performance Analysis:**
1. `districtPerformanceRanking.gpkg` - Districts ranked by response time (best to worst)
2. `performanceQuartiles.gpkg` - Districts grouped into performance quartiles
3. `relativeWorkload.gpkg` - Districts ranked by requests per capita/area
4. `serviceEfficiency.gpkg` - Districts with fastest response relative to workload

### **🗺️ Geographic Pattern Analysis:**
5. `requestDensityZones.gpkg` - High/Medium/Low density areas (relative to city average)
6. `spatialClusters.gpkg` - Geographic clusters regardless of absolute numbers
7. `serviceConcentration.gpkg` - Areas with concentrated vs scattered requests
8. `geographicVariability.gpkg` - Areas with consistent vs variable request patterns

### **⏰ Temporal Pattern Analysis:**
9. `temporalPatterns.gpkg` - Areas with consistent weekday vs weekend patterns
10. `peakHourConsistency.gpkg` - Areas that consistently have off-hours requests
11. `requestVolatility.gpkg` - Areas with stable vs volatile request patterns
12. `timeBasedClusters.gpkg` - Geographic areas with similar temporal patterns

### **🔧 Service Type Insights:**
13. `serviceTypeMix.gpkg` - Areas with diverse vs specialized service needs
14. `recurringIssueZones.gpkg` - Areas with consistent recurring problem types
15. `serviceComplexity.gpkg` - Areas requiring simple vs complex service types
16. `infrastructureAge.gpkg` - Areas with frequent infrastructure-related requests

### **📈 Trend & Predictive Analysis:**
17. `trendDirection.gpkg` - Areas with increasing vs decreasing request trends
18. `predictiveHotspots.gpkg` - Areas likely to need attention based on patterns
19. `capacityPlanning.gpkg` - Areas approaching service capacity limits
20. `optimizationOpportunities.gpkg` - Areas with biggest improvement potential

*More files may be added in the future as the project evolves.*

## Using in ArcGIS Pro

1. **Download** the `.gpkg` file using the S3 URL above
2. **Open ArcGIS Pro** and create a new project
3. **Add Data** → Browse to the downloaded `.gpkg` file
4. **Visualize** using the built-in attributes and calculated metrics

Each GeoPackage contains:
- Spatial geometries (polygons or points)
- Calculated metrics and rankings
- Descriptive attributes ready for styling
- Proper coordinate reference system (EPSG:4326)

## Storage

All files are stored in AWS S3 at:
```
s3://nashville311-gis-analysis-data/gpkg-public/
```

Files are automatically generated and updated as new data is processed.

