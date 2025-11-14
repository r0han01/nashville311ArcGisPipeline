# Council District Boundaries

## Why We Need This Dataset

The Nashville 311 API provides service request data with coordinates, but it doesn't include the polygon boundaries for council districts. To create meaningful GIS visualizations (like district performance maps), we need the actual district boundaries.

## Data Source

We're using the **Council Districts** dataset from the [Nashville Open Data Portal](https://data.nashville.gov/datasets/council-districts-current-1/explore).

![Dataset Overview](<img width="1920" height="961" alt="dataset-overview.png" src="https://github.com/user-attachments/assets/7b2ff11e-11b4-4ce8-8148-88c1ad20df09" />)

This dataset contains the polygon boundaries for all 35 Nashville council districts, which we use to:
- Create district-level performance analysis
- Generate polygon-based shapefiles for ArcGIS Pro
- Merge service request data with geographic boundaries

## Storage

All boundary files are automatically uploaded to AWS S3 at:
```
s3://nashville311-gis-analysis-data/boundaries/nashvilleCouncilDistricts/
```

Our Python scripts download these files from S3 when needed, so you don't need to keep local copies.

## Using the Original Dataset

If you need the most up-to-date boundaries or want to explore the dataset yourself, you can access it directly from the source:

**Nashville Open Data Portal:** https://data.nashville.gov/datasets/council-districts-current-1/explore

The dataset is updated by the city, so this is the authoritative source for council district boundaries.

## Files in This Directory

This directory is kept for reference, but the actual files used in processing are stored in S3. The Python code automatically downloads from S3 when generating shapefiles.

