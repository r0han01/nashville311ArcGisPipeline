# Processed Nashville 311 Data

## What's Here

This directory represents processed Nashville 311 service request data stored in AWS S3. The actual files are in the cloud, not locally.

## How We Process the Data

1. **Fetch from API**: We pull the last 3 months of service requests from the [Nashville 311 API](https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/hubNashville_311_Service_Requests_Current_Year_view/FeatureServer/0/query)

2. **Filter Dynamically**: The system automatically calculates the date range for the last 3 months, so it always stays current without manual updates

3. **Convert to Parquet**: We convert the JSON data to Parquet format for better performance:
   - **11.5x smaller** file size (compression)
   - **Faster** reading and processing
   - **Production-ready** format for analytics

4. **Upload to S3**: The processed Parquet file is stored at:
   ```
   s3://nashville311-gis-analysis-data/processed-data/
   ```

## Current Data

The latest processed file contains:
- **Time Period**: Last 3 months (dynamically calculated)
- **Format**: Parquet (`.parquet`)
- **Location**: S3 bucket `processed-data/` folder
- **Example**: `nashville311_last3months_2025-07_to_2025-10_20251016_050742.parquet`

## Why Parquet Instead of JSON?

- **Storage**: Much smaller files (4.1 MB vs ~45 MB for JSON)
- **Speed**: Faster to read and process
- **Analytics**: Columnar format optimized for data analysis
- **Production**: Industry standard for data pipelines

## How It's Used

The processed Parquet files are automatically loaded by our GeoPackage generators to create GIS visualizations. You don't need to download or manage these files manually—the Python scripts handle everything.

## Running the Pipeline

To generate fresh processed data, run:
```bash
python3 main.py
```

This fetches the latest 3 months, processes it, and uploads to S3 automatically.

