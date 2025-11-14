# Raw Nashville 311 Data

## Data Source

We fetch raw service request data directly from the **Nashville 311 ArcGIS API**:

**API URL:** `https://services2.arcgis.com/HdTo6HJqh92wn4D8/arcgis/rest/services/hubNashville_311_Service_Requests_Current_Year_view/FeatureServer/0/query`

This is the official public API that provides real-time Nashville 311 service requests.

## What the Data Looks Like

The API returns data in GeoJSON format. Each record contains:

```json
{
  "type": "Feature",
  "geometry": {
    "type": "Point",
    "coordinates": [-86.7816, 36.1627]
  },
  "properties": {
    "Request_Type": "Cart Service",
    "Status": "Closed",
    "Council_District": 12,
    "Address": "123 Main St",
    "City": "Nashville",
    "Date_Time_Opened": 1729123200000,
    "Date_Time_Closed": 1729126800000,
    "Latitude": 36.1627,
    "Longitude": -86.7816,
    "ZIP": "37203"
  }
}
```

## Key Fields

- **Request_Type**: Type of service request (Cart Service, Noise Complaint, etc.)
- **Status**: Current status (Open, Closed, In Progress)
- **Council_District**: District number (1-35)
- **Date_Time_Opened/Closed**: Timestamps in milliseconds
- **Latitude/Longitude**: Geographic coordinates
- **Address, City, ZIP**: Location information

## Processing

Raw JSON data from the API is automatically converted to **Parquet format** for faster processing and storage. The Parquet files are stored in the `processed-data/` folder in S3.

**Why Parquet?** It's 11.5x smaller and much faster to read than JSON, making it perfect for production data pipelines.

## Storage

Raw data isn't stored locally—it's fetched on-demand and immediately processed. The processed Parquet files are stored in S3 at:
```
s3://nashville311-gis-analysis-data/processed-data/
```

