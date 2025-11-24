# Nashville 311 GeoPackage Pipeline

Concise documentation for the ArcGIS automation project that converts rolling three-month HubNashville 311 data into analysis-ready GeoPackages plus step-by-step ArcGIS usage guidance.

## What this repo contains

- Python generators under `geoPackageGenerators/` that pull the latest parquet snapshot from S3, calculate data-driven metrics, and publish `*.gpkg` files with polygon and point layers.
- A shared `README` in the GeoPackage folder with per-file summaries, output URLs, and rerun instructions.
- Automation utilities in `src/` for authentication, S3 transfers, and boundary management.
- ArcGIS Pro instructions, ModelBuilder recipes, and layout notes captured throughout the repo history.

## Data source

- **API:** HubNashville Service Requests  
  https://data.nashville.gov/Public-Services/HubNashville-Service-Requests/9udr-2rbj  
  (JSON, CSV, and Socrata API supported)
- **S3 parquet staging:** `s3://nashville311-gis-analysis-data/processed-data/<timestamp>.parquet`
- **GeoPackage output prefix:** `s3://nashville311-gis-analysis-data/gpkg-public/<generatorName>/<generatorName>.gpkg`
- **Current snapshot:** Rolling three months ending 2025‑11‑24 (~49k requests across 35 districts)

## How the pipeline runs

1. **Fetch latest parquet:** Each generator loads the most recent file in the S3 `processed-data/` prefix.
2. **Compute metrics:** All classifications use medians, percentiles, coefficients of variation, and silhouette-guided quantiles so nothing is hardcoded.
3. **Write GeoPackage:** District polygons and service request points are written to a temporary `.gpkg`.
4. **Upload:** The GeoPackage is pushed to the public S3 prefix; the console prints the HTTPS URL for ArcGIS download.

## Re-running a generator

```bash
cd /home/r0han/personalProjects/arcGIS
python3 geoPackageGenerators/capacityPlanning/capacityPlanning.py
```

Optional overrides (cloud vs local):

```python
from geoPackageGenerators.predictiveHotspots.predictiveHotspots import PredictiveHotspotsGenerator

generator = PredictiveHotspotsGenerator(bucketName='my-alt-bucket')
generator.createShapefile()  # writes locally then uploads to the specified bucket
```

To skip uploads entirely, comment out `uploadGeoPackageToS3` inside the generator and copy the local file path printed during runtime.

## API schema quick look

| Field | Example | Notes |
|-------|---------|-------|
| `Request_ID` | `23-001234` | Primary key |
| `Request_Type` | `Public Works WO` | High level grouping |
| `Subrequest_Type` | `Cart Services` | Specific issue |
| `Status` | `Closed` | Also includes New, Assigned, Pending |
| `Date_Time_Opened` | `1698710400000` | Epoch milliseconds |
| `Date_Time_Closed` | `1698796800000` | Epoch milliseconds, null if open |
| `Council_District` | `5` | Joined to boundary shapefile |
| `Latitude`,`Longitude` | `36.1678`, `-86.7784` | Used for point geometry |
| `Address` | `123 MAIN ST` | Cleansed to uppercase |

All joins and enrichments happen inside the generators so analysts only need the final GeoPackages.

## Key GeoPackages

| File | Focus | Where it lives |
|------|-------|----------------|
| `serviceEfficiency.gpkg` | Response vs workload efficiency | `gpkg-public/serviceEfficiency/` |
| `geographicPatterns.gpkg` | Density, clustering, variability | `gpkg-public/geographicPatterns/` |
| `temporalPatterns.gpkg` | Weekday vs weekend trends | `gpkg-public/temporalPatterns/` |
| `peakHourConsistency.gpkg` | Peak vs off-hour demand | `gpkg-public/peakHourConsistency/` |
| `requestVolatility.gpkg` | Stability of demand | `gpkg-public/requestVolatility/` |
| `timeBasedClusters.gpkg` | Temporal K-means clusters | `gpkg-public/timeBasedClusters/` |
| `serviceTypeInsights.gpkg` | Mix, recurring issues, complexity | `gpkg-public/serviceTypeInsights/` |
| `trendDirection.gpkg` | Upward vs downward trends | `gpkg-public/trendDirection/` |
| `predictiveHotspots.gpkg` | Multi-factor risk scoring | `gpkg-public/predictiveHotspots/` |
| `capacityPlanning.gpkg` | Utilization and time-to-capacity | `gpkg-public/capacityPlanning/` |
| `optimizationOpportunities.gpkg` | Highest impact improvements | `gpkg-public/optimizationOpportunities/` |

Open `geoPackageGenerators/README.md` for deeper per-file highlights.

## ArcGIS Pro workflow snapshot

Screenshots go here:

1. _[Add screenshot of map view or layout]_
2. _[Add screenshot of symbology or chart configuration]_

Typical sequence:

1. Download the desired `.gpkg` via the HTTPS URL printed after running a generator.
2. Drag the GeoPackage into ArcGIS Pro, confirm polygon and point layers load with camelCase fields.
3. Use the recommended techniques (Optimized Hot Spot, Cluster and Outlier, 3D scenes, map series) described in the generator docstrings.
4. Publish layouts, dashboards, or 3D scenes as needed.

## Contributing

- Keep column names in camelCase across all generators.
- Add new GeoPackages under `geoPackageGenerators/<name>/` with a dedicated script and update the folder README.
- Follow the data-driven rule: no hardcoded categories, weights, or thresholds.
- Run `read_lints` and regenerate the GeoPackage before committing.

Questions or ideas? Open an issue or ping the project maintainers listed in `src/nashvilleGis/config.py`.

