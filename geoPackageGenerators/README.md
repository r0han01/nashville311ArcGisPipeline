## GeoPackage Generators Overview

This directory contains every Python generator that produces the final ArcGIS-ready GeoPackages (`*.gpkg`). Each generator pulls the latest three months of Nashville 311 data from S3, calculates the required metrics with fully data-driven methods (no hardcoded thresholds), writes the GeoPackage locally, and uploads it back to our public bucket.

### Why GeoPackages?
- **Multi-layer friendly:** Each `.gpkg` stores both the district polygon layer and the service request point layer, so analysts never have to re-join tables in ArcGIS Pro.
- **Attribute-driven symbology:** GeoPackages keep numeric precision, field aliases, and camelCase column names intact—ideal for attribute-driven symbology, charting, 3D scenes, and ModelBuilder workflows.
- **Cloud ready:** Finished files are automatically uploaded to the S3 prefix `s3://nashville311-gis-analysis-data/gpkg-public/<generatorName>/<generatorName>.gpkg` and can be shared or downloaded directly.

### Data snapshot used for the current run
- **Temporal coverage:** Rolling three months ending **2025‑11‑24** (latest data pull at run time).
- **Volume:** ~49,000 service requests and 35 council districts per GeoPackage.
- **Re-running later:** When a teammate runs any generator at a different time, it will automatically pull the latest three months from S3. No code changes needed—just ensure AWS credentials point to the correct bucket or update the generator’s `bucketName` argument for alternative storage (e.g., your own S3 bucket or a local folder path if you skip the upload step).

### Folder & generator reference
```
/home/r0han/personalProjects/arcGIS/geoPackageGenerators
├── capacityPlanning/                → capacityPlanning.gpkg
├── districtPerformanceRanking/      → historical ranking reference (legacy)
├── geographicPatterns/              → geographicPatterns.gpkg
├── optimizationOpportunities/       → optimizationOpportunities.gpkg
├── peakHourConsistency/             → peakHourConsistency.gpkg
├── performanceQuartiles/            → historical quartiles reference (legacy)
├── predictiveHotspots/              → predictiveHotspots.gpkg
├── relativeWorkload/                → historical workload reference (legacy)
├── requestDensityZones/             → historical density reference (legacy)
├── requestVolatility/               → requestVolatility.gpkg
├── serviceEfficiency/               → serviceEfficiency.gpkg
├── serviceTypeInsights/             → serviceTypeInsights.gpkg
├── temporalPatterns/                → temporalPatterns.gpkg
├── timeBasedClusters/               → timeBasedClusters.gpkg
└── trendDirection/                  → trendDirection.gpkg
```
> **Tip:** The “legacy” folders remain for historical context but the active 10-GeoPackage roadmap is driven by the bolded `.gpkg` outputs above.

### Generator highlights
| GeoPackage & script | What it explains | Key highlights | GIS download |
|---------------------|------------------|----------------|----------|
| `serviceEfficiency.gpkg` (`serviceEfficiency/serviceEfficiency.py`) | Where districts deliver the fastest responses relative to workload. | Efficiency score = workload ratio ÷ response-time ratio, quartiles, percentile ranks, request context for pop-ups. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/serviceEfficiency/serviceEfficiency.gpkg) |
| `geographicPatterns.gpkg` (`geographicPatterns/geographicPatterns.py`) | Spatial density, clustering, concentration, variability. | Silhouette + Quantiles for categories, centroid-based stats, request distribution fields. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/geographicPatterns/geographicPatterns.gpkg) |
| `temporalPatterns.gpkg` (`temporalPatterns/temporalPatterns.py`) | Weekday vs. weekend behavior by district. | Day-of-week breakdowns, weekday/weekend ratios, consistency measures, percentile labels. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/temporalPatterns/temporalPatterns.gpkg) |
| `peakHourConsistency.gpkg` (`peakHourConsistency/peakHourConsistency.py`) | Peak vs. off-hour service demand. | Hour-of-day profiles, coefficient-of-variation metrics, Silhouette-based peak pattern types. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/peakHourConsistency/peakHourConsistency.gpkg) |
| `requestVolatility.gpkg` (`requestVolatility/requestVolatility.py`) | Stability of request volumes. | Daily/weekly/monthly volatility coefficients, stability indexes, data-driven volatility categories. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/requestVolatility/requestVolatility.gpkg) |
| `timeBasedClusters.gpkg` (`timeBasedClusters/timeBasedClusters.py`) | Temporal clustering of districts. | K-means with silhouette-selected k, cluster metrics (distance, confidence, dominant patterns). | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/timeBasedClusters/timeBasedClusters.gpkg) |
| `serviceTypeInsights.gpkg` (`serviceTypeInsights/serviceTypeInsights.py`) | Service mix, recurring issues, complexity. | Diversity/concentration (Herfindahl), recurring issue detection, Silhouette-based categories. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/serviceTypeInsights/serviceTypeInsights.gpkg) |
| `trendDirection.gpkg` (`trendDirection/trendDirection.py`) | Increasing vs. decreasing request trends. | Monthly regression, slope/R²/p-value, projected next month, trend confidence scores. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/trendDirection/trendDirection.gpkg) |
| `predictiveHotspots.gpkg` (`predictiveHotspots/predictiveHotspots.py`) | Future risk assessment based on multi-factor scoring. | Variance-based risk weights, dominant risk factors, percentile-based priority labels. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/predictiveHotspots/predictiveHotspots.gpkg) |
| `capacityPlanning.gpkg` (`capacityPlanning/capacityPlanning.py`) | Who is at/near capacity and when they’ll hit limits. | Median + MAD + IQR thresholds, utilization & pressure, time-to-capacity, Silhouette categories. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/capacityPlanning/capacityPlanning.gpkg) |
| `optimizationOpportunities.gpkg` (`optimizationOpportunities/optimizationOpportunities.py`) | Biggest improvement opportunities and expected impact. | Response/efficiency/workload gaps, volatility impact, variance-based opportunity scores, priority/category labels. | [🔗](https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/optimizationOpportunities/optimizationOpportunities.gpkg) |

### Running a generator locally or in another cloud account
1. **Set credentials:** Ensure your AWS credentials (or equivalent cloud storage creds) are configured. By default the scripts use `NashvilleConfig.BUCKET_NAME`.
2. **Override storage (optional):**
   ```python
   from geoPackageGenerators.capacityPlanning.capacityPlanning import CapacityPlanningGenerator

   generator = CapacityPlanningGenerator(bucketName='my-custom-bucket')
   generator.createShapefile()  # Uploads to your bucket’s gpkg-public prefix
   ```
   If you want to skip uploads entirely, comment out `uploadGeoPackageToS3` and copy the generated `.gpkg` directly from the temp directory printed during runtime.
3. **Customize prefixes:** Each generator sets its own `self.s3ShapefilePrefix`; adjust if you need separate folders per environment.

### How to find a generated GeoPackage
Every generator prints the final HTTPS URL when it completes. Example for capacity planning:
```
https://nashville311-gis-analysis-data.s3.amazonaws.com/gpkg-public/capacityPlanning/capacityPlanning.gpkg
```
Download the file, add it to ArcGIS Pro, and you’ll see:
- **Layer 1:** District polygons with camelCase attributes tailored to the analysis.
- **Layer 2:** Service request points already joined with district context to power pop-ups, filters, charts, and 3D scenes without extra joins.

### Keep in mind
- **Data freshness:** The numbers will change every run because everything is percentile-, median-, or variance-based off the current three-month slice.
- **ArcGIS techniques:** Use attribute-driven symbology, spatial statistics tools (Hot Spot, Moran’s I), 3D scenes, ModelBuilder, and map series described in the individual generator docstrings for advanced visualization.
- **Extensibility:** Each `.py` script is fully self-contained—feel free to copy, tweak input parameters (e.g., different time windows), or swap the storage target to fit your environment.

Need help interpreting a specific GeoPackage? Open the corresponding `.py` file for inline documentation and examples of what each column means.

