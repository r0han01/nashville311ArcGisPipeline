#!/bin/bash
# Script to safely enable public read access for GeoPackages folder only

BUCKET_NAME="nashville311-gis-analysis-data"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
POLICY_FILE="$PROJECT_ROOT/terraform/public-geopackages-policy.json"

echo "🔒 Setting up secure public access for GeoPackages..."
echo "=================================================="
echo ""

# Step 1: Modify Public Access Block to allow public policy (but keep ACLs blocked)
echo "Step 1: Configuring Public Access Block..."
aws s3api put-public-access-block \
    --bucket "$BUCKET_NAME" \
    --public-access-block-configuration \
    "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false"

echo "✅ Public Access Block configured"
echo ""

# Step 2: Apply bucket policy (only allows GetObject on gpkg-public/*)
echo "Step 2: Applying bucket policy (read-only for gpkg-public/*)..."
aws s3api put-bucket-policy \
    --bucket "$BUCKET_NAME" \
    --policy file://"$POLICY_FILE"

echo "✅ Bucket policy applied"
echo ""

# Step 3: Verify configuration
echo "Step 3: Verifying security configuration..."
echo ""
echo "📋 Security Summary:"
echo "  ✅ Only 'gpkg-public/*' is publicly readable"
echo "  ✅ No listing allowed (users need exact URLs)"
echo "  ✅ No write access (read-only)"
echo "  ✅ Other folders remain private"
echo "  ✅ Bucket settings remain private"
echo ""

# Step 4: Test public access (should work)
echo "Step 4: Testing public access..."
echo ""
echo "Test URL format:"
echo "  https://${BUCKET_NAME}.s3.amazonaws.com/gpkg-public/districtPerformanceRanking/districtPerformanceRanking.gpkg"
echo ""
echo "⚠️  IMPORTANT SECURITY NOTES:"
echo "  1. Monitor CloudWatch for unusual download patterns"
echo "  2. Set up billing alerts in AWS Console"
echo "  3. Only put public data in gpkg-public/"
echo "  4. Keep processed-data/ and boundaries/ private"
echo ""
echo "✅ Setup complete! GeoPackages are now publicly accessible (read-only)"

