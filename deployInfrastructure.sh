#!/bin/bash

set -e

cd terraform

terraform init
terraform plan
terraform apply -auto-approve

terraform output