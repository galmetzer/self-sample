#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

# train first part with random subset to reduce noise and filter outliers
python -u $BASE_PATH/main.py \
--lr 0.001 --name scanned_leg_stage1 \
--iterations 161000 \
--export-interval 500 \
--pc data/scanned_leg.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage1 \
--sampling-mode sweep \
--kmeans \
--batch-size 40 \
--k 10 \
--force-normal-estimation

# generate a new clean pointset
python -u $BASE_PATH/inference.py \
--lr 0.001 --name scanned_leg_stage1_result \
--iterations 50 \
--export-interval 100 \
--pc data/scanned_leg.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage1 \
--generator demo-results/scanned_leg_stage1/generators/model4000.pt \
--sampling-mode sweep \
--force-normal-estimation \
--downsample 30000

# train with density sampling to fill in holes
python -u $BASE_PATH/main.py \
--lr 0.001 --name scanned_leg_stage2 \
--iterations 161000 \
--export-interval 500 \
--pc demo-results/scanned_leg_stage1/scanned_leg_stage1_result.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage2 \
--sampling-mode density \
--kmeans \
--batch-size 40 \
--k 10 \
--p1 0.95 --p2 0.1 \
--force-normal-estimation

# generate a new sparse consolidated set and append the input
python -u $BASE_PATH/inference.py \
--lr 0.001 --name scanned_leg_stage2_result \
--iterations 200 \
--export-interval 100 \
--pc demo-results/scanned_leg_stage1/scanned_leg_stage1_result.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage2 \
--generator demo-results/scanned_leg_stage2/generators/model4000.pt \
--sampling-mode density \
--kmeans \
--p1 0.95 --p2 0.2 \
--batch-size 2 \
--k 10 \
--force-normal-estimation \
--cat-input --downsample 20000

# train a second pass with density sampling to fill in holes
python -u $BASE_PATH/main.py \
--lr 0.001 --name scanned_leg_stage3 \
--iterations 161000 \
--export-interval 500 \
--pc demo-results/scanned_leg_stage2/scanned_leg_stage2_result.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage3 \
--sampling-mode density \
--kmeans \
--batch-size 40 \
--k 10 \
--p1 0.9 --p2 0.1 \
--force-normal-estimation

# generate a new sparse consolidated set and append the input
python -u $BASE_PATH/inference.py \
--lr 0.001 --name scanned_leg_stage3 \
--iterations 200 \
--export-interval 100 \
--pc demo-results/scanned_leg_stage2/scanned_leg_stage2_result.xyz \
--init-var 0.15 \
--D1 500 --D2 500 \
--save-path demo-results/scanned_leg_stage3 \
--generator demo-results/scanned_leg_stage3/generators/model4000.pt \
--sampling-mode density \
--kmeans \
--p1 0.95 --p2 0.2 \
--batch-size 2 \
--k 10 \
--force-normal-estimation \
--cat-input --downsample 20000