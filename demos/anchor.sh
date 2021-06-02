#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.005 \
--iterations 10004 \
--export-interval 500 \
--pc data/anchor.xyz \
--init-var 0.15 \
--D1 8000 --D2 8000 \
--save-path demo-results/anchor \
--sampling-mode curvature \
--batch-size 2 \
--k 20 \
--p1 0.85 --p2 0.2 \
--force-normal-estimation \
--mse


python -u $BASE_PATH/inference.py \
--lr 0.001 --name anchor_result \
--iterations 10 \
--export-interval 100 \
--pc data/anchor.xyz \
--init-var 0.15 \
--D1 8000 --D2 8000 \
--save-path demo-results/anchor \
--generator demo-results/anchor/generators/model5000.pt \
--sampling-mode curvature \
--batch-size 2 \
--k 20 \
--force-normal-estimation
