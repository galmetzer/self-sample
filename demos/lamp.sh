#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.005 --name test-density \
--iterations 10004 \
--export-interval 1000 \
--pc data/lamp.xyz \
--init-var 0.15 \
--D1 8000 --D2 8000 \
--save-path demo-results/lamp \
--sampling-mode curvature \
--batch-size 2 \
--k 40 \
--p1 0.85 --p2 0.2 \
--force-normal-estimation --mse

python -u $BASE_PATH/inference.py \
--lr 0.001 --name lamp_result \
--iterations 25 \
--export-interval 100 \
--pc data/lamp.xyz \
--init-var 0.15 \
--D1 8000 --D2 8000 \
--save-path demo-results/lamp \
--generator demo-results/lamp/generators/model5000.pt \
--sampling-mode curvature \
--p1 0.8 --p2 0.2 \
--batch-size 2 \
--k 40 \
--force-normal-estimation
