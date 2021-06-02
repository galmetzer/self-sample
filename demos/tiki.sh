#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.0001 --name density \
--iterations 20040 \
--export-interval 500 \
--pc data/tiki.xyz \
--init-var 0.15 \
--D1 6000 --D2 6000 \
--save-path demo-results/tiki \
--sampling-mode density \
--kmeans \
--batch-size 2 \
--k 15 \
--p1 0.85 --p2 0.1 \
--force-normal-estimation


python -u $BASE_PATH/inference.py \
--lr 0.001 --name tiki_result \
--iterations 25 \
--export-interval 50 \
--pc data/tiki.xyz \
--init-var 0.15 \
--D1 6000 --D2 6000 \
--save-path demo-results/tiki \
--generator demo-results/tiki/generators/model7500.pt \
--sampling-mode density \
--kmeans \
--batch-size 2 \
--k 15 \
--p1 0.85 --p2 0.1 \
--cat-input --downsample 100000 \
--force-normal-estimation