#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.0005 --name test-density \
--iterations 40040 \
--export-interval 500 \
--pc data/candle.xyz \
--init-var 0.15 \
--D1 5000 --D2 5000 \
--save-path demo-results/candle \
--sampling-mode density \
--kmeans \
--batch-size 4 \
--k 10 \
--p1 0.9 --p2 0.1 \
--force-normal-estimation


python -u $BASE_PATH/inference.py \
--lr 0.001 --name candle_result \
--iterations 5 \
--export-interval 50 \
--pc data/candle.xyz \
--init-var 0.15 \
--D1 5000 --D2 5000 \
--save-path demo-results/candle \
--generator demo-results/candle/generators/model10000.pt \
--sampling-mode density \
--kmeans \
--batch-size 4 \
--k 10 \
--p1 0.9 --p2 0.1 \
--cat-input --downsample 100000 \
--force-normal-estimation