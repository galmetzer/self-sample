#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.001 \
--iterations 16004 \
--export-interval 1000 \
--pc data/alien140.xyz \
--init-var 0.15 \
--D1 10000 --D2 10000 \
--save-path demo-results/alien \
--sampling-mode curvature \
--batch-size 1 \
--k 40 \
--p1 0.85 --p2 0.2 \
--force-normal-estimation


python -u $BASE_PATH/inference.py \
--lr 0.001 --name alien_result \
--iterations 15 \
--export-interval 100 \
--pc data/alien140.xyz \
--init-var 0.15 \
--D1 10000 --D2 10000 \
--save-path demo-results/alien \
--generator demo-results/alien/generators/model16000.pt \
--sampling-mode curvature \
--p1 0.85 --p2 0.2 \
--batch-size 1 \
--k 40 \
--force-normal-estimation