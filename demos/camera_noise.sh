#!/bin/bash

export BASE_PATH=$(pwd)
export PYTHONPATH=$BASE_PATH

python -u $BASE_PATH/main.py \
--lr 0.005 --name camera_noise \
--iterations 40017 \
--export-interval 500 \
--pc data/camera_noised.xyz \
--init-var 0.15 \
--D1 4000 --D2 4000 \
--save-path demo-results/camera_noise \
--sampling-mode random \
--batch-size 8 \
--k 40 \
--force-normal-estimation \
--mse


python -u $BASE_PATH/inference.py \
--name camera_noise_result \
--iterations 20 \
--export-interval 100 \
--pc data/camera_noised.xyz \
--D1 2000 --D2 2000 \
--save-path demo-results/camera_noise \
--generator demo-results/camera_noise/generators/model5000.pt \
--sampling-mode sweep
