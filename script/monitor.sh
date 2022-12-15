#!/bin/bash

tensorboard dev upload --logdir ./tensorboard \
                       --name "crop_classification" \
                       --description "2022AICUP"
