#!/bin/bash

directory="../pia_results"

cd src

for file in $directory/*; do   
    echo "python classification.py $file"
    python classification.py $file
done
