#!/bin/bash

for FILES_PATH in $1
do
	echo $FILES_PATH
	python run.py -i $FILES_PATH --save_png False
done
