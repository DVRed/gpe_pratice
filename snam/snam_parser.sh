#!/bin/sh
python3 snam_loader.py

#uncomment if you want to save xlsx-file to working directory
cp *.xlsx /tmp

cd /tmp
ls -n