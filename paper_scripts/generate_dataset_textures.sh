#! /bin/bash

for i in {1..1000}
do
    blenderproc run scenario1_test.py ../paper_datasets/HDRI_worktoptexture/pattern/dataset$1 /home/avena/Dropbox/3Dobj
    sleep 1
done