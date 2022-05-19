#! /bin/bash
for i in {1..1000}
do
    blenderproc run shadows_dataset.py ../paper_datasets/HDRI_worktoptexture_shadows/random/dataset$1 /home/avena/Dropbox/3Dobj
    sleep 1
done