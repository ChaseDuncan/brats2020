

for N in 1 2 3 4 5 6 7 8
do
    python annotate_osf.py --dir data/models/largepatch/ --output_dir /shared/mrfil-data/cddunca2/GliomaOSF/GLI00$N/segmentation/ --data_dir /shared/mrfil-data/cddunca2/GliomaOSF/GLI00$N/ -c 300 -g 1 -L --model MonoUNet

done
