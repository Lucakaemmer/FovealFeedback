#!/bin/bash

# List of subjects
subjects=()

# Directories
data_nii_dir="nifti data directory"
data_bids_dir="bids data directory"
data_phase_dir="phase data directory"
data_pre_dir="preprocessed data directory"
data_work_dir="working directory"
fmriprep_license_dir="fmriprep licence directory"

# Clean out working directory
rm -rf $data_work_dir/*

for sub in "${subjects[@]}"
do
    echo "Processing $sub..."

    # Creating directories to store bids data, and preprocessed data
    mkdir -p "$data_bids_dir/$sub"
    mkdir -p "$data_pre_dir/$sub"
    mkdir -p "$data_work_dir/$sub"

    # Copy the files to a phase directory and delete phase data from nii data directory
    if [ ! -d "$data_phase_dir/$sub" ]; then
      cp -r "$data_nii_dir/$sub" "$data_phase_dir"
    fi
    find "$data_nii_dir/$sub" -type f -name '*_ph*' -exec rm {} +

    # Converting files to bids format
    niix2bids -i $data_nii_dir/$sub -o $data_bids_dir/$sub --copyfile

    echo "starting docker"

    # Runs docker in the background, use -ti instead of -d to run in interactive terminal
    docker run -d --rm \
    -u $(id -u):$(id -g) \
    -v $data_bids_dir/$sub:/data:ro \
    -v $data_pre_dir/$sub:/out \
    -v $data_work_dir/$sub:/work \
    -v $fmriprep_license_dir:$fmriprep_license_dir \
    poldracklab/fmriprep:20.2.0 \
    --output-spaces MNI152NLin2009cAsym T1w func \
    --bold2t1w-dof 9 \
    --fs-license-file $fmriprep_license_dir/license.txt \
    --use-aroma -w /work /data /out participant \
    --skip-bids-validation
done
