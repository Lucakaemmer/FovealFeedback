# conda activate foveal_decoding

subjects=()
subject_ids=()


echo "Extracting deliniation"
for i in "${!subjects[@]}"
do
    # Set global variables
    export sub="${subjects[i]}"
    export sub_id="${subject_ids[i]}"

    echo $sub

    SUBJECTS_DIR="subject directory" python -m neuropythy atlas $sub_id &
done
wait


echo "Reconstructing volume"
for i in "${!subjects[@]}"
do
    # Set global variables
    export sub="${subjects[i]}"
    export sub_id="${subject_ids[i]}"

    echo $sub

    if [ ! -d "directory to store masks" ]; then
        mkdir -p "directory to store masks"
    fi 

    export freesurfer_dir = "subject's directory to freesurfer data"
    export mask_dir = "subject's directory to masks"

    # Reconstructing anatomical volume
    python -m neuropythy surface_to_image $freesurfer_dir \
        $mask_dir/vol_vareas.nii.gz \
        --lh $freesurfer_dir/surf/lh.benson14_varea.mgz \
        --rh $freesurfer_dir/surf/rh.benson14_varea.mgz &

    # Reconstructing Benson Eccentricity
    python -m neuropythy surface_to_image $freesurfer_dir \
        $mask_dir/vol_eccen.nii.gz \
        --lh $freesurfer_dir/surf/lh.benson14_eccen.mgz \
        --rh $freesurfer_dir/surf/rh.benson14_eccen.mgz &

done
wait