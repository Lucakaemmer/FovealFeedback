#!/bin/bash

# List of subjects
subjects=()
subject_ids=()
runs=("1" "2" "3" "4" "5" "6" "7" "8" "9" "10")

# Directories
data_pre_dir="preprocessed data directory"
data_out_dir="output directory"
feat_directory_decoding="decoding feat scripts"
feat_directory_ret="retinotopy feat scripts"
feat_directory_object="object localiser feat scripts"

# Select which analyses to run
decoding=True
retinotopy=True
object_localizer=True

for i in "${!subjects[@]}"
do
    sub="${subjects[i]}"
    sub_id="${subject_ids[i]}"
    sub_data="${data_pre_dir}/${sub}/fmriprep/${sub_id}/ses-1"

    if [ ! -d "$data_out_dir/${sub}" ]; then
        mkdir -p "$data_out_dir/${sub}"
    fi 

    ### OBJECT LOCALIZER ###
    if [ "$object_localizer" = "True" ]; then

        export OBJ_OUT_DIR="$data_out_dir/${sub}/obj_loc"
        export OBJ_IN_DIR="$sub_data/func/${sub_id}_ses-1_task-cmrrmbep2dbold2isorun12_dir-AP_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        if [ ! -f "$OBJ_IN_DIR" ]; then
            echo "Error: Subject $sub has no object localizer run"
            continue  # Skip this iteration
        fi  

        # Copy the template to a new file
        obj_loc_template_fsf="$feat_directory_object/obj_loc_template.fsf"
        obj_loc_fsf="$feat_directory_object/obj_loc_${sub}.fsf"
        cp "$obj_loc_template_fsf" "$obj_loc_fsf"  

        sed -i "s|@OBJ_OUT_DIR@|$OBJ_OUT_DIR|g" "$obj_loc_fsf"
        sed -i "s|@OBJ_IN_DIR@|$OBJ_IN_DIR|g" "$obj_loc_fsf"

        feat "$obj_loc_fsf" &
        echo "running object localizer for $sub"
    fi

    ### RETINOTOPY ###
    # DEGREES
    if [ "$retinotopy" = "True" ]; then

        # Set data input and output directory for the participant for the retinotopy
        export RET_OUT_DIR="$data_out_dir/${sub}/retinotopy"
        export RET_IN_DIR="$sub_data/func/${sub_id}_ses-1_task-cmrrmbep2dbold2isorun11_dir-AP_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
        if [ ! -f "$RET_IN_DIR" ]; then
            echo "Error: Subject $sub has no retinotopic run"
            continue  # Skip this iteration
        fi  

        # Copy the template to a new file
        retinotopy_template_fsf="$feat_directory_ret/full_retinotopy_template.fsf"
        retinotopy_fsf="$feat_directory_ret/full_retinotopy_${sub}.fsf"
        cp "$retinotopy_template_fsf" "$retinotopy_fsf"  

        sed -i "s|@RET_OUT_DIR@|$RET_OUT_DIR|g" "$retinotopy_fsf"
        sed -i "s|@RET_IN_DIR@|$RET_IN_DIR|g" "$retinotopy_fsf"

        feat "$retinotopy_fsf" &
        echo "running full retinotopy for $sub"

        if [ ! -d "$data_out_dir/${sub}/masks" ]; then
            mkdir -p "$data_out_dir/${sub}/masks"
        fi 
    fi

    ### FOVEAL DECODING ###
    # BLOCK-DESIGN
    if [ "$decoding" = "True" ]; then
        for j in "${!runs[@]}"
        do  
            run="${runs[j]}"

            # Set data input and output directory for the participant for each run
            export DATA_OUT_DIR="$data_out_dir/${sub}/foveal_decoding/run_${run}"
            export DATA_IN_FUNC_DIR="$sub_data/func/${sub_id}_ses-1_task-cmrrmbep2dbold2isorun${run}_dir-AP_run-1_space-MNI152NLin2009cAsym_desc-preproc_bold.nii.gz"
            if [ ! -f "$DATA_IN_FUNC_DIR" ]; then
                echo "Error: Subject $sub has no run $run"
                continue  # Skip this iteration
            fi  

            # Copy the template to a new file
            foveal_decoding_template_fsf="$feat_directory_decoding/univariate_decoding_template.fsf"
            foveal_decoding_fsf="$feat_directory_decoding/univariate_decoding_${sub}_${run}.fsf"
            cp "$foveal_decoding_template_fsf" "$foveal_decoding_fsf"

            # Replacing input and output directories in the fsf file
            sed -i "s|@DATA_OUT_DIR@|$DATA_OUT_DIR|g" "$foveal_decoding_fsf"
            sed -i "s|@DATA_IN_FUNC_DIR@|$DATA_IN_FUNC_DIR|g" "$foveal_decoding_fsf"

            feat "$foveal_decoding_fsf" &
        done
        echo "running foveal decoding for $sub"
    fi

done
wait
