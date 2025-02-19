import glob
import re
import os 
import numpy as np
import pandas as pd
import nibabel as nib
from nilearn.image import concat_imgs
from nilearn.image import resample_to_img
from utils.const import SUBJECTS, SUBJECT_IDS, ALL_DEGREES, LABELS

class PrepareData:
    """
    Imports the processed data (z-scores), prepares them for decoding, and trains the decoding models.

    Attributes:
        labels (list): List of labels.
        subjects (list): List of subjects.
        subject_ids (list): List of subject IDs.
        subject (int): Current subject.
        subject_id (int): Current subject ID.
        out_dir (str): Output directory.
        func_dir (str): Functional directory.
        ret_dir (str): Retinotopy directory.
        eye_dir (str): Eye data directory.
        runs (list): List of runs.
        images_raw (dict): Raw images.
        data_raw (dict): Raw data.
        data_masked (dict): Masked data.
        white_cov_matrix (ndarray): White covariance matrix.
        ret_mask_data (ndarray): Retinotopy mask data.
        anat_mask_data (ndarray): Anatomy mask data.
        eccen_mask_data (ndarray): Eccentricity mask data.
        control_mask_data (ndarray): Control mask data.
        object_mask_data (ndarray): Object mask data.
        full_mask_data (ndarray): Full mask data.
        full_dataset (dict): Full dataset.
    """
    
    def __init__(self):
        """
        Initializes the PrepareData class with default values.
        """
        self.labels = LABELS
        self.subjects = SUBJECTS
        self.subject_ids = SUBJECT_IDS
        self.subject = 0
        self.subject_id = 0
        self.out_dir = 0
        self.func_dir = 0
        self.ret_dir = 0
        self.eye_dir = 0
        self.runs = 0
        self.images_raw = 0
        self.data_raw = 0
        self.data_masked = 0
        self.white_cov_matrix = 0
        self.ret_mask_data = 0
        self.anat_mask_data = 0
        self.eccen_mask_data = 0
        self.control_mask_data = 0
        self.object_mask_data = 0
        self.ret_mask_data = 0
        self.full_mask_data = 0
        self.full_dataset = {}
    
    def prepare_all_data(self, vareas=list, degrees=list, event_related=bool, exclude=str, anatomy=True, retinotopy=True, frontal_control=False, eccen=False, object=False, print_mask=True):    
        """
        Prepares all data for all subjects.

        Parameters:
            vareas (list): List of visual areas.
            degrees (list): List of degrees.
            event_related (bool): Whether the data is event-related.
            exclude (str): Exclusion criteria.
            anatomy (bool): Whether to include anatomy mask.
            retinotopy (bool): Whether to include retinotopy mask.
            frontal_control (bool): Whether to include frontal control mask.
            eccen (bool): Whether to include eccentricity mask.
            object (bool): Whether to include object mask.
            print_mask (bool): Whether to print the mask information.

        Returns:
            dict: Full dataset.
        """
        for s in range(len(self.subjects)):
            self.setup_variables(s=s, event_related=event_related)
            self.import_data()
            self.create_mask(anatomy=anatomy, retinotopy=retinotopy, frontal_control=frontal_control, eccen=eccen, object=object, vareas=vareas, degrees=degrees, print_mask=print_mask)
            self.apply_mask()
            self.package_data(exclude=exclude, distance=2.5)
        return self.full_dataset
    
    def setup_variables(self, s, event_related):
        """
        Sets up variables for the current subject.

        Parameters:
            s (int): Subject index.
            event_related (bool): Whether the data is event-related.
        """
        self.subject = self.subjects[s]
        self.subject_id = self.subject_ids[s]
        self.out_dir = f"/BULK/lkaemmer/data/foveal_decoding/data_out/{self.subject}"
        if event_related:
            self.func_dir = f"{self.out_dir}/foveal_decoding_event/"
        else:
            self.func_dir = f"{self.out_dir}/foveal_decoding/"
        self.ret_dir = f"{self.out_dir}/retinotopy.feat/"
        self.eye_dir = f"/BULK/lkaemmer/data/foveal_decoding/data_eye/{self.subject}"
        self.runs = []
        self.images_raw = {}
        self.data_raw = {}
        self.data_masked = {}
        
    def import_data(self):
        """
        Imports the data for the current subject.
        """
        # Extracting names for the folders that contain the data for the different runs
        folder_names_unsorted = next(os.walk(self.func_dir))[1]
        folder_names = sorted(folder_names_unsorted, key=lambda x: int(re.search(r'run_(\d+)', x).group(1)))
        for folder in folder_names:
            match = re.search(r'run_(\d+)', folder)
            if match: self.runs.append(int(match.group(1)))
        # Extracting the names of the files containing the zscores for all the blocks within one run
        for run in range(len(folder_names)):
            files_dir = self.func_dir + folder_names[run] + "/stats"
            zstat_files_unsorted = glob.glob(os.path.join(files_dir, "zstat*"))
            zstat_files = sorted(zstat_files_unsorted, key=lambda x: int(re.search(r'zstat(\d+)', x).group(1)))
            # Importing the betas for each block and putting them together in a 4D file containing all the betas. This is done for all runs
            images = []
            for block_dir in (zstat_files):
                img = nib.load(block_dir)
                images.append(img)
            self.images_raw[self.runs[run]] = concat_imgs(images, auto_resample=True, ensure_ndim=4)
        # Extracting the raw data from the raw image files
        for run in self.runs:
            self.data_raw[run] = self.images_raw[run].get_fdata()

        
    def create_mask(self, anatomy=bool, retinotopy=bool, eccen=bool, frontal_control=bool, object=bool, vareas=list, degrees=list, print_mask=bool):
        """
        Creates the mask for the current subject.

        Parameters:
            anatomy (bool): Whether to include anatomy mask.
            retinotopy (bool): Whether to include retinotopy mask.
            eccen (bool): Whether to include eccentricity mask.
            frontal_control (bool): Whether to include frontal control mask.
            object (bool): Whether to include object mask.
            vareas (list): List of visual areas.
            degrees (list): List of degrees.
            print_mask (bool): Whether to print the mask information.
        """
        reference = self.create_anat_mask(vareas=vareas)
        self.full_mask_data = np.ones_like(self.anat_mask_data, dtype=int)
        if anatomy:
            self.full_mask_data = np.logical_and(self.anat_mask_data, self.full_mask_data).astype(np.float32)
        if retinotopy:
            reference = self.create_ret_mask(degrees=list(map(str, degrees)))
            self.full_mask_data = np.logical_and(self.ret_mask_data, self.full_mask_data).astype(np.float32)
        if eccen:
            reference = self.create_eccen_mask(degrees=degrees)
            self.full_mask_data = np.logical_and(self.eccen_mask_data, self.full_mask_data).astype(np.float32)
        if frontal_control:
            reference = self.create_control_mask()
            self.full_mask_data = np.logical_and(self.control_mask_data, self.full_mask_data).astype(np.float32)
        if object:
            reference = self.create_object_mask()
            self.full_mask_data = np.logical_and(self.object_mask_data, self.full_mask_data).astype(np.float32)
            
        if print_mask:
            print(f"Full mask of {self.subject} contains {int(np.sum(self.full_mask_data))} voxels")

        full_mask_image = nib.Nifti1Image(self.full_mask_data, reference.affine, reference.header)
        nib.save(full_mask_image, f"{self.out_dir}/masks/full_mask.nii.gz")
    
    def apply_mask(self):
        """
        Applies the mask to the data for the current subject.
        """
        # Apply mask to each run and flatten data
        mask_bool = self.full_mask_data.astype(bool)
        for run in self.runs:
            self.data_masked[run] = self.data_raw[run][mask_bool]
        
    def package_data(self, exclude=str, distance=float):
        """
        Packages the data for the current subject.

        Parameters:
            exclude (str): Exclusion criteria.
            distance (float): Distance threshold for exclusion.
        """
        all_runs = np.concatenate([np.repeat(run_id, self.data_masked[run_id].shape[1]) for run_id in self.data_masked])
        all_data = np.vstack([self.data_masked[run].T for run in self.data_masked])
        all_labels = np.tile(self.labels, len(self.runs))
        
        if exclude == 'online':
            exclude_blocks = self.exclude_blocks(saccade_count_exclude=False, gaze_distance_exclude=True, distance=distance, min_saccades=3, measure="post_flip")
            all_runs = all_runs[exclude_blocks]
            all_data = all_data[exclude_blocks]
            all_labels = all_labels[exclude_blocks]
            print(f"{np.count_nonzero(~exclude_blocks)} blocks excluded")

        self.full_dataset[self.subject] = {
            'data': all_data,
            'covariance': self.white_cov_matrix,
            'runs': all_runs,
            'labels': all_labels
        }
    
    def exclude_blocks(self, saccade_count_exclude=False, gaze_distance_exclude=False, distance=float, min_saccades=int, measure=str):
        """
        Excludes blocks based on eye_tracking criteria.

        Parameters:
            saccade_count_exclude (bool): Whether to exclude based on saccade count.
            gaze_distance_exclude (bool): Whether to exclude based on gaze distance.
            distance (float): Distance threshold for exclusion.
            min_saccades (int): Minimum number of saccades for exclusion.
            measure (str): Measure for exclusion.

        Returns:
            ndarray: Exclusion mask.
        """
        # Get sorted file paths for both gaze and data files
        data_file_pattern = os.path.join(self.eye_dir, '*data*C1*.csv')
        gaze_file_pattern = os.path.join(self.eye_dir, '*gaze*C1*.csv')
        data_file_paths = sorted(glob.glob(data_file_pattern), key=lambda x: int(os.path.basename(x).split('_R')[1].split('.')[0]))
        gaze_file_paths = sorted(glob.glob(gaze_file_pattern), key=lambda x: int(os.path.basename(x).split('_R')[1].split('.')[0]))
        
        exclude_saccade = []
        exclude_distance = []
        for f in range(len(data_file_paths)):
            # Get both gaze and data file for each run
            data_file_path = data_file_paths[f]
            gaze_file_path = gaze_file_paths[f]
            data = pd.read_csv(data_file_path)
            gaze = pd.read_csv(gaze_file_path)
            gaze.replace(9999, np.nan, inplace=True)
            gaze /= 88.7

            # Identify blocks that have fewer than MIN_SACCADE number of saccades 
            saccade_count_run = []
            for block_num in range(1, 21):
                # Count the number of saccades for each block
                count = data['event'].str.contains(f"Block {block_num} .*caught", na=False).sum()
                saccade_count_run.append(count)     
            exclude_saccade_run = [sac > min_saccades for sac in saccade_count_run]
            # exclude_saccade_run = exclude_saccade_run[:-4] # activate it saccades are unbalanced
            exclude_saccade = np.concatenate((exclude_saccade, exclude_saccade_run, exclude_saccade_run))
            
            # Creating block indeces for the gaze data and add it to the dataframe
            block_indices = []
            for i in range(len(saccade_count_run)):
                block_indices.extend([i+1] * saccade_count_run[i])
            gaze["Block"] = block_indices

            # Get minimum gaze distance at stimulus disappearance for each block and create exclude file
            min_distance = gaze.groupby('Block')[measure].min()
            exclude_distance_run = min_distance > distance
            # exclude_distance_run = exclude_distance_run[:-4] # activate it saccades are unbalanced
            exclude_distance = np.concatenate((exclude_distance, exclude_distance_run, exclude_distance_run))
        
        exclude_saccade = exclude_saccade.astype(bool)
        exclude_distance = exclude_distance.astype(bool)
        
        if saccade_count_exclude and gaze_distance_exclude:
            exclude = np.logical_and(exclude_saccade, exclude_distance)
        elif saccade_count_exclude:
            exclude = exclude_saccade
        elif gaze_distance_exclude:
            exclude = exclude_distance
            
        return exclude
        
    def create_ret_mask(self, degrees=list):
        """
        Creates the retinotopy mask for the current subject.

        Parameters:
            degrees (list): List of degrees.

        Returns:
            Nifti1Image: Retinotopy mask image.
        """
        ret_data = {}
        for d in ALL_DEGREES:
            ret_image = nib.load(f"{self.ret_dir}thresh_zstat{d}.nii.gz")
            ret_data[d] = ret_image.get_fdata()
            
        # Create masks for each visual degree according to winner take all system     
        stacked_array = np.stack([ret_data[d] for d in ALL_DEGREES], axis=-1)
        max_values = np.max(stacked_array, axis=-1)
        max_mask = (stacked_array == max_values[..., np.newaxis])
        result_arrays = np.zeros_like(stacked_array)
        result_arrays[max_mask] = 1
        zero_mask = (max_values == 0)
        for i in range(len(ALL_DEGREES)):
            result_arrays[..., i][zero_mask] = 0
        ret_masks_data = {ALL_DEGREES[i]: result_arrays[..., i] for i in range(len(ALL_DEGREES))}
        
        # Combine all the masks into the mask of interest
        self.ret_mask_data = np.zeros_like(next(iter(ret_masks_data.values())), dtype=int)
        for deg in degrees:
            self.ret_mask_data += ret_masks_data[deg].astype(int)
        
        # Save retinal mask as nifti file
        ret_mask_image = nib.Nifti1Image(self.ret_mask_data, ret_image.affine, ret_image.header)
        ret_mask_image.to_filename(f"{self.out_dir}/masks/ret_mask.nii.gz")
        return ret_mask_image
    
    def create_control_mask(self):
        """
        Creates the control mask for the current subject.

        Returns:
            Nifti1Image: Control mask image.
        """
        control_image = nib.load(f"{self.out_dir}/masks/frontalpole_mask.nii.gz")
        resampled_control_image = resample_to_img(control_image, self.images_raw[self.runs[0]], interpolation='nearest')
        self.control_mask_data = resampled_control_image.get_fdata()
        
        # Save nifti image of mask
        control_mask_image = nib.Nifti1Image(self.control_mask_data, control_image.affine, control_image.header)
        nib.save(control_mask_image, f"{self.out_dir}/masks/control_mask.nii.gz")
        return control_mask_image
    
    def create_eccen_mask(self, degrees=list):
        """
        Creates the eccentricity mask for the current subject.

        Parameters:
            degrees (list): List of degrees.

        Returns:
            Nifti1Image: Eccentricity mask image.
        """
        eccen_image = nib.load(f"{self.out_dir}/masks/vol_eccen.nii.gz")
        resampled_eccen_image = resample_to_img(eccen_image, self.images_raw[self.runs[0]], interpolation='nearest')
        eccen_data = np.squeeze(resampled_eccen_image.get_fdata(), axis=-1)
        
        self.eccen_mask_data = np.where(np.isin(eccen_data, degrees), eccen_data, 0)
        self.eccen_mask_data = np.where(np.isin(self.eccen_mask_data, [1, 0]), self.eccen_mask_data, 1)
        
        # Save nifti image of mask
        eccen_mask_image = nib.Nifti1Image(self.eccen_mask_data, eccen_image.affine, eccen_image.header)
        nib.save(eccen_mask_image, f"{self.out_dir}/masks/eccen_mask.nii.gz")
        return eccen_mask_image

    def create_anat_mask(self, vareas=list):
        """
        Creates the anatomy mask for the current subject.

        Parameters:
            vareas (list): List of visual areas.

        Returns:
            Nifti1Image: Anatomy mask image.
        """
        vareas_image = nib.load(f"{self.out_dir}/masks/vol_vareas.nii.gz")
        resampled_vareas_image = resample_to_img(vareas_image, self.images_raw[self.runs[0]], interpolation='nearest')
        vareas_data = np.squeeze(resampled_vareas_image.get_fdata(), axis=-1)

        # Convert all the areas that are not of interest to 0 the ones of interest to 1
        self.anat_mask_data = np.where(np.isin(vareas_data, vareas), vareas_data, 0)
        self.anat_mask_data = np.where(np.isin(self.anat_mask_data, [1, 0]), self.anat_mask_data, 1)
        
        # Save nifti image of mask
        anat_mask_image = nib.Nifti1Image(self.anat_mask_data, vareas_image.affine, vareas_image.header)
        nib.save(anat_mask_image, f"{self.out_dir}/masks/anat_mask.nii.gz")
        return anat_mask_image
    
    def create_object_mask(self):
        """
        Creates the object mask for the current subject.

        Returns:
            Nifti1Image: Object mask image.
        """
        object_image = nib.load(f"{self.out_dir}/obj_loc.feat/thresh_zstat3.nii.gz")
        self.object_mask_data = object_image.get_fdata()
        self.object_mask_data = (self.object_mask_data > 3).astype(int)
        
        # Limit LOC mask to later visual areas
        anat_loc_mask_image = self.create_anat_mask(vareas=[7,8])
        anat_loc_mask_data = anat_loc_mask_image.get_fdata()
        self.object_mask_data = np.logical_and(self.object_mask_data, anat_loc_mask_data).astype(np.float32)
        
        object_mask_image = nib.Nifti1Image(self.object_mask_data, object_image.affine, object_image.header)
        object_mask_image.to_filename(f"{self.out_dir}/masks/object_mask.nii.gz")
        return object_mask_image

