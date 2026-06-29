import torchio as tio, numpy as np
import torch, pandas as pd, nibabel as nib, tempfile
import os
from typing import Dict, List, Optional, Sequence, Tuple, Union
import fnmatch
from scipy.ndimage import binary_dilation, generate_binary_structure
from torchio import Subject


def fill_mask_with_nearest_value(volume: np.ndarray) -> np.ndarray:
    if not isinstance(volume, np.ndarray):
        raise TypeError("Input must be a numpy.ndarray")
    if volume.ndim != 4 or volume.shape[0] != 1:
        raise ValueError("Expected shape (1, D, H, W)")

    out = volume.copy()
    labels = volume[0]

    background = labels == 0
    if not np.any(background):
        return out

    # Connected components of background
    labeled_bg, _ = ndi_label(background)

    # Background connected to the border
    border_labels = np.unique(np.concatenate([
        labeled_bg[0].ravel(),
        labeled_bg[-1].ravel(),
        labeled_bg[:, 0].ravel(),
        labeled_bg[:, -1].ravel(),
        labeled_bg[:, :, 0].ravel(),
        labeled_bg[:, :, -1].ravel(),
    ]))
    border_labels = border_labels[border_labels != 0]

    # Holes = background NOT connected to border
    holes = background & (~np.isin(labeled_bg, border_labels))
    if not np.any(holes):
        return out

    # distance to nearest NON-ZERO
    indices = distance_transform_edt(
        labels == 0,
        return_indices=True,
        return_distances=False
    )

    z, y, x = np.where(holes)
    out[0, z, y, x] = labels[
        indices[0][z, y, x],
        indices[1][z, y, x],
        indices[2][z, y, x],
    ]

    return out

class OneHotTransform(tio.OneHot):
    """
    Extend TorchIO's OneHot to allow inversion and hole-filling.
    """
    def __init__(
        self,
        num_classes: int = -1,
        invert_transform: bool = False,
        background_segmented: bool = True,
        exclude: Optional[Sequence[str]] = None,
        fill_holes: bool = False,
        **kwargs
    ):
        """
        Initialize the transform.

        Parameters
        ----------
        num_classes : int
            Number of classes (-1 to infer).
        invert_transform : bool
            If True, invert one-hot back to labels.
        background_segmented : bool
            If False, drop the background channel after one-hot.
        exclude : Optional[Sequence[str]]
            Image names to exclude.
        fill_holes : bool
            If True, fill holes after inversion.
        **kwargs
            Passed to `tio.OneHot`.
        """
        super().__init__(num_classes, exclude=exclude, **kwargs)
        self.background_segmented = background_segmented
        self.invert_transform = invert_transform
        self.fill_holes = fill_holes

    def apply_transform(self, subject: tio.Subject) -> tio.Subject:
        """
        Apply on each image in the subject.

        - If `invert_transform` is False: one-hot encode; optionally drop background.
        - If True: argmax back to labels; optionally fill holes.

        Returns
        -------
        tio.Subject
            The (modified) subject.
        """
        for image in self.get_images(subject):
            if not self.invert_transform:
                # Forward one-hot
                self.one_hot(image)
                if not self.background_segmented:
                    image.set_data(image.data[1:])
            else:
                # Invert back to labels
                self.argmax(image)
                if self.fill_holes:
                    filled = fill_mask_with_nearest_value(image.data.numpy())
                    image.set_data(torch.from_numpy(filled))
        return subject
class RandomMorphologyTransform(tio.LabelTransform):
    """
    TorchIO transform that applies random dilations/erosions

    Parameters
    ----------
    """

    def __init__(
        self,
        p: float = 1,
        label_to_dilate: str = "",
        label_within: str = "",
        nb_iter_min: int = 1,
        nb_iter_max: int = 2,
        transform_suffix: str = "dil",
        label_within_delete: str = None,
        verbose_dev: bool = True,
        **kwargs
    ):
        super().__init__(p=p, **kwargs)
        #configure_logging(dev_mode=verbose_dev, module_name=__name__)


        self.nb_iter_min = nb_iter_min
        self.nb_iter_max = nb_iter_max
        self.transform_suffix = transform_suffix
        self.label_within_delete = label_within_delete
        self.label_to_dilate=label_to_dilate
        self.label_within = label_within


    def apply_transform(self, subject: Subject) -> Subject:
        #logger.info("%s[RandomMorphologyTransform]%s Applying morphology transform %s%s%s", BLUE, RESET, BLUE, self.transform_suffix, RESET)
        if len(self.get_images(subject)) > 1:
            raise NotImplementedError(f"Morphology transform with more than one labelmap image.")

        if "suffix" in subject:
            transform_suffix = subject["suffix"]
        else:
            transform_suffix = ""
        for label_map in self.get_images(subject):
            data = label_map.data.numpy()

            nb_iter = torch.randint(self.nb_iter_min, self.nb_iter_max, (1,)).item()
            if self.label_within_delete is not None:
                data = self._wm_carve_gm(data,self.label_within_delete, nb_iter)
            else:
                data = self._morphology(
                    data,
                    target_labels=self.label_to_dilate,
                    iterations=nb_iter,
                    within_labels=self.label_within,
                )

            transform_suffix += self.transform_suffix +  f"_It{nb_iter}"

            label_map.set_data(torch.from_numpy(data))
            subject["label"] = label_map
            subject["suffix"] = transform_suffix
        return subject

    @staticmethod
    def _morphology(
        data,
        target_labels,
        iterations=1,
        within_labels=None,
    ):
        for label in target_labels:
            # Prepare boolean mask for dilation
            mask_total = np.isin(data, label)
            new_mask = binary_dilation(
                mask_total, iterations=iterations
            )

            if within_labels is not None:
                within_mask = np.isin(data, within_labels)
                new_mask &= within_mask

            # For label in target_labels:
            data[(new_mask) & (~mask_total)] = label
        return data

    def _wm_carve_gm(self, data, replace_label_by, iters):
        if len(self.label_within) != 1:
            raise NotImplementedError(f"Currently only one label_within is supported for wm_carve_gm. got {self.label_within}")
        if len(self.label_to_dilate) != 1:
            raise NotImplementedError(f"Currently only one label_to_dilate is supported for wm_carve_gm. got {self.label_to_dilate}")

        gm_mask = np.isin(data, self.label_within)
        wm_mask = np.isin(data, self.label_to_dilate)

        dilated_wm = binary_dilation(
            wm_mask[0], generate_binary_structure(3, 2), iters
        ).astype(int)
        dilated_wm = np.expand_dims(dilated_wm, 0)
        dilated_wm[wm_mask] = 0

        # Temporarily convert GM voxels to CSF to avoid overlap
        data[gm_mask] = replace_label_by

        # Create GM boundary
        gm_boundary = binary_dilation(gm_mask, iterations=4).astype(int)
        # if not we add GM near ventricul branstem ect ...
        new_gm = dilated_wm * gm_boundary
        data[new_gm > 0] = self.label_within[0] # Possible because len(self.label_within) == 1

        return data

def isnotNaN(num):
    return num == num
def get_siam_data_dir():
    siam_data = os.environ.get('SIAM_DATA')
    if not siam_data:
        siam_data = '/network/iss/opendata/data/template/siam_label/v4_release'
        #raise RuntimeError('Environment variable SIAM_DATA is not set')
    return siam_data

def get_csv_remaping(modelname):

    siam_data = get_siam_data_dir()

    if 'vasc' in modelname:
        labels_csv = os.path.join(siam_data, 'vascular', 'vascular_full_label_v4.csv')
    if 'mida' in modelname:
        labels_csv = os.path.join(siam_data, 'mida', 'mida_labels_v4_RU_SN.csv')
    if 'skull' in modelname:
        labels_csv = os.path.join(siam_data, 'skull', 'brain_and_skull_Ultra_V4.csv')
    if 'allen' in modelname:
        labels_csv = os.path.join(siam_data, 'AllenB', 'label_AllenB.csv')
    if 'big' in modelname:
        labels_csv = os.path.join(siam_data, 'bigB', 'label_BigBrain.csv')
    return labels_csv

def get_target_remaping(modelname, add_WM_ANO=False, add_ANO=False):
    label_csv = get_csv_remaping(modelname)
    df = pd.read_csv(label_csv,comment='#')
    synth_col = "synth" #you one parameter identical in all DS

    #TODO make it specific in the config not in the code
    if ('mida' in modelname)|('skull' in modelname)|('big' in modelname): # Case MIDA, Skull and BigBrain datasets
        synth_label_to_target = {int(ll[synth_col]): int(ll[synth_col]) for _, ll in df.iterrows()if not pd.isna(ll[synth_col])}
        synth_label_to_target, preprocessing_target_map_name, preprocessing_target_map_to_synth = None,None,None
    elif ('vasc' in modelname)|('allen' in modelname):  # Case Vascular and Allen Brain datasets
        #synth_label_to_target = {int(ll[synth_col]): int(ll['synth_target']) for _, ll in df.iterrows() if not pd.isna(ll['synth_target'])}
        # not used here, we keep all label
        preprocessing_target_map_name = {ll['NameTargetRegion']: int(ll['synth_targetRegion']) for _, ll in df.iterrows() if not pd.isna(ll['synth_targetRegion'])} # if not nan in synth_targetRegion
        preprocessing_target_map_name = dict(sorted(preprocessing_target_map_name.items(), key=lambda item: item[1]))
        preprocessing_target_map_to_synth = {int(ll[synth_col]): int(ll['synth_targetRegion']) for _, ll in df.iterrows() if not pd.isna(ll['synth_targetRegion'])}
        #preprocessing_target_map = (preprocessing_target_map_name, preprocessing_target_map_to_synth)


    dic_map_target = {ll['synth']:ll['targetRegV4'] for ii,ll in df.iterrows() if isnotNaN(ll['synth'] ) }
    dic_map_tissue = {ll['synth']:ll['synth_tissu'] for ii,ll in df.iterrows() if isnotNaN(ll['synth'] ) }
    label_dic = {ll['NametargetRegV4']:ll['targetRegV4'] for ii,ll in df.iterrows() if isnotNaN(ll['synth'] )}
    label_dic = {k: v for k, v in sorted(label_dic.items(), key=lambda item: item[1])}
    label_dic_all =  {ll['Name']:ll['synth'] for ii,ll in df.iterrows()  if isnotNaN(ll['synth'] )}
    if add_WM_ANO:
        max_synth,max_tar = max(dic_map_target.keys()), max(label_dic.values())
        for k in range(5):
            dic_map_target[max_synth+1+k] = max_tar + 1
        label_dic["AnoWM"] = max_tar + 1
    elif add_ANO:
        max_synth,max_tar = max(dic_map_target.keys()), max(label_dic.values())
        for k in range(5):
            dic_map_target[max_synth+1+k] = max_tar
        # already in the csv ....   label_dic["Anomalies"] = max_tar

    else:
        if 'lesion' in label_dic.keys():
            label_dic.pop('lesion',None)
        if 'Lesion' in label_dic.keys():
            label_dic.pop('Lesion',None)

    #print(dic_map_target, label_dic)
    #print(label_dic)
    return dic_map_tissue, dic_map_target, label_dic, label_dic_all, preprocessing_target_map_name, preprocessing_target_map_to_synth

def get_siam_data(modelname):
    siam_data = get_siam_data_dir()
    dir_subregion,subregions_lab_list = None,[]

    if 'vasc1' in modelname:
        flab = os.path.join(siam_data, 'vascular', 'Svas_03', 'r025s05_SynthVas_03_V4.nii.gz')
    if 'vasc2' in modelname:
        flab = os.path.join(siam_data, 'vascular', 'Svas_03', 'r025s05_SynthVas_03_V4_Dill.nii.gz')
    if 'vasc3' in modelname:
        flab = os.path.join(siam_data, 'vascular', 'Svas_04', 'r025s05_SynthVas_03_V4.nii.gz')
    if 'vasc4' in modelname:
        flab = os.path.join(siam_data, 'vascular', 'Svas_04', 'r025s05_Synth_Vas_04_v4_VentDill.nii.gz')

    if 'vasc' in modelname:
        dir_subregion = os.path.join(siam_data, 'vascular', 'Svas_03', 'subregions_labels')
        subregions_lab_list = ['WMregion','cerGM*cereb','WMCereb*_cereb','hip','Thal*[Yy]eb','Caud*[Yy]eb','Put*[Yy]eb','Pal*[Yy]eb','STN*[Yy]eb','gm']

    if 'skull1' in modelname:
        flab = os.path.join(siam_data, 'skull', 'V4_inVesSkCT_r025s05_r025_PVdown_GM__head_U_006_midaV4s05.nii.gz')
    if 'skull2' in modelname:
        flab = os.path.join(siam_data, 'skull', 'V4_inVesSkCT_r025s05_r025_PVdown_GM__head_U_013_midaV4s05.nii.gz')
    if 'skull3' in modelname:
        flab = os.path.join(siam_data, 'skull', 'V4_inVesSkCT_r025s05_r025_PV_down_up_head_U_006_midaV4s05.nii.gz')
    if 'skull4' in modelname:
        flab = os.path.join(siam_data, 'skull', 'V4_inVesSkCT_r025s05_r025_PV_down_up_head_U_010_midaV4s05.nii.gz')
    if 'skull5' in modelname:
        flab = os.path.join(siam_data, 'skull', 'V4_inVesSkCT_r025s05_r025_PV_down_up_head_U_013_midaV4s05.nii.gz')

    if 'mida1' in modelname:
        flab = os.path.join(siam_data, 'mida', 'csf_veine_r025s05_mida_new_std_v4_cor_csf.nii.gz')
    if 'mida2' in modelname:
        flab = os.path.join(siam_data, 'mida', 'csf_veine_r025s05_mida_new_std_v4_cor_csf.nii.gz')

    if 'allen' in modelname:
        flab = os.path.join(siam_data,'AllenB','r025noS_region_Allen.nii.gz')
        subregions_lab_list = ['WM','cerGM','Hippo','Amyg','dGM1','dGM2','dGM3','dGM5','_GM', 'ventricle']
        dir_subregion = os.path.join(siam_data,'AllenB','subregion')

    (dic_map_tissue, dic_map_target, label_dic, label_dic_all,
     preprocessing_target_map_name, preprocessing_target_map_to_synth) = get_target_remaping(modelname)

    return_dict = {"file_lab":flab, "subregion_dir":dir_subregion, "subregion_regex": subregions_lab_list,
                   "dic_map_tissue":dic_map_tissue, "dic_map_target" :dic_map_target,
                   "label_name_tissue": label_dic, "label_name_all": label_dic_all,
                   "initial_label_name":preprocessing_target_map_name,
                   "dic_map_region":preprocessing_target_map_to_synth}
    return return_dict

def rescale_affine_corrected(
    affine: np.ndarray,
    shape: Sequence[int],
    zooms: Sequence[float],
    new_shape: Optional[Sequence[int]] = None,
) -> np.ndarray:
    """
    Compute a new affine for resampled image to preserve centering.
    """
    shape = np.array(shape)
    new_shape = np.array(new_shape)

    # Compute the original voxel spacing from the affine
    old_spacing = nib.affines.voxel_sizes(affine)
    # Compute the new scaling matrix for the affine
    scale = affine[:3, :3] * (zooms / old_spacing)

    # Compute the center of the old image in world coordinates
    center_old = nib.affines.apply_affine(affine, (shape - 1) / 2)
    # Compute the center of the new image in world coordinates (using new scale)
    center_new = scale @ ((new_shape - 1) / 2)
    # Compute the translation needed to keep the image centered
    trans = center_old - center_new

    # Return the new affine matrix with updated scale and translation
    return nib.affines.from_matvec(scale, trans)


def pool_remap(
    image: Union[tio.LabelMap, tio.ScalarImage],
    pooling_size: int = 2,
    ensure_multiple: Optional[int] = None,
    transform_map: Optional[tio.Transform] = None,
    keep_missing_label: bool = True,
    is_label: bool = True,
) -> Union[Tuple[tio.LabelMap, tio.LabelMap], tio.ScalarImage]:
    """
    Downsample (pool) a label map or scalar image, with optional one-hot remapping for labels.

    For label maps, returns both a pooled binary label map and a multi-channel label map.
    For scalar images, returns the pooled image.

    Parameters
    ----------
    image : tio.LabelMap or tio.ScalarImage
        The input image to downsample (label or scalar).

    pooling_size : int, optional
        The size of the pooling kernel, by default=2.

    ensure_multiple : int, optional
        If set, pad the image to ensure its shape is a multiple of this value, by default=None.

    transform_map : tio.Transform, optional
        An optional transform to apply before pooling, by default=None.

    keep_missing_label : bool, optional
        If True, keep all possible label channels up to the max label, by default=True.

    is_label : bool, optional
        If True, treat the input as a label map; otherwise as a scalar image, by default=True.

    Returns
    -------
    tuple of (tio.LabelMap, tio.LabelMap)
        If is_label is True: (binary_labels, multi_channel_labels)

    tio.ScalarImage
        If is_label is False: the pooled scalar image.

    """
    thot_inv = OneHotTransform(invert_transform=True)
    pool = torch.nn.AvgPool3d(kernel_size=pooling_size, ceil_mode=True)
    tpad = tio.EnsureShapeMultiple(ensure_multiple or pooling_size)

    # Apply optional pre-transform
    image = tpad(transform_map(image)) if transform_map else tpad(image)

    # Get the new affine
    original_shape = np.array(image.shape[1:])
    new_shape = original_shape // pooling_size
    new_voxel_size = nib.affines.voxel_sizes(image.affine) * pooling_size
    new_affine = rescale_affine_corrected(
        image.affine, original_shape, new_voxel_size, new_shape
    )

    if is_label:

        labels = image.data.squeeze(0).int()
        label_values = labels.unique()
        nb_channel = (
            int(label_values.max()) + 1 if keep_missing_label else len(label_values)
        )
        binary_map = torch.zeros([nb_channel, *new_shape])

        # For each label/channel, create a binary mask and pool it
        for channel in range(nb_channel):
            mask = labels == (channel if keep_missing_label else label_values[channel])
            binary_map[channel] = pool(mask.float().unsqueeze(0))[0]

        label_image = tio.LabelMap(tensor=binary_map, affine=new_affine)
        binary = thot_inv(label_image)

        return binary, label_image
    else:
        img = image.data.float().unsqueeze(0)
        down = pool(img)[0]
        return tio.ScalarImage(tensor=down, affine=new_affine)


def add_subregions_labels(
        directory_path: str,
        add_subregions_lab: List[str],
        label_image: tio.LabelMap,
        name_map_current: Dict[str, int],
        preprocessing_target_map: Dict[str, int],
        name_col: str = "Name_continuous",
        label_col: str = "label_continuous",
):
    """
    Add subregion labels to a label image using external NIfTI and CSV files.

    For each subregion in `add_subregions_lab`, this function:
      - Loads the corresponding NIfTI and CSV files from `directory_path`.
      - Resolves target and source label IDs using the provided mapping dictionaries.
      - Applies a mask to restrict label assignment to the relevant region.
      - Remaps the subregion labels according to the CSV and inserts them into the output label image.
      - Checks for missing or overlapping labels and raises errors or warnings as appropriate.
      - Ensures that all target regions are covered, either by subregion assignment or by copying from the original label image.

    Parameters
    ----------
    directory_path : str
        Path to the directory containing subregion NIfTI and CSV files.
    add_subregions_lab : list of str
        List of subregion names to add.
    label_image : tio.LabelMap
        The base label image to which subregions will be added.
    name_map_current : dict
        Mapping from region names to target label IDs.
    preprocessing_target_map : dict
        Mapping from target label IDs to source label IDs.
    name_col : str, optional
        Name of the column in the CSV containing region names, by default="Name_continuous".
    label_col : str, optional
        Name of the column in the CSV containing label values, by default="label_continuous".

    Returns
    -------
    tio.LabelMap
        The label image with subregion labels added and checked for consistency.

    Raises
    ------
    RuntimeError
        If the expected NIfTI or CSV files are missing or ambiguous.
    ValueError
        If a mask is empty, if no value is assigned in the mask, or if overlaps are detected.
    """
    out_put_label_image = torch.zeros_like(label_image.data)
    target_not_in_subregion = list(preprocessing_target_map.keys())

    for lab in add_subregions_lab:
        print(f"Processing subregion  {lab}")
        # File matching between parcel nii and parcel csv
        nii_matches = fnmatch.filter(os.listdir(directory_path), f"*{lab}*.nii.gz")
        csv_matches = fnmatch.filter(os.listdir(directory_path), f"*{lab}*.csv")

        if len(nii_matches) != 1 or len(csv_matches) != 1:
            raise RuntimeError(
                f"{lab}: expected 1 nii + 1 csv, got {nii_matches}, {csv_matches}"
            )

        # Load the label map and CSV for the subregion
        nii_path = os.path.join(directory_path, nii_matches[0])
        csv_path = os.path.join(directory_path, csv_matches[0])

        label_to_add_image = tio.LabelMap(nii_path)
        df_add = pd.read_csv(csv_path)

        # Resolve target IDs and source IDs
        target_ids = {name_map_current[name] for name in df_add[name_col]}
        source_ids = {preprocessing_target_map[target_key] for target_key in target_ids}
        for target_key in target_ids:
            target_not_in_subregion.remove(target_key)

        mask_add = torch.isin(label_image.data, torch.tensor(list(source_ids), device=label_image.data.device))

        # Check if the mask is empty
        if not torch.any(mask_add):
            raise ValueError(f"{lab}: mask is empty")

        label_to_add_image.data[~mask_add] = 0  # Set to zero outside the mask

        # Remap the label map to target IDs
        dict_remap = dict(zip(df_add[label_col], target_ids))
        label_map_image = tio.RemapLabels(dict_remap)(label_to_add_image)

        if torch.all(label_map_image.data[mask_add] == 0):
            raise ValueError(f"No assigned value in dilation mask for {lab}")

        # Add the subregion labels to the output label image
        out_put_label_image[mask_add] = label_map_image.data[mask_add]

        # Check for missing target IDs in the output label image
        unique_labels_in_image = torch.unique(out_put_label_image)
        missing_ids = [target_id for target_id in target_ids if target_id not in unique_labels_in_image]

        if missing_ids:
            # Name of the missing ids
            names_missing = [name for name, target_id in name_map_current.items() if target_id in missing_ids]
            print(
                f"WARNING {lab}: missing target IDs {missing_ids} Corresponding names {names_missing}"
            )

    # Add remaining target IDs that are not in subregions, and check for overlaps
    target_id_overlaps = []
    for target_id in target_not_in_subregion:
        # Target ID in the preprocessed subject
        target_id_few = preprocessing_target_map[target_id]
        mask_target = label_image.data == target_id_few

        # Overlap with the added subregions
        overlap = out_put_label_image[mask_target]
        if torch.any(overlap != 0):
            target_id_overlaps.append((target_id, torch.unique(overlap)))
        out_put_label_image[mask_target] = target_id

    # Check for overlaps
    if target_id_overlaps:
        overlap_info = ";\n ".join(
            f"target ID {tid} overlaps with labels {olap.cpu().numpy()}"
            for tid, olap in target_id_overlaps
        )
        raise ValueError(f"Overlaps detected for target IDs not in subregions: {overlap_info}")

    # Check for continuous labels
    if not torch.all(torch.diff(torch.unique(out_put_label_image.data)) == 1):
        print(f"WARNING  Non-continuous labels after adding subregions: {torch.unique(out_put_label_image.data)}")

    label_image.set_data(out_put_label_image)
    return label_image

if __name__ == '__main__':


    for suj in ['vasc1', 'allen','skull3','mida1']: #
        print(suj)
        break

    siamdic = get_siam_data(suj)
    label_img = tio.LabelMap(siamdic['file_lab'])

    if siamdic['initial_label_name'] is not None:
        # Special case vascular, where initial volume has left and right GM with value 1 and 2
        # so we remap label 2 to 1, to have like other only one GM label
        if "GM" not in siamdic['initial_label_name']:
            tmap = tio.RemapLabels({siamdic['initial_label_name']['lGM']: 1, siamdic['initial_label_name']["rGM"]: 1})
            label_img = tmap(label_img)  # Concatenate left and right GM
            siamdic['initial_label_name']["GM"] = 1


    dir_subregion, subregions_lab_list = siamdic['subregion_dir'], siamdic['subregion_regex']
    label_ini, label_dic_all = siamdic['initial_label_name'], siamdic['label_name_all']
    dic_map_tissue, dic_map_target, dic_map_region  = siamdic['dic_map_tissue'], siamdic['dic_map_target'], siamdic['dic_map_region']

    do_morpho=False
    if do_morpho:
        #here we perform first global GM dilation befor to insert subregions
        tdillWM = RandomMorphologyTransform(label_to_dilate=[label_ini['WM']], label_within=[label_ini['GM']],
                                          nb_iter_max=5, nb_iter_min=4, transform_suffix="erodGM",
                                           label_within_delete=[label_ini['CSF']])
        # with the param "label_within_delete" all GM is replace by CSF, and new one is construct from dillate WM
        # without "label_within_delete" argument to have a simple dillation

        label_img_dill = tdillWM(label_img)

        il_all_region = add_subregions_labels(dir_subregion,subregions_lab_list,label_img_dill,
                                              label_dic_all, dic_map_region)
        il_all_region.save(f'si_{suj}_all_region_WM4.nii.gz')
        il_tissue = tio.RemapLabels(dic_map_tissue)(il_all_region)
        il_tissue.save(f'si_{suj}_tissueWM4.nii.gz')

    else:
        if dir_subregion is not None:
            il_all_region = add_subregions_labels(dir_subregion, subregions_lab_list, label_img,
                                                  label_dic_all, dic_map_region)
            il_all_region.save(f'si_{suj}_all_region.nii.gz')
            il_tissue = tio.RemapLabels(dic_map_tissue)(il_all_region)
            il_tissue.save(f'si_{suj}_tissue.nii.gz')
        else :
            label_img.save(f'si_{suj}_all_region.nii.gz')
            il_tissue = tio.RemapLabels(dic_map_tissue)(label_img)
            il_tissue.save(f'si_{suj}_tissue.nii.gz')


    #alternatively you can perform dillation on the full roi targeting specific GM regions
    # the label need to be adapt for each dataset
    tdill = RandomMorphologyTransform(label_to_dilate=[1], label_within=[90,91,92,93,94,95,96,97],
                                      nb_iter_max=4, nb_iter_min=3, transform_suffix="erodGM")
    il_all_region_morphoLocal = tdill(il_all_region)
    #then again you can transform to tissue and saved

    #last step, from 0.25 mm to lower resolution with partial volume estimation
    nb_pool=3 # to achieve 3*0.25 mm resolution
    label_bin, label_4d = pool_remap(il_all_region, pooling_size=nb_pool, ensure_multiple=nb_pool*2)
    label_4d_tissu = tio.RemapLabels(dic_map_tissue)(label_4d)

    suj = tio.Subject(label=label_4d_tissu)
    timg = tio.RandomLabelsToImage(label_key='label')
    for i in range(5):
        sujo = timg (suj)
        sujo.image_from_labels.save(f'r075_sim{i}_vasc.nii.gz')


