# SIAM_HR_Templates

This repo collects data and information on head and brain label templates used to build the synthetic training dataset for our favorit segmentation model [SIAM](https://github.com/romainVala/SIAM).

We’ll keep improving these labels and hopefully add new ones over time, may be with your help ...


# Template version 1 : 

 [Download](https://zenodo.org/records/20399109/files/siam_label_tempate_v1.tar.gz?download=1) 

This version brings together four datasets, each with different labeling schemes at 0.25 mm³ resolution.
We also provide mappings to get consistent tissue labels across all datasets.


- **MIDA**: 1 subject -> 2 volumes. 
Based on the original (slightly corrected) template.
We also built a version with larger ventricles by non-linear registration to an older subject to insert its ventricule.

- **skull**: 3 subjects -> 5 volumes. 
For two subjects, we generated an extra volume with a thinner cortical GM (see below).

- **vascular**: 2 subjects -> 4 volumes. For each subject we manually dilated the ventricule

- **AllenB**: 1subject -> 1 volume
not used in the 2026 paper, but still great to have!
It includes unique subcortical labels and will be part of our next release.


If usefull, please cite 
> Valabregue, R., Khemir, I., Bardinet, E., Rousseau, F., Auzias, G. & Dorent R. (2026).
> _SIAM : Head and Brain MRI Segmentation from Few High-Quality Templates via Synthetic Training._
> ArXiv. [https://arxiv.org/abs/2605.02737)


The full generative pipeline used for SIAM will be hosted elsewere as it is an other topic: data (and label) augmentations 

We will post here only minimal example for simple generative based on torchio.

Comming soon minimal example to derived varying cortical gray matter thickness, with an estimation of Partial Volume at desired resolution

