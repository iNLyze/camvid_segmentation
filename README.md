# camvid_segmentation

This project was created while studying semantic segmentation as part of the highly recommended course *Deep Learning for coders part 2*
thought by Rachel Thomas and Jeremy Howard. See the details on the [course page](http://course.fast.ai/) and be sure to read the [forums](http://forums.fast.ai)

## Goal of this project
The notebook implements semantic segmentation using a Densenet architecture called [Tiramisu](https://arxiv.org/abs/1611.09326). 
It uses the camvid data set, still images of street scenes extracted from a head mounted camera while driving. The model learns to recognize cars, trees, pedestrians, etc. with pixel-level accuracy. 

## How to run
The jupyter notebook contains all the experiments, but data is not provided. The camvid data should be copied to the respective subdirectories as pointed out in the notebook.
The directory *DL_utils* contains some helper functions created in part by myself and in part provided by the above mentioned course. 
The directory *segm_utils* contains code for creating the model and for building data iterators with real-time augmentation as provided by the course and modified by myself. 

**Note:** The notebook represents a snapshot of the project and may be a little rough around the edges.**
