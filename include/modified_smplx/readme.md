# prepare data for avatarhq
### File Structure
    ```bash
    rootpath
        ├── Actor02/Sequence1
        │   ├── objs/
        │   ├── 1x/
        │   │    ├── calibration.csv
        │   │    ├── rgbs/
        │   │    ├── s_masks/
        │   │    └── normals/
        │   └── smpl_params.npz
        └── ...
    ```
### Download data
1. Download data from [ActorHQ](https://actors-hq.com/#dataset), data should be in `1x/rgbs/Cam{$view}/Cam{$view}_rgb{$pose_idx}.jpg`
2. Download smplx parameters from [Animatable Gaussian](https://github.com/lizhe00/AnimatableGaussians?tab=readme-ov-file), the link is [here](https://drive.google.com/file/d/1DVk3k-eNbVqVCkLhGJhD_e9ILLCwhspR/view).

### Generate template obj
1. prepare [SMPLX](https://smpl-x.is.tue.mpg.de/): download smplx models (extract the .npz and .pkl files) to `models/smplx/`
2. run `generate_obj.py` to generate template file
```bash
python3 generate_obj.py -s=$scene -d=$data_dir -i=$pose_idx
# example
python3 generate_obj.py -s=Actor07 -d=/data/actorshq/ -i=0
```

### Generate mask and normal with sapiens
1. download and install [Sapiens](https://github.com/facebookresearch/sapiens). We used 1b torchscript checkpoint for segmentation (place in `$SAPIENS_ROOT/pretrain/checkpoints/torchscript/seg/checkpoints/sapiens_1b/sapiens_1b_goliath_best_goliath_mIoU_7994_epoch_151_torchscript.pt2`),  2b torchscript checkpoint for normal generation (place in `$SAPIENS_ROOT/pretrain/checkpoints/torchscript/normal/checkpoints/sapiens_2b/sapiens_2b_normal_render_people_epoch_70_torchscript.pt2`).
2. I dont remember whether it matters, but i changed a line in `$SAPIENS_ROOT/lite/demo/vis_depth.py`, from `normals = np.dstack((-grad_x, -grad_y, z))` to `normals = np.dstack((-grad_x, -grad_y, -z))`
3. first use the segmentation script in `$SAPIENS_ROOT/lite/scripts/demo/torchscript/seg.sh` for segmentation, save the files in `$DATA_PATH/$SCENE/1x/s_masks/Cam{$view}_rgb{$pose_idx}.npy`
    
    We give an example of `seg.sh` in [sapiens_example/seg.sh](sapiens_example/seg.sh) to substitute `$SAPIENS_ROOT/lite/scripts/demo/torchscript/seg.sh`. First create folder `$SAPIENS_ROOT/test/input` and `$SAPIENS_ROOT/test/output` under Sapiens root directory. Then put the [sapiens_example/lst.txt](sapiens_example/lst.txt) under `$SAPIENS_ROOT/test/input`. Output should be in `$SAPIENS_ROOT/test/output/actor07/sapiens_1b`. Copy them to target folder.
4. then use the normal generation script in `$SAPIENS_ROOT/lite/scripts/demo/torchscript/normal.sh`, save the files in `$DATA_PATH/$SCENE/1x/normals/Cam{$view}_rgb{$pose_idx}.npy`

    An example of `normal.sh` is in [sapiens_example/normal.sh](sapiens_example/normal.sh) to substitute `$SAPIENS_ROOT/lite/scripts/demo/torchscript/normal.sh`. Output shoudl be in `$SAPIENS_ROOT/test/output/actor07/sapiens_2b`. Copy them to target folder.
