The code has only been tested on Ubuntu22.04 with Cuda 11.8, Python 3.10
1. Install [PyTorch](https://pytorch.org/get-started/locally/) and [PyTorch3D](https://github.com/facebookresearch/pytorch3d) (May take ~30 minutes).
    ```bash
    pip install torch torchvision torchaudio  --index-url https://download.pytorch.org/whl/cu118
    pip install git+https://github.com/facebookresearch/pytorch3d@stable --no-build-isolation
    ```
2. Install rasterizer
    ```bash
    pip install --no-build-isolation git+https://github.com/hbb1/diff-surfel-rasterization.git
    pip install --no-build-isolation git+https://github.com/ShuyiZhou495/diff-gaussian-rasterization.git
    ```
3. Install splat projection function in `include/gaussian_splat_mesh`
    ```bash
    cd include/gaussian_splat_mesh
    python3 setup.py install
    ```
4. Install rest packages in [requirements.txt](../requirements.txt)