from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="gaussian_splat_mesh",
    ext_modules=[
        CUDAExtension(
            "gaussian_splat_mesh",
            [
                "gaussian_splat_mesh_kernel.cu",
            ],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
