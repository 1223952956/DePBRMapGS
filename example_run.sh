# !/bin/bash

rootpath=/mimlab/ren/project/demapgs/DeMapGS
scene=buddha

# train
python3 main.py -o=$scene -e=$scene-test -r=$rootpath

# Refine texture
python3 refine_texture.py -o=$scene -e=$scene-test -r=$rootpath

# render a frame with GS model
python3 main.py -o=$scene -e=$scene-test --vis --app="vis" -r=$rootpath

# measure mesh distance
python3 evaluation/mesh_distance.py -o=$scene -e=$scene-test -r=$rootpath

# prepare data for blender
python3 scripts/attribute.py -o=$scene -e=$scene-test

# render the result mesh (shading) with OpenGL
python3 scripts/geometry.py -o=$scene -e=$scene-test -r=$rootpath

# render the result mesh (texture) with OpenGL
python3 scripts/texture.py -o=$scene -e=$scene-test -r=$rootpath

