## <h1 style="text-align: center;">Image Classification Project</h1>

This is my implementation for educational purpose, of the Image Classification project, taken from the *"Computer Vision Projects with PyTorch"* book.

The project is about classifying [X-ray](https://www.kaggle.com/datasets/tolgadincer/labeled-chest-xray-images) images, with the help of Computer Vision modeling techniques, as either having pneumonia, or being normal.


It also includes my personal notes that helped me understand some concepts.

>*Project layout (so far):*

<div style="margin-left: auto;
            margin-right: auto;
            width: 30%">

| Chapter |   Title        |
|--------:|----------------|
|        1|[Introduction](https://github.com/DrT3a/Image_Classification_Pytorch/blob/main/1.Intro.ipynb)    |
|        2|[Basic Model](https://github.com/DrT3a/Image_Classification_Pytorch/blob/main/2.First%20Basic%20Model.ipynb)     |
|        3|[Second Variation](https://github.com/DrT3a/Image_Classification_Pytorch/blob/main/3.Second%20Variation.ipynb)|
</div>

>**Chapter 1** is an introduction describing the base concepts and showcasing the workflow.

>**Chapter 2** buids the basic model. During this proccess, I have added detailed notes on every concept encountered as well as detailed description on the code functionality. Finally the results are displayed and commented on along with suggestions on improving them.

>**Chapter 3** is the second models variation. Methods like ColorJitter and random image rotation and flipping are implemented, the tests are rerun and results compared to the basic model.

*To do: There are two more variations to be implemented.*


<h1 style="text-align: center;">Technical details</h1>

The project was run in an [Anaconda](https://www.anaconda.com/) virtual environment.
<div style="margin-left: auto;
            margin-right: auto;
            width: 30%">

| Name    | Version        |
|--------:|----------------|
|[Python](https://www.python.org/)   |3.10.13         |
|[PyTorch](https://pytorch.org/)  |2.1.0           |
|[OpenCv](https://opencv.org/)   |4.6.0           |
|[Cuda*](https://developer.nvidia.com/cuda-downloads)    |12.2            |
</div>  
(*Cuda is recommended but not required, if an Nvidia card is not available)


Below is a detailed printout of the `conda list` used in this project:

```python
# Name                    Version                   Build  Channel
blas                      1.0                         mkl
brotli                    1.0.9                ha925a31_2
brotlipy                  0.7.0           py310h2bbff1b_1002
bzip2                     1.0.8                he774522_0
ca-certificates           2023.7.22            h56e8100_0    conda-forge
certifi                   2023.7.22          pyhd8ed1ab_0    conda-forge
cffi                      1.15.1          py310h2bbff1b_3
charset-normalizer        2.0.4              pyhd3eb1b0_0
colorama                  0.4.6              pyhd8ed1ab_0    conda-forge
cryptography              41.0.3          py310h89fc84f_0
cuda-cccl                 12.2.140                      0    nvidia
cuda-cudart               11.8.89                       0    nvidia
cuda-cudart-dev           11.8.89                       0    nvidia
cuda-cupti                11.8.87                       0    nvidia
cuda-libraries            11.8.0                        0    nvidia
cuda-libraries-dev        11.8.0                        0    nvidia
cuda-nvrtc                11.8.89                       0    nvidia
cuda-nvrtc-dev            11.8.89                       0    nvidia
cuda-nvtx                 11.8.86                       0    nvidia
cuda-profiler-api         12.2.140                      0    nvidia
cuda-runtime              11.8.0                        0    nvidia
cycler                    0.12.1             pyhd8ed1ab_0    conda-forge
eigen                     3.4.0                h2d74725_0    conda-forge
ffmpeg                    4.2.3                ha925a31_0    conda-forge
filelock                  3.9.0           py310haa95532_0
fonttools                 4.25.0             pyhd3eb1b0_0
freetype                  2.12.1               ha860e81_0
giflib                    5.2.1                h8cc25b3_3
glib                      2.69.1               h5dc1a3c_2
gst-plugins-base          1.18.5               h9e645db_0
gstreamer                 1.18.5               hd78058f_0
hdf5                      1.12.1               h51c971a_3
icc_rt                    2022.1.0             h6049295_2
icu                       58.2                 ha925a31_3
idna                      3.4             py310haa95532_0
intel-openmp              2023.1.0         h59b6b97_46319
jinja2                    3.1.2           py310haa95532_0
jpeg                      9e                   h2bbff1b_1
kiwisolver                1.4.4           py310hd77b12b_0
lerc                      3.0                  hd77b12b_0
libclang                  12.0.0          default_h627e005_2
libcublas                 11.11.3.6                     0    nvidia
libcublas-dev             11.11.3.6                     0    nvidia
libcufft                  10.9.0.58                     0    nvidia
libcufft-dev              10.9.0.58                     0    nvidia
libcurand                 10.3.3.141                    0    nvidia
libcurand-dev             10.3.3.141                    0    nvidia
libcusolver               11.4.1.48                     0    nvidia
libcusolver-dev           11.4.1.48                     0    nvidia
libcusparse               11.7.5.86                     0    nvidia
libcusparse-dev           11.7.5.86                     0    nvidia
libdeflate                1.17                 h2bbff1b_1
libffi                    3.4.4                hd77b12b_0
libiconv                  1.17                 h8ffe710_0    conda-forge
libjpeg-turbo             2.0.0                h196d8e1_0
libnpp                    11.8.0.86                     0    nvidia
libnpp-dev                11.8.0.86                     0    nvidia
libnvjpeg                 11.9.0.86                     0    nvidia
libnvjpeg-dev             11.9.0.86                     0    nvidia
libogg                    1.3.4                h8ffe710_1    conda-forge
libpng                    1.6.39               h8cc25b3_0
libprotobuf               3.20.3               h23ce68f_0
libtiff                   4.5.1                hd77b12b_0
libuv                     1.44.2               h2bbff1b_0
libvorbis                 1.3.7                h0e60522_0    conda-forge
libwebp                   1.3.2                hbc33d0d_0
libwebp-base              1.3.2                h2bbff1b_0
libxml2                   2.10.4               h0ad7f3c_1
libxslt                   1.1.37               h2bbff1b_1
lz4-c                     1.9.4                h2bbff1b_0
markupsafe                2.1.1           py310h2bbff1b_0
matplotlib                3.5.3           py310h5588dad_2    conda-forge
matplotlib-base           3.5.3           py310h7329aa0_2    conda-forge
mkl                       2023.1.0         h6b88ed4_46357
mkl-service               2.4.0           py310h2bbff1b_1
mkl_fft                   1.3.8           py310h2bbff1b_0
mkl_random                1.2.4           py310h59b6b97_0
mpmath                    1.3.0           py310haa95532_0
munkres                   1.1.4              pyh9f0ad1d_0    conda-forge
networkx                  3.1             py310haa95532_0
numpy                     1.26.0          py310h055cbcc_0
numpy-base                1.26.0          py310h65a83cf_0
opencv                    4.6.0           py310ha36de5b_5
openssl                   3.0.11               h2bbff1b_2
packaging                 23.2               pyhd8ed1ab_0    conda-forge
pcre                      8.45                 h0e60522_0    conda-forge
pillow                    9.4.0           py310hd77b12b_1
pip                       23.2.1          py310haa95532_0
ply                       3.11                       py_1    conda-forge
pycparser                 2.21               pyhd3eb1b0_0
pyopenssl                 23.2.0          py310haa95532_0
pyparsing                 3.1.1              pyhd8ed1ab_0    conda-forge
pyqt                      5.15.7          py310hd77b12b_0
pyqt5-sip                 12.11.0         py310hd77b12b_0
pysocks                   1.7.1           py310haa95532_0
python                    3.10.13              he1021f5_0
python-dateutil           2.8.2              pyhd8ed1ab_0    conda-forge
python_abi                3.10                    2_cp310    conda-forge
pytorch                   2.1.0           py3.10_cuda11.8_cudnn8_0    pytorch
pytorch-cuda              11.8                 h24eeafa_5    pytorch
pytorch-model-summary     0.1.1                      py_0    conda-forge
pytorch-mutex             1.0                        cuda    pytorch
pyyaml                    6.0             py310h2bbff1b_1
qt-main                   5.15.2               he8e5bd7_7
qt-webengine              5.15.9               h5bd16bc_7
qtwebkit                  5.212                h2bbfb41_5
requests                  2.31.0          py310haa95532_0
setuptools                68.0.0          py310haa95532_0
sip                       6.6.2           py310hd77b12b_0
six                       1.16.0             pyh6c4a22f_0    conda-forge
sqlite                    3.41.2               h2bbff1b_0
sympy                     1.11.1          py310haa95532_0
tbb                       2021.8.0             h59b6b97_0
tk                        8.6.12               h2bbff1b_0
toml                      0.10.2             pyhd8ed1ab_0    conda-forge
torchaudio                2.1.0                    pypi_0    pypi
torchinfo                 1.8.0              pyhd8ed1ab_0    conda-forge
torchvision               0.16.0                   pypi_0    pypi
tornado                   6.2             py310he2412df_0    conda-forge
tqdm                      4.66.1             pyhd8ed1ab_0    conda-forge
typing_extensions         4.7.1           py310haa95532_0
tzdata                    2023c                h04d1e81_0
urllib3                   1.26.16         py310haa95532_0
vc                        14.2                 h21ff451_1
vs2015_runtime            14.27.29016          h5e58377_2
wheel                     0.41.2          py310haa95532_0
win_inet_pton             1.1.0           py310haa95532_0
xz                        5.4.2                h8cc25b3_0
yaml                      0.2.5                he774522_0
zlib                      1.2.13               h8cc25b3_0
zstd                      1.5.5                hd43e919_0
```