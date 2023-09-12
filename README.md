# Data Augmentation with Stable Diffusion for Object Detection using YOLOv8

This is a data generation framework that uses stable diffusion with ControlNet. Models can be trained using a mix of real and generated data. They can also be logged and evaluated.

<img src="docs/images/general_pipeline.png" />

Make sure to install the requirements :

```
pip install -r requirements
```

We recommend using a virtual environment ;)

## Datasets

We experimented with PEOPLE from the COCO datasets :

As always `run` will take care of it.
```
./run.sh coco
```
Data will be downloaded and put in the respective files for images, labels and captions.

## List of some tested SD models and Compatible ControlNet Models :

- SD MODEL
    - CONTROL MODEL

### Canny

- runwayml/stable-diffusion-v1-5 :
    - lllyasviel/sd-controlnet-canny


### OpenPose

- runwayml/stable-diffusion-v1-5
    - lllyasviel/sd-controlnet-openpose
    - frankjoshua/control_v11p_sd15_openpose

- stabilityai/stable-diffusion-2-1
    - thibaud/controlnet-sd21-openposev2-diffusers
    - thibaud/controlnet-sd21-openpose-diffusers


## Generate test images

To generate some images for the moment you can use
```bash
./run.sh gen
```
See the `conf/config.yaml` file for more details, you can configurate your run
from the config file or direclty like
```bash
./run.sh gen model.cn_use=openpose prompt.base="Trump" prompt.modifier="dancing" data_path.generated=mysupertest
```
You will find your images in `bank/data/mysupertest_openpose` along with the base image and the feature extracted.

<p float="left">
    <img width="350" src="docs/images/b_1.png"/>
    <img width="350" src="docs/images/f_1.png"/>
</p>
<p float="left">
    <img width="350" src="docs/images/1_1.png"/>
    <img width="350" src="docs/images/2_1.png"/>
</p>
<p float="left">
    <img width="350" src="docs/images/3_1.png"/>
    <img width="350" src="docs/images/4_1.png"/>
</p>


## Multi run

Here's an example of a multi-run with 3 different generators :

```
./run.sh gen model.cn_use=frankjoshua_openpose,fusing_openpose,lllyasviel_openpose
```

List of available models can be found in `conf/config.yaml`. We have 3 available extractors at the moment (OpenPose, Canny, MediaPipeFace), If you add another control net model, make sure you add one of the following strings to its name to set the extractor to use :

- openpose
- canny
- mediapipe_face


## Test the quality of images with IQA measures

One way of testing the quality of the generated images is to use computational and statistical
methods. One good library for it is [IQA-PyTroch](https://github.com/chaofengc/IQA-PyTorch), you
can go read its [paper](https://arxiv.org/pdf/2208.14818.pdf).


Because images are generated there is no reference image to compare to. We will be using with the
no-reference metrics Note that methods using here are agnostic to the content of the image, no
subjective or conceptual score is given. Measures generated here only give an idea of how
*good looking* the images are.

Methods used:
  - [brisque](https://www.sciencedirect.com/science/article/abs/pii/S0730725X17301340)
  - [cliipiqa](https://arxiv.org/pdf/2207.12396.pdf)
  - [dbccn](https://arxiv.org/pdf/1907.02665v1.pdf)
  - [niqe](https://live.ece.utexas.edu/research/quality/nrqa.html)


You can use these measures in the same way the generation is done:
```bash
./run iqa
```
It follows the same configuration that the generation part, with the same file `conf/config.yaml`.

**Note**: `iqa` is going to search for a directory following the same naming convention that `gen.py`,
that is, the directory has the name `<name chosen by the user>_<cn model used to generate>`.
This are of course in the `config.yaml` file and can be changed statically or dynamically.

<img src="docs/images/iqa_measure.png" />



## Create Dataset and Train :

```
python create_dataset.py --real_images_dir ./bank/real/ --syn_images_dir ./bank/real/ --txt_dir ./bank/  --n_file 0 --per_syn_data 0

python train.py
```