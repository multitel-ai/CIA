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

```
# Download COCO Person
chmod +x download_coco.sh
./download_coco.sh

# Extract images, captions, and bbox labels
python prep_coco.py
```

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
python gen.py -m model.cn_use=frankjoshua_openpose,fusing_openpose,lllyasviel_openpose
```

List of available models can be found in `conf/config.yaml`. We have 3 available extractors at the moment (OpenPose, Canny, MediaPipeFace), If you add another control net model, make sure you add one of the following strings to its name to set the extractor to use :

- openpose
- canny
- mediapipe_face