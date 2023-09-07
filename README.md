# Data Augmentation with Stable Diffusion for Object Detection using YOLOv8

This is a data generation framework that uses stable diffusion with control net. Models can be trained using a mix of real and generated data. They can also be logged and evaluated.

<img src="docs/images/general_pipeline.png" />


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
