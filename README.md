# Data Augmentation with Stable Diffusion for Object Detection using YOLOv8

This is a data generation framework that uses stable diffusion with ControlNet. Models can be trained using a mix of real and generated data. They can also be logged and evaluated.

<img src="docs/images/general_pipeline.png" />

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