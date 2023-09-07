import hydra
from omegaconf import DictConfig, OmegaConf

# In this file the approach to measure quality will be the extensive library
# IQA-Pytorch: https://github.com/chaofengc/IQA-PyTorch
# Read also the paper: https://arxiv.org/pdf/2208.14818.pdf

# There are basically two approaches to measure image quality
# - full reference: compare againts a real pristine image
# - no reference: compute metrics following a learned opinion

# Because images are generated there is no reference image to compare to. We
# will be using with the no-reference metrics

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    print(cfg)

if __name__ == '__main__':
    main()
