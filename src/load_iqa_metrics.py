import hydra
import matplotlib.pyplot as plt
import os
from omegaconf import DictConfig
from pathlib import Path
import json


def plot_details(metrics_content, save_to):
    metric_names = list(metrics_content.keys())
    metric_names.remove('image_paths')
    metric_names.remove('name')

    dataset_size = len(metrics_content[metric_names[0]])
    x_axis = [i for i in range(dataset_size)]
    nbr_metrics = len(metric_names)

    fig, axs = plt.subplots(nbr_metrics, figsize=(30, 20))

    for i in range(nbr_metrics):
        metric = metric_names[i]
        scores = metrics_content[metric]
        mean_score = sum(scores) / dataset_size

        axs[i].set_title(label=f'metric: {metric}, mean score: {mean_score}')
        axs[i].plot(x_axis, scores)

    plt.legend()
    plt.savefig(save_to, dpi=100)
    plt.show()


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    # BASE PATHS, please used these when specifying paths
    data_path = cfg['data']
    # keep track of what feature was used for generation too in the name
    base_path = os.path.join(*data_path['base']) if isinstance(data_path['base'], list) else data_path['base']
    GEN_DATA_PATH =  Path(base_path) / data_path['generated'] / cfg['model']['cn_use']

    IQA_PATH = Path(base_path) / 'iqa'
    IQA_PATH.mkdir(parents=True, exist_ok=True)
    file_json_iqa = IQA_PATH / f"{cfg['model']['cn_use']}_iqa.json"
    image_save_to = IQA_PATH / f"{cfg['model']['cn_use']}_iqa.png"

    metrics_content = {}
    with open(file_json_iqa, 'r') as f:
        metrics_content = json.load(f)

    plot_details(metrics_content, image_save_to)


if __name__ == '__main__':
    main()