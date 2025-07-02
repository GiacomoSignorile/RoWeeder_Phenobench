# RoWeeder for the PhenoBench Dataset

This repository documents the process and results of adapting the **RoWeeder** framework \[[1](#citation)\] for unsupervised weed detection on the **PhenoBench** dataset \[[2](#citation)\]. The original RoWeeder pipeline was designed for the multi-channel WeedMap dataset. This work details the necessary modifications to apply its powerful pseudo-labeling logic to the RGB-only PhenoBench benchmark.

Our primary goal is to provide a fair and comprehensive evaluation of the RoWeeder unsupervised labeling strategy on a new, challenging dataset. We detail the engineering solutions for data loading and feature extraction and present a full set of experiments training modern segmentation architectures like SegFormer and ERFNet on the generated pseudo-ground truth.

![RoWeeder method](res/method.svg)

---

## Installation

First, prepare the Conda environment.

```bash
# 1. Create the environment
conda create -n RoWeederPhenoBench python=3.11

# 2. Activate the environment
conda activate RoWeederPhenoBench

# 3. Install dependencies from the environment file
conda env update --file environment.yml
```

## Data Preparation: PhenoBench

This project uses the PhenoBench dataset. The following steps will guide you through downloading and preparing it.

### 1. Download and Extract PhenoBench

Download the dataset from the [official PhenoBench website](https://www.phenobench.org/dataset.html). You will need the `train` and `val` splits containing both images and semantic annotations.

After downloading, extract the archives to a `dataset` directory. Your file structure should look like this:

```
.
└── dataset/
    └── PhenoBench/
        ├── train/
        │   ├── images/
        │   └── semantics/
        └── val/
            ├── images/
            └── semantics/
```

### 2. (Optional) Visualize and Tune Parameters

Before generating labels for the entire dataset, it is highly recommended to use the interactive Streamlit tool to find the optimal parameters for the vegetation and row detectors.

```bash
streamlit run visualize_weed.py
```
In the web interface, select "PhenoBench" as the modality and adjust the sliders for the `ExGDetector` and `HoughCropRowDetector` to find settings that produce a good vegetation mask and accurate row detection across several different images. Note down these optimal parameters.

## Generating Pseudo-Ground Truth

Once you have tuned your parameters, you can generate the pseudo-GT for the entire PhenoBench `train` split.

1.  **Update the Configuration:** Open a configuration file (e.g., `configs/phenobench_config.yaml`). Ensure the parameters under `plant_detector_params` and `hough_detector_params` match the optimal values you found during visualization.

2.  **Run the Labeling Script:** Execute the `save_and_label` script. This process is computationally intensive and may take a significant amount of time. The script has been optimized to run much faster by removing the unnecessary SLIC superpixel generation.

    ```bash
    # This command reads the parameters from the YAML and saves the pseudo-GT
    # to the directory specified inside the file.
    python main.py label --parameters configs/phenobench_config.yaml
    ```
    This will create a new directory (e.g., `dataset/RoWeeder_PseudoGT_for_PhenoBench/`) containing the generated pseudo-labels for the training set.

## Training a Model on Pseudo-GT

With the pseudo-GT generated, you can now train a segmentation model. The framework is configured to use the `train` split (with pseudo-GT) for training and the `val` split (with real GT) for validation.

```bash
# This command launches a full experiment, including training and final testing.
# It uses the model, loss, and dataset paths defined in the YAML file.
python main.py experiment --parameters configs/your_experiment_config.yaml
```
For example, to run the experiment with the ERFNet model and Lovász-Softmax loss, you would point to a YAML configured for that setup.

## Citation

If you use this adaptation or the original RoWeeder framework in your research, please consider citing the original paper:

[1] **RoWeeder: Unsupervised Weed Mapping through Crop-Row Detection**
```bibtex
@inproceedings{roweeder,
  title={RoWeeder: Unsupervised Weed Mapping through Crop-Row Detection},
  author={Pasquale De Marinis, Gennaro Vessio, Giovanna Castellano},
  booktitle = {Proceedings of the IEEE/CVF European Conference on Computer Vision (ECCV) Workshops},
  year={2024}
}
```

[2] **PhenoBench: A Large-Scale Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain**
```bibtex
@article{phenobench,
  author={Weyler, Jonas and Magistri, Federico and Marks, Elias and Chong, Y. L. and Sodano, M. and Roggiolani, G. and others},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  title={PhenoBench: A Large-Scale Dataset and Benchmarks for Semantic Image Interpretation in the Agricultural Domain},
  year={2024},
  publisher={IEEE}
}
```