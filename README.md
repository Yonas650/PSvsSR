# PSvsSR

This repository contains code and environments for training and evaluating EEG-based classification models using Leave-One-Subject-Out (LOSO) cross-validation. Models include ATCNet, EEGConformer, and EEGInception. Each model is applied to four datasets: Delay, Thermal, Urgency, and Vibration.

## Repository Structure

```
├── environments/
│   └── environment.yml        #conda environment specification
├── models/
│   ├── ATCNet.py              #ATCNet model implementation
│   ├── Conformer.py           #EEGConformer model implementation
│   └── Inception.py           #EEGInception model implementation
├── train_test.py              #main script for data processing, training, and evaluation
├── LICENSE                    #MIT License
└── README.md                  #this file
```

## Requirements

* Python 3.10+
* CUDA-compatible GPU (optional but recommended)
* Conda or Virtualenv

Key Python packages:

```
- numpy
- scipy
- mat73
- torch
- torcheval
- scikit-learn
- matplotlib
- tensorboard
```

## Setup

1. **Clone the repo**

   ```bash
    git clone https://github.com/Yonas650/PSvsSR.git
    cd PSvsSR
   ```
2. **Create environment**

   ```bash
   conda env create -f environments/environment.yml
   conda activate eeg-gpu
   ```

## Usage

1. **Configure script**

   * In `train_test.py`, set:

     ```python
     model_choice = ATCNet            # or EEGConformer, EEGInception
     dataset_choice = "Vibration.mat"  # or Delay.mat, Thermal.mat, Urgency.mat
     label_choice = "PS"             # or "SR"
     ```

2. **Run training & evaluation**

   ```bash
   python train_test.py
   ```

3. **Results**

   * Outputs are saved under: `/scratch/yma9130/PSvsSR/results_<MODELDATA_LABEL>/`
   * Subfolders:

     * `accuracy/` — per-subject scores (`.npy`)
     * `cfx/` — confusion matrices (`.pdf`)
     * `curves/` — learning curves (`.pdf`)
     * `logs/` — TensorBoard logs

4. **Visualize with TensorBoard**

   ```bash
   tensorboard --logdir /scratch/yma9130/PSvsSR/results_<...>/logs
   ```

## License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file for details.

## Contact

Yonas Atinafu — [github.com/yonas650](https://github.com/yonas650) — [yma9130@nyu.edu](mailto:yma9130@nyu.edu)
