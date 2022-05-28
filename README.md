# Fairness in Oculomotoric Biometric Identification
This repo provides the code for reproducing the experiments in [Fairness in Oculomotoric Biometric Identification](https://doi.org/10.1145/3517031.3529633).

In the paper, we investigate the fairness of a biometric system based on eye-tracking data with respect to gender, ethnicity, and age.
![DeepEyedentification embedding evolution animation](https://user-images.githubusercontent.com/43832476/170812609-cb6d8b5a-cfc1-4e03-928a-b39596871229.gif)
The figure above shows the embedding evolution of the DeepEyedentificationLive subnets across ethnicities.


## Reproduce the experiments

### Download the data
You can download the publicly available data here: [GazeBase, a large-scale, multi-stimulus, longitudinal eye movement dataset](https://figshare.com/articles/dataset/GazeBase_Data_Repository/12912257). The corresponding paper can be found [here](https://www.nature.com/articles/s41597-021-00959-y).

### Clone this repository 
You can clone this repository by either using 
```bash
git clone git@github.com:aeye-lab/etra-fairness
```
or 
```bash
git clone https://github.com/aeye-lab/etra-fairness
```
depending on your preferences and settings.

### Install packages
Install all required python packages via:
```bash
pip install -r requirements.txt
```
### Extract data
After moving the zipped gazebase data download in the first step into the repository, extract all files by executing:
```bash
python3 extract_gazebase_data.py
```

Then you can directly start using the [DeepEyedentification](https://ecmlpkdd2019.org/downloads/paper/231.pdf) network by adjusting and executing the following scripts for the experiment you want to investigate. A description of the CLI arguments is available via `python3 deepEye_fairness_gazebase.py --help`.

A list of all bash scripts used is below. Note: Running the experiments, especially on CPU, will take some time.
### Pipeline [DeepEyedentification](https://ieeexplore.ieee.org/abstract/document/9555831)
* run experiments:
    * run_random_sampling_all_settings.sh
    * run_experiment_age.sh
    * run_experiments_VD1_VD2.sh
    * run_experiments_RAN.sh
    * run_experiments_HSS.sh
    * run_experiments_FXS.sh
    * run_experiments_BLG.sh
    * run_experiments_TEX.sh
* create score dicts:
    * run_create_score_dicts.sh

Unfortunately the [Lohr _et al._](https://ieeexplore.ieee.org/document/9304859) takes a bit more work. After downloading the data you have to adjust the following two scripts:
* extract the eye movement events (fixation, saccades, PSO)
    * adjust the path in the scripts from  [A novel evaluation of two related and two independent algorithms for eye movement classification during reading](https://digital.library.txstate.edu/handle/10877/6874). The paper can be found [here](https://link.springer.com/epdf/10.3758/s13428-018-1050-7).
* adjust the statistical feature extraction: [Study of an Extensive Set of Eye Movement Features: Extraction Methods and Statistical Analysis](https://digital.library.txstate.edu/handle/10877/6904). The paper can be found [here](https://pubmed.ncbi.nlm.nih.gov/33828682/). Note, you don't only have to adjust the script to match your data path (the data created by MNH) but also such that the algorithm only takes one eye movement event as input instead of the complete sequence.

Afterwards copy the data to `lohr_feature_data/`. You can execute all experiments with the bash scripts below. CLI options are available via  `python3 lohr_fairness_gazebase.py --help`.
### Pipeline [Lohr _et al._](https://ieeexplore.ieee.org/document/9304859)
* run experiments:
    * run_experiments_lohr.sh
    * run_experiments_lohr_adam_w.sh
* create score dicts:
    * run_create_score_dicts_lohr.sh

### Calculate fairness metrics and visualize results
You can calculate the fairness metrics and visualize the results from your experiments with the notebooks: 
* Fairness:
    * plot_results_deepEye.ipynb
    * plot_results_lohr.ipynb
* Visualize embeddings:
    * plot_t-sne_visualization_deepEye.ipynb
    * plot_t-sne_visualization_lohr.ipynb
* Eye movement similarities for different demographics:
    * inspect_differences.ipynb

## Contribution
If you find any issues, please open an issue in the issue tracker.

If you want, you can also test your own oculomotoric biometric models substituting it within the piplines described above. 


## Cite our work
If you use our code for your research, please consider citing our paper:

```bibtex
@inproceedings{10.1145/3517031.3529633,
author = {Prasse, Paul and Reich, David Robert and Makowski, Silvia and J\"{a}ger, Lena A. and Scheffer, Tobias},
title = {Fairness in Oculomotoric Biometric Identification},
year = {2022},
isbn = {9781450392525},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
doi = {10.1145/3517031.3529633},
booktitle = {2022 Symposium on Eye Tracking Research and Applications},
articleno = {22},
numpages = {8},
keywords = {fairness, neural networks, biometrics},
location = {Seattle, WA, USA},
series = {ETRA '22}
}
```
