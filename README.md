This repository includes all code related to our work, entitled: Improved Brain Disorder Classification via Inter-individual Variability Characterization within a Self-Supervised Learning Framework. 
Our repository includes the following Python scripts:
1) SSL: performs the contrastive training and testing (fingerprinting) using HCP data
2) SSL_ABIDE: performs the contrastive training using data from the ABIDE I dataset and evaluates the model on an ASD classification task
3) SL_ABIDE: performs a supervised training, for ASD classfication, using data from the ABIDE I dataset. The testing and validtion sets of SSL_ABIDE and SL_ABIDE are the same
4) benchmark_results_full_signal: calculates the subject fingerprinting rates of conventional PCC-based functional connectomes, using the whole rs-fMRI signals (1200 samples)
5) benchmark_results: calculates the subject fingerprinting rates of conventional PCC-based functional connectomes, using parts of the rs-fMRI signals
6) classifier: performs the classification of ASD vs NC, using a single linear layer
7) utils: includes some utility functions and classes
8) plot_results_task1: plots the results of task 1 (subject fingerprinting)
9) get_results_task2: summarizes and prints the results of tasks 2 and 3 (classification of ASD using contrastive and supervised learning, respectively)
10) prepare_train_data: prepares and saves the data that are used to perform the contrastive training on HCP data (used by the SSL.py script)
11) prepare_test_data: prepares and saves the data that are used to perform the subject fingerprinting (used by the SSL.py and benchmark_results.py scripts)
12) prepare_train_test_ABIDE: prepares the data that are used to perform the contrastive training and ASD classification (testing) on ABIDE I data (used by the SSL_ABIDE and SL_ABIDE.py scripts)

ABIDE I data can be downloaded using download_abide_preprocessed_dataset.ipynb script from the following GitHub repository: https://github.com/ShawonBarman/How-to-download-ABIDE-Preprocessed-dataset-for-autism-detection/blob/main/download_abide_preprocessed_dataset.ipynb

HCP data can be downloaded from the following link (account is required): https://db.humanconnectome.org/app/action/DownloadPackagesAction
