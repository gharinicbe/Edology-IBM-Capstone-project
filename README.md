Revolutionizing 3D Brain Tumor Segmentation with SLMSA Network for Transformative Neuro-Insights
Overview
This repository houses the code and resources for a groundbreaking project focused on advancing 3D brain tumor segmentation using the Short Long-Term Memory Self-Attention (SLMSA) network. The project's core methodology is based on a recently published paper dated February 16, 2023, accessible here. The SLMSA model represents a transformative approach, leveraging a transformer-based architecture to enhance the accuracy and efficiency of brain tumor segmentation in volumetric datasets.

Files Uploaded
Capstone project PPT.pptx: Presentation summarizing the key aspects of the project.
Capstone Summary.docx: Document providing an overview and summary of the capstone project.
bratsslmsa.py: Python script containing the SLMSA model implementation.
diceioumetrics4ch.py: Python script with Dice and IoU metrics calculation functions.
slmsa-diceloss-model-train.ipynb: Jupyter notebook for training the SLMSA model with Dice loss.
slmsa-diceloss-model-retrain.ipynb: Jupyter notebook for retraining the SLMSA model.
finaltestdice7.ipynb: Jupyter notebook for final testing and evaluation.
Project Objective
Brain tumor segmentation is a critical step in medical diagnosis and treatment planning. The SLMSA model, introduced in this project, addresses the challenges posed by traditional methods in handling 3D imaging data. This repository presents a comprehensive solution, offering a novel approach that promises improved accuracy and efficiency in delineating tumor boundaries.

Methodology
The SLMSA model, detailed in the referenced publication, combines Short-Term Memory (STM) and Long-Term Memory (LTM) mechanisms with self-attention, making it well-suited for complex 3D imaging datasets. The project utilizes the BraTS 2020 dataset, incorporating manual annotations for ground truth labels. The training process involves the AdamW optimizer, and metrics such as accuracy, IoU score, and loss are monitored.

Results
The SLMSA model exhibits outstanding performance, achieving metrics of 98.64% accuracy, 74.15% IoU score, and 16.63% loss on the training set within 100 epochs. On the validation set, metrics remain high at 96.26% accuracy, 46.22% IoU score, and 44.97% loss. These results underscore the model's proficiency in capturing intricate details of brain tumor structures in 3D.

Conclusion
This project marks a significant milestone in the realm of 3D brain tumor segmentation, representing a pioneering effort in the field. The SLMSA model, crafted with inspiration from recent strides in neural architecture design, not only showcases exceptional performance but also holds the promise of unlocking transformative neuro-insights. The amalgamation of an advanced model architecture, rigorous data preparation, and cutting-edge optimization techniques stands as a testament to the project's success.

One of the distinguishing features of this project is the incorporation of 3D visualization using the itkwidgets library. This tool facilitates an interactive and immersive environment for exploring segmentation results, providing a valuable aid for medical professionals and researchers. The visualization capabilities elevate the project beyond conventional approaches, making it an advanced and indispensable tool in the domain of brain tumor analysis. As we look ahead, the strides made in this project have the potential to significantly enhance medical image analysis, ultimately leading to improved patient care.

Citations and References
[1] B. H. Menze, A. Jakab, S. Bauer, J. Kalpathy-Cramer, K. Farahani, J. Kirby, et al. "The Multimodal Brain Tumor Image Segmentation Benchmark (BRATS)", IEEE Transactions on Medical Imaging 34(10), 1993-2024 (2015) DOI: 10.1109/TMI.2014.2377694
[2] S. Bakas, H. Akbari, A. Sotiras, M. Bilello, M. Rozycki, J.S. Kirby, et al., "Advancing The Cancer Genome Atlas glioma MRI collections with expert segmentation labels and radiomic features", Nature Scientific Data, 4:170117 (2017) DOI: 10.1038/sdata.2017.117
[3] S. Bakas, M. Reyes, A. Jakab, S. Bauer, M. Rempfler, A. Crimi, et al., "Identifying the Best Machine Learning Algorithms for Brain Tumor Segmentation, Progression Assessment, and Overall Survival Prediction in the BRATS Challenge", arXiv preprint arXiv:1811.02629 (2018)
[4] https://www.kaggle.com/datasets/awsaf49/brats20-dataset-training-validation
[5] https://ietresearch.onlinelibrary.wiley.com/doi/10.1049/cit2.12179
[6] https://github.com/bnsreenu/python_for_microscopists









