# RF-Carer
This project proposes RF-Carer, a fully zero-effort cross-domain respiration monitoring system.

# Abstract
Respiratory monitoring using wireless technologies has garnered significant attention for its potential in healthcare, smart cockpits, and various applications. Though extensively studied, existing systems face practical challenges in adapting to new data domains without substantial customization efforts. Current solutions attempt to address this limitation through domain-independent feature extraction or cross-domain feature translation, employing either knowledge-based sensing models or data-driven neural networks. However, these approaches typically require additional data collection or model retraining for new domains, significantly hindering their practical deployment. This project introduces RF-Carer, a fully zero-effort cross-domain respiration monitoring system. Our key innovation lies in building an explainable propagation model to transform any heterogeneous signals under unknown domains into a unified form in the signal processing layer. To further address accidental irrelevant factors, we propose to align the feature spaces while suppressing the noisy ones with contrastive learning. On this basis, we develop a one-fits-all model. To the best of our knowledge, RF-Carer is the first zero-effort cross-domain respiration monitoring work with wireless RF signals and would be a fundamental step toward real-world deployments. 

# Introduction

This project contains two parts, the lower **Signal Processing Layer** and the higher **Data Training Layer**. 

# Dataset
The dataset can be found in [RF-Carer Dataset](https://drive.google.com/drive/folders/1fX-nAjrjg7fBlwBQtSeDgTav1de6OnMJ?usp=drive_link)

# Code instruction

**Signal Processing Layer**: The input is the raw data, the output is the processed data (i.e., the RF-Carer* in the paper). We give some processed data in our [dataset](https://drive.google.com/drive/folders/1-i_IeHzX8VMRaH8-5FLwaCGnPl_b7e0z?usp=drive_link)

**Data Training Layer**: The input is the processed data (i.e., the RF-Carer* in the paper) or the augmented data (i.e., the AUG in the paper), the output is the final prediction data (i.e., the RF-Carer in the paper).

# Citation

If possible, please cite our work:
