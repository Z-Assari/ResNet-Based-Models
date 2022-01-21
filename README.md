# Discrimination of benign and malignant solid breast masses using deep residual learning-based bimodal computer-aided diagnosis system
By Zahra Assari[^1], Ali Mahloojifar, and Nasrin Ahmadinejad.

[https://doi.org/10.1016/j.bspc.2021.103453](https://doi.org/10.1016/j.bspc.2021.103453)

This is an implementation of the models in the following paper which is published in the 
**Biomedical Signal Processing and Control**:

```
Z. Assari, A. Mahloojifar, and N. Ahmadinejad, "Discrimination of benign and malignant solid breast masses using deep residual learning-based bimodal computer-aided diagnosis system," Biomed. Signal Process. Control, vol. 73, p. 103453, 2022, https://doi.org/10.1016/j.bspc.2021.103453.
```

### Citation
In case you find the code useful, please consider giving appropriate credit to it by citing the paper above.
```
@article{assari2022103453,
title = {Discrimination of benign and malignant solid breast masses using deep residual learning-based bimodal computer-aided diagnosis system},
author = {Assari, Zahra and Mahloojifar, Ali and Ahmadinejad, Nasrin},
journal = {Biomedical Signal Processing and Control},
volume = {73},
pages = {103453},
year = {2022},
issn = {1746-8094},
doi = {https://doi.org/10.1016/j.bspc.2021.103453},
url = {https://www.sciencedirect.com/science/article/pii/S1746809421010508},
publisher = {Elsevier}
}
```

### Abstract
One of the most common breast cancer mammographic manifestation is solid mass. If the information obtained from mammography is inadequate, complementary modalities such as ultrasound imaging are used to achieve additional information. Although interest in the combination of information from different modalities is increasing, it is an extremely challenging task. In this regard, a computer-aided diagnosis (CAD) system can be an efficient solution to overcome these difficulties. However, most of the studies have focused on the development of mono-modal CAD systems, and a few existing bimodal ones rely on the extracted hand-crafted features of mammograms and sonograms. In order to meet these challenges, this paper proposes a novel bimodal deep residual learning model. It consists of the following major steps. First, the informative representation for each input image is separately constructed. Second, in order to construct the high-level joint representation of every two input images and effectively explore complementary information among them, the representation layers of them are fused. Third, all of these joint representations are fused to obtain the final common representation of the input images for the mass. Finally, the recognition result is obtained based on information extracted from all input images. The augmentation strategy was applied to enlarge the collected dataset for this study. Best recognition results on the sensitivity, specificity, F1-score, area under ROC curve, and accuracy metrics of 0.898, 0.938, 0.916, 0.964, and 0.917, respectively, are achieved by our model. Extensive experiments demonstrate the effectiveness and superiority of the proposed model over other state-of-the-art models.

### Keywords:
`Solid breast mass`, `Mammography`, `Ultrasound imaging`, `Bimodal computer-aided diagnosis system`, `Deep learning`, `Residual learning`.

[^1]: E-mail address: ZahraAssari.bme@gmail.com.
