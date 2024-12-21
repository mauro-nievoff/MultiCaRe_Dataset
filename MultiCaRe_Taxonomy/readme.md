# MultiCaRe Taxonomy
The MultiCaRe taxonomy is a hierarchical set of classes for medical image classification, with specific logical constraints among classes (such as mutual exclusivity). It has two versions: a full form with 146 classes, and a simplified version with 81 classes used in ML models. These are some of the classes present in the taxonomy:

<p align="center">
  <img src=https://github.com/user-attachments/assets/d6130af1-fb84-4f45-a3d9-b6e459155bf8 width="1200">
</p>

This folder contains:
- taxonomy owl files with the class structures from the [labels created based on image captions](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/MultiCaRe_Taxonomy/GT_MCR_TX.owx) and the [labels created using ML classifiers](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/MultiCaRe_Taxonomy/ML_MCR_TX.owx). These taxonomies can also be found on BioPortal as [MCR_TX](https://bioportal.bioontology.org/ontologies/MCR_TX) and [ML_MCR_TX](https://bioportal.bioontology.org/ontologies/ML_MCR_TX), respectively.
- a [taxonomy documentation file](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/MultiCaRe_Taxonomy/MultiCaRe%20Taxonomy%20Documentation.pdf), with detail about the classes and the class structure.
- [input taxonomies](https://github.com/mauro-nievoff/MultiCaRe_Dataset/tree/main/MultiCaRe_Taxonomy/input_format) with the format from the [Multiplex Classification Framework](https://github.com/mauro-nievoff/Multiplex_Classification), which is an approach for ML classification problems that involve many classes and different logical constraints among them.
