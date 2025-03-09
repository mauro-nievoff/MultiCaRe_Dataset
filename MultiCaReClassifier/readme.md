# MultiCaReClassifier for Medical Image Classification

The MultiCaReClassifier is a model ensemble used for multilabel medical image classification. It includes classes such as:
- image_type: 'radiology', 'pathology', 'endoscopy', 'ophthalmic_imaging', 'medical_photograph', 'electrography', 'chart'.
- image_subtype: 'ultrasound', 'x_ray', 'ct', 'mri', 'h&e', 'immunostaining', 'fundus_photograph', 'ekg', 'eeg', etc.
- radiology_region: 'thorax', 'head', 'abdomen', 'upper_limb', 'lower_limb', etc.
- radiology_view: 'frontal', 'sagittal', 'axial', 'oblique', etc.


1. Clone the HuggingFace repo:
```python
!git clone https://huggingface.co/mauro-nievoff/MultiCaReClassifier
```

2. Change the directory:
```python
%cd /content/MultiCaReClassifier
```

3. Import the MultiCaReClassifier class:
```python
from MultiCaReClassifier.pipeline import *
```

4. Get the predictions for a given image folder:
```python
predictions = MultiCaReClassifier(image_folder = '/sample/image/folder/path')
predictions.data.head()
```

The confusion matrices of all the models from the model ensemble can be found in the confusion_matrices folder.

**Model Training by:** Facundo Roffet

**Data Curation and Postprocessing by:** Mauro Nievas Offidani
