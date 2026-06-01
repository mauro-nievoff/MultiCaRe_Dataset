# 🏥 MultiCaRe — A Multimodal Clinical Case Dataset

The [MultiCaRe dataset](https://doi.org/10.5281/zenodo.10079369) is an open-source clinical case dataset for medical image classification and multimodal AI applications, built from PubMed Central open-access case reports.

## 📊 Key Facts

- **98,000+** de-identified clinical cases from **72,000+** PubMed Central case reports
- **139,000+** medical images across multiple specialties
- Covers diverse domains including **oncology, cardiology, pathology, surgery**, and more
- Image taxonomy with **140+ classes** organized in a hierarchical structure with logical constraints (mutual exclusivity, subsumption, etc.)
- Fully open-source under **CC0 license**

---

## 🗂️ Dataset Structure

The dataset contains the following data elements:

![MultiCaRe data elements](https://github.com/user-attachments/assets/2c3f4009-dbca-4fe6-b3f0-ddbc58783cdf)

---

## ✅ Create Your Own Custom Subset

The [`multiversity`](https://github.com/mauro-nievoff/multiversity) Python library lets you create customized subsets of MultiCaRe based on filters like patient demographics, clinical keywords, image labels, and more.

### Installation

```bash
pip install multiversity
```

### Basic Usage

```python
from multiversity.multicare_dataset import MedicalDatasetCreator

# Load the dataset (downloads from Zenodo, takes 5–10 min)
mdc = MedicalDatasetCreator(directory='medical_datasets')

# Define filters
filters = [
    {'field': 'min_age', 'string_list': ['18']},
    {'field': 'gender', 'string_list': ['Male']},
    {'field': 'case_strings', 'string_list': ['tumor', 'cancer', 'carcinoma'], 'operator': 'any'},
    {'field': 'caption', 'string_list': ['metastasis', 'tumor', 'mass'], 'operator': 'any'},
    {'field': 'label', 'string_list': ['mri', 'head']}
]

# Create dataset (multimodal, text, image, or case_series)
mdc.create_dataset(
    dataset_name='male_brain_tumor_dataset',
    filter_list=filters,
    dataset_type='multimodal'
)
```

➡️ For full library documentation, visit the [`multiversity` repository](https://github.com/mauro-nievoff/multiversity).

---

## 🔍 Exploring the Data

```python
mdc.display_example()
```

This will render a sample case with its clinical narrative, image, image labels, and citation metadata.

---

## 📁 Repository Contents

| Folder | Description |
|---|---|
| `Dataset_Creation_Process/` | Notebooks detailing how the dataset was built |
| `Demos/` | Example notebooks for creating subsets and classification datasets |
| `MultiCaReClassifier/` | Classification model trained on MultiCaRe |
| `MultiCaRe_Taxonomy/` | The full image taxonomy (140+ classes) |

---

## 💡 Useful Resources

1. 📄 [Data Article (MDPI Data)](https://doi.org/10.3390/data10080123) — Full description of the dataset
2. 🗄️ [Dataset on Zenodo](https://doi.org/10.5281/zenodo.10079369) — Download the data
3. 📓 [Subset creation demo](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/Demos/customized_subset_creation.ipynb)
4. 🖼️ [Image classification demo](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/Demos/create_image_classification_datasets.ipynb)
5. 🏷️ [MultiCaRe Taxonomy](https://github.com/mauro-nievoff/MultiCaRe_Dataset/tree/main/MultiCaRe_Taxonomy)

---

## 📦 Legacy Code

If you need to work with MultiCaRe v1.0:

```python
from multiversity.multicare_v1 import *
```

---

## 🤓 How to Cite

If you use MultiCaRe in your work, please cite:

**Data Article:**
```bibtex
Nievas Offidani, M., Roffet, F., González Galtier, M. C., Massiris, M., & Delrieux, C. (2025).
An Open-Source Clinical Case Dataset for Medical Image Classification and Multimodal AI Applications.
Data, 10(8), 123. https://doi.org/10.3390/data10080123
```

**Dataset (Zenodo v3):**
```bibtex
Nievas Offidani, M. (2025). MultiCaRe: An open-source clinical case dataset for medical image
classification and multimodal AI applications (version 3) [Data set].
Zenodo. https://doi.org/10.5281/zenodo.10079369
```

---

## 🤝 Contributing

Contributions are welcome! Feel free to open issues or submit pull requests.

If you find this project useful, please consider giving it a ⭐ — it helps a lot with visibility.

For questions or collaborations, reach out on [LinkedIn](https://www.linkedin.com/in/mauronievasoffidani/).
