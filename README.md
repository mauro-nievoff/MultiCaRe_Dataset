# 🏥 MultiCaRe - A Multi-Modal Clinical Dataset

The [MultiCaRe dataset](https://doi.org/10.5281/zenodo.10079369) is an open-source clinical case dataset for medical image classification and multimodal AI applications. Some facts about it:
- It contains open-access and de-identified data from over __72K case reports__ from PubMed Central, summing up more than __93K clinical cases__ and __130K images__
- The dataset contains images and cases from __different medical specialties__, such as oncology, cardiology, surgery and pathology
- Its taxonomy for medical image classification includes __more than 140 classes__ organized in a hierarchical structure with different types of logical constraints among them (such as mutual exclusivity)

This dataset contains the following data elements:
<p align="center">
  <img src=https://github.com/user-attachments/assets/2c3f4009-dbca-4fe6-b3f0-ddbc58783cdf width="800">
</p>

## ✅ Create Your Own Dataset

While you may find the whole dataset useful, you probably only need a subset of it based on your specific use case. In order to create a specific subset, first you need to install the `multiversity` library and import the MedicalDatasetCreator class:
```python
!pip install multiversity

from multiversity.multicare_dataset import MedicalDatasetCreator
```
Then, you have to instantiate the MedicalDatasetCreator class. It will take some minutes (5 to 10), as it will import all the MultiCaRe files from Zenodo. The MultiCaRe dataset and any dataset that you create based on it will all be included in same the directory (_medical_datasets_ in this example).

```python
mdc = MedicalDatasetCreator(directory = 'medical_datasets')
```
Now it's time to define what specific dataset we want by creating filters. In the example below, we are including data that meet all these criteria:
- male patients who are at least 18 years old
- cases that contain words such as 'tumor' or 'cancer'
- cases with images with 'mri' and 'brain' labels, and with captions that contain words such as 'tumor' or 'mass'

```python
filters = [{'field': 'min_age', 'string_list': ['18']},
           {'field': 'gender', 'string_list': ['Male']},
           {'field': 'case_strings', 'string_list': ['tumor', 'cancer', 'carcinoma'], 'operator': 'any'},
           {'field': 'caption', 'string_list': ['metastasis', 'tumor', 'mass'], 'operator': 'any'},
           {'field': 'label', 'string_list': ['mri', 'head']}]
```
Finally, let's create the dataset using these filters and selecting the type of dataset that we want (it can be _multimodal_, _text_, _image_ or _case_series_). This step should not take longer than 2 minutes. You can create as many datasets as you want by using the same mdc instance (you just need to change the name of the dataset and the filters).
```python
mdc.create_dataset(dataset_name = 'male_brain_tumor_dataset', filter_list = filters, dataset_type = 'multimodal')
```
Done! The dataset is ready to use now. 

## 🔍 First Approach to the Data

Now let's see one example from the dataset.

```python
mdc.display_example()
```
__Case PMC10018421_01:__

'Gender: Male'
'Age: 32'
'Clinical Case:'
"A 32-year-old male presented with a history of intermittent headache for 5 months followed by progressive gait disturbances and blurry vision. His medical and family history was insignificant. His physical examination showed ataxia and cerebellar signs including dysmetria and dysdiadochokinesia in the right upper and lower limbs. Decreased visual acuity was noted in both eyes with bilateral papilledema on ophthalmologic exam. The rest of the examination was otherwise normal. Brain MRI revealed a lesion in the right cerebellar hemisphere. The patient underwent a suboccipital craniotomy. During the procedure, frozen sections were misinterpreted as high-grade malignant glioma. The neurosurgeon decided to proceed with subtotal resection because the risks of gross total resection (GTR) outweighed the benefits considering the aggressive nature of the suspected tumor. Adjuvant therapy with a combination of radiotherapy and chemotherapy with temozolomide was initiated. Two years after surgery, the patient complained of reemergence of symptoms including gait disturbance and morning headaches suggestive of increased intracranial pressure. Brain MRI showed a 4.4 x 4.0 cm ill-defined lesion in the right cerebellum with a mixed cystic-solid pattern (shown in <PMC10018421_crn-2023-0015-0001-529741_f1_a_1_4.webp><PMC10018421_crn-2023-0015-0001-529741_f1_b_2_4.webp><PMC10018421_crn-2023-0015-0001-529741_f1_c_3_4.webp><PMC10018421_crn-2023-0015-0001-529741_f1_d_4_4.webp>). Cystic components of the tumor were hypointense on T1- and hyperintense on T2-weighted images, whereas the solid components of the tumor were hypointense or isointense on T1- and slightly hyperintense on T2-weighted images. Gadolinium-enhanced T1-weighted images showed marked enhancement of the cystic walls. The unusual clinical course and radiologic features raised suspicion for a more benign tumor than high-grade glioma. The patient underwent another surgery to alleviate the symptoms and reassess the residual lesion. During this surgery, the frozen sections suggested the diagnosis of PXA; therefore, the neurosurgeon conducted GTR. The histopathological and the immunohistochemical studies confirmed the diagnosis of PXA and eliminated the initial diagnosis as high-grade glioma is not consistent with the histopathological studies' findings. The postoperative follow-up was without complications. The patient was closely monitored thereafter."

__Image:__

![head_multicare](https://github.com/mauro-nievoff/MultiCaRe_Dataset/assets/55700369/402c63e5-408f-4f24-8e79-87832fbefb7d)

__Image Labels:__
['radiology', 't1', 'head', 'sagittal', 'mri', 'mass', 'contrast', 'spin_echo', 'head', 'radiology', 'sagittal', 'mri']

__Image Caption:__

Mixed cystic-solid pattern of PXA. MR images show multiple cystic lesions and solid masses located in the right cerebellar hemisphere. A; Sagittal view T1-weighted with contrast.

__Citation Information:__
{'pmcid': 'PMC10018421',
 'doi': '10.1159/000529741',
 'pmid': '36938309',
 'title': 'A Recurrent Pleomorphic Xanthoastrocytoma in the Cerebellum in a Young Adult: A Case Report and Review of the Literature',
 'year': '2023',
 'authors': ['Ruba Aljendi',
             'Mohammed Amr Knifaty',
             'Mohammed Amin',
             'Souliman Diab',
             'Muhammad Saleh Ali',
             'Zuheir Alshehabi'],
 'journal': 'Case Rep Neurol',
 'journal_detail': '2023 Feb 17;15(1):54-62.',
 'link': 'https://pubmed.ncbi.nlm.nih.gov/36938309/',
 'license': 'CC BY-NC'}

## :bulb: Useful Resources

1. For a detailed insight about the contents of this dataset, please refer to this [data article](https://www.sciencedirect.com/science/article/pii/S2352340923010351) published in Data In Brief (it describes MultiCaRe 1.0).
2. You can find further details about how the dataset was created by reading the notebooks from the [Dataset_Creation_Process folder](https://github.com/mauro-nievoff/MultiCaRe_Dataset/tree/main/Dataset_Creation_Process).
3. For more information about the taxonomy, refer to this [folder](https://github.com/mauro-nievoff/MultiCaRe_Dataset/tree/main/MultiCaRe_Taxonomy).
4. If you want to see a more detailed demo about how to create customized subsets, please refer to [this notebook](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/Demos/customized_subset_creation.ipynb).
5. If you want to create image classification datasets based on multiple subsets, refer to [this other notebook](https://github.com/mauro-nievoff/MultiCaRe_Dataset/blob/main/Demos/create_image_classification_datasets.ipynb).
6. A 20-minute video tutorial on MultiCaRe 1.0 is available on [YouTube](http://www.youtube.com/watch?v=LeVdLKvfQNc&t).

<p align="center">
  <a href="http://www.youtube.com/watch?v=LeVdLKvfQNc&t">
    <img src="https://github.com/user-attachments/assets/fd6a9d56-e880-46d0-9348-a6ad85f5d258" alt="multicare_tutorial_image">
  </a>
</p>

## 📓 Using an old version of the code

If you intend to work with MultiCaRe 1.0, use the following module:

```python
from multiversity.multicare_v1 import *
```

That module works exactly as the deprecated `multicare.py`.

## 🤓 How to cite
If you use this dataset, please cite.

- Data Article from Data In Brief:
```
Nievas Offidani, M. A., & Delrieux, C. A. (2024). Dataset of clinical cases, images, image labels and captions from open access case reports from PubMed Central (1990–2023). In Data in Brief (Vol. 52, p. 110008). Elsevier BV. https://doi.org/10.1016/j.dib.2023.110008
```

- Dataset from Zenodo:
```
Nievas Offidani, M. (2024). MultiCaRe: An open-source clinical case dataset for medical image classification and multimodal AI applications (2.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.13936721
```
## :wave: Final words

That's all the basics that you need to know to create a customized subset based on the MultiCaRe Dataset. I hope this was useful!

Just a few comments before you leave:

- Contributions are welcome! If you would like to collaborate on this project, feel free to open pull requests or submit issues.
- If you find this project useful or interesting, please consider giving it a star ⭐. It's a great way to show your support and helps the project gain visibility.
- If you have any questions, suggestions, or just want to say hello, feel free to reach out. You can contact me on [LinkedIn](https://www.linkedin.com/in/mauronievasoffidani/).
  
Thank you! And goodbye for now!
