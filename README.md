# 🏥 MultiCaRe - Customized Medical Dataset Creation

The [MultiCaRe Dataset](https://zenodo.org/records/10079370) is a dataset of clinical cases, images, image labels and captions from open access case reports from PubMed Central. Some facts about it:
- It contains data from over 75K open-access and de-identified case reports, summing up almost 100K clinical cases and more than 135K images
- Almost 100K patients and 400K medical doctors and researchers were involved in the creation of the articles included in the dataset (see metadata.parquet for citations)
- The dataset contains images and cases from different medical specialties, such as oncology, cardiology, surgery and pathology

You can find further details about how the dataset was created by reading the notebooks from the [Dataset_Creation_Process folder](https://github.com/mauro-nievoff/MultiCaRe_Dataset/tree/main/Dataset_Creation_Process).

## ✅ Create Your Own Dataset

While you may find the whole dataset useful, you probably only need a subset of it based on your specific use case. In order to create a specific subset, first you need to clone this repository and import the MedicalDatasetCreator class:
```python
!git clone https://github.com/mauro-nievoff/MultiCaRe_Dataset

from MultiCaRe_Dataset.multicare import MedicalDatasetCreator
```
Then, you have to instantiate the MedicalDatasetCreator class. It will take some minutes (5 to 10), as it will import all the MultiCaRe files from Zenodo. The MultiCaRe dataset and any dataset that you create based on it will all be included in same the directory (_medical_datasets_ in this example).

```python
mdc = MedicalDatasetCreator(directory = 'medical_datasets')
```
Now it's time to define what specific dataset we want by creating filters. In the example below, we are including data that meet all these criteria:
- male patients who are at least 18 years old
- cases that contain words such as 'tumor' or 'cancer'
- cases with images with 'mri' and 'brain' as normalized extractions, with captions that contain words such as 'tumor' or 'mass'

```python
filters = [{'field': 'min_age', 'string_list': ['18']},
           {'field': 'gender', 'string_list': ['Male']},
           {'field': 'case_strings', 'string_list': ['tumor', 'cancer', 'carcinoma'], 'operator': 'any'},
           {'field': 'caption', 'string_list': ['metastasis', 'tumor', 'mass'], 'operator': 'any'},
           {'field': 'normalized_extractions', 'string_list': ['mri', 'brain']}]
```
Finally, let's create the dataset using these filters and selecting the type of dataset that we want (it can be _multimodal_, _text_, _image_ or _case_series_). This step should not take longer than 2 minutes. You can create as many datasets as you want by using the same mdc instance (you just need to change the name of the dataset and the filters).
```python
mdc.create_dataset(dataset_name = 'male_brain_tumor_dataset', filter_list = filters, dataset_type = 'multimodal')
```
Done! The dataset is ready to use now.

## 🔍 First Approach to the Data
Let's see how many cases and images were included:
```python
print(f"Amount of patients: {len(mdc.filtered_cases)}")
print(f"Amount of images: {len(mdc.filtered_image_metadata_df)}")
```
```
Amount of patients: 10243
Amount of images: 352
```
Nice! Now let's take a look at one example image and its corresponding clinical case. We will create a specific function for this purpose.

```python
from IPython.display import Image, Markdown, display

def display_example(mdc, image_index):
  image_path = mdc.filtered_image_metadata_df['file_path'][image_index]
  image_labels = mdc.filtered_image_metadata_df['normalized_extractions'][image_index]

  case_id = mdc.filtered_image_metadata_df['case_id'][image_index]
  for c in mdc.filtered_cases:
    if c['case_id'] == case_id:
      gender = c['gender']
      age = c['age']
      case_text = c['case_text']
      break

  pmcid = case_id.split('_')[0]
  for r in mdc.reference_list:
    if r['pmcid'] == pmcid:
      citation = r
      break

  display(Markdown(f"**Case {case_id}:**"))
  display(f"Gender: {gender}")
  display(f"Age: {age}")
  display(f"Clinical Case:")
  display(case_text)
  display(Markdown(f"**Image:**"))
  display(Image(image_path))
  display(Markdown(f"**Image Labels:**"))
  display(Markdown(f"{image_labels}"))
  display(Markdown(f"**Citation Information:**"))
  display(citation)
```
This is how the first case of the dataset looks like:
```python
display_example(mdc, image_index = 0)
```

__Case_ID:__ PMC10018421_01

__Gender:__ Male

__Age:__ 32

__Clinical Case:__
A 32-year-old male presented with a history of intermittent headache for 5 months followed by progressive gait disturbances and blurry vision. His medical and family history was insignificant. His physical examination showed ataxia and cerebellar signs including dysmetria and dysdiadochokinesia in the right upper and lower limbs. Decreased visual acuity was noted in both eyes with bilateral papilledema on ophthalmologic exam. The rest of the examination was otherwise normal. Brain MRI revealed a lesion in the right cerebellar hemisphere. The patient underwent a suboccipital craniotomy. During the procedure, frozen sections were misinterpreted as high-grade malignant glioma. The neurosurgeon decided to proceed with subtotal resection because the risks of gross total resection (GTR) outweighed the benefits considering the aggressive nature of the suspected tumor. Adjuvant therapy with a combination of radiotherapy and chemotherapy with temozolomide was initiated. Two years after surgery, the patient complained of reemergence of symptoms including gait disturbance and morning headaches suggestive of increased intracranial pressure. Brain MRI showed a 4.4 x 4.0 cm ill-defined lesion in the right cerebellum with a mixed cystic-solid pattern (shown in Fig. 1). Cystic components of the tumor were hypointense on T1- and hyperintense on T2-weighted images, whereas the solid components of the tumor were hypointense or isointense on T1- and slightly hyperintense on T2-weighted images. Gadolinium-enhanced T1-weighted images showed marked enhancement of the cystic walls. The unusual clinical course and radiologic features raised suspicion for a more benign tumor than high-grade glioma. The patient underwent another surgery to alleviate the symptoms and reassess the residual lesion. During this surgery, the frozen sections suggested the diagnosis of PXA; therefore, the neurosurgeon conducted GTR. The histopathological and the immunohistochemical studies confirmed the diagnosis of PXA and eliminated the initial diagnosis as high-grade glioma is not consistent with the histopathological studies' findings. The postoperative follow-up was without complications. The patient was closely monitored thereafter.

__Image:__

![head_multicare](https://github.com/mauro-nievoff/MultiCaRe_Dataset/assets/55700369/402c63e5-408f-4f24-8e79-87832fbefb7d)

__Image Labels:__
['mri', 'contrast', 'pathological_finding', 'brain', 'right', 't1', 'sagittal']

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

## 🤓 How to cite
If you use this dataset, please cite:

```
Nievas Offidani, M., & Delrieux, C. (2023). The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10079370
```
## :wave: Final Words

That's all the basics that you need to know to create a customized subset based on the MultiCaRe Dataset. I hope this was useful! 😄

Just a few comments before you leave:

- Contributions are welcome! If you would like to collaborate on this project, feel free to open pull requests or submit issues.
- If you find this project useful or interesting, please consider giving it a star ⭐. It's a great way to show your support and helps the project gain visibility.
- If you have any questions, suggestions, or just want to say hello, feel free to reach out. You can contact me on [LinkedIn](https://www.linkedin.com/in/mauronievasoffidani/).
  
Thank you! And goodbye for now!
