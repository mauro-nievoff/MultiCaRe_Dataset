import shutil
import os
import pyarrow.parquet as pq
import json
import pandas as pd
import ast
from datetime import datetime
import requests

class MedicalDatasetCreator():

  def __init__(self, directory = 'MultiCaRe'):

    '''The multicare_v1 module belongs to the first version of the MultiCaRe dataset (use the multicare_dataset module for an updated version of it).

    The MedicalDatasetCreator class is used to datasets based on the MultiCaRe Dataset given a list of filters and a dataset type.

    - directory (str): Name of the directory that will contain the MultiCaRe Dataset and all the datasets that are created based on it.
    '''

    ### Creation of the directory
    self.directory = directory
    self.whole_dataset_path = f"{self.directory}/whole_multicare_dataset"
    if not os.path.exists(self.whole_dataset_path):
      os.makedirs(self.whole_dataset_path)

    ### The MultiCaRe Dataset is downloaded and unpacked
    if os.listdir(self.whole_dataset_path) == []:
      print('Downloading the MultiCaRe Dataset from Zenodo. This may take 5 to 10 minutes.')
      self._download_dataset()
    else:
      print('The MultiCaRe Dataset is already downloaded.')

    ### The dataset files are imported
    print('Importing and pre-processing the main files.')
    self.full_metadata = pq.read_table(f"{directory}/whole_multicare_dataset/metadata.parquet").to_pydict()
    self.full_cases = pq.read_table(f"{directory}/whole_multicare_dataset/cases.parquet").to_pydict()
    self.full_image_metadata_df = pd.read_csv(f"{directory}/whole_multicare_dataset/captions_and_labels.csv")

    ### The image metadata df is preprocessed by turning string columns with lists into list type, by merging all normalized extractions into one column and adding a column with the link to the raw image
    for list_column in self.full_image_metadata_df.columns[6:]:
      self.full_image_metadata_df[list_column] = self.full_image_metadata_df[list_column].apply(ast.literal_eval)
    self.full_image_metadata_df['normalized_extractions'] = self.full_image_metadata_df.apply(lambda row: list(set([element for col in self.full_image_metadata_df.columns[8:] for element in row[col]])), axis=1)
    self.full_image_metadata_df['raw_image_link'] = self.full_image_metadata_df.main_image.apply(lambda x: f"https://www.ncbi.nlm.nih.gov/pmc/articles/{x.split('_')[0]}/bin/{'_'.join(x.split('_')[2:])}")

    ### Parameters are created with all the possible values for article year, licenes, keywords, mesh terms, ages, genders, image labels and normalized extractions.
    self.year_list = list(set([m['year'] for m in self.full_metadata['article_metadata']]))
    self.license_list = list(set([m['license'] for m in self.full_metadata['article_metadata']]))
    self.keyword_list = list(set([
        k for m in self.full_metadata['article_metadata']
        if 'keywords' in m and m['keywords'] is not None
        for k in m['keywords']
    ]))
    self.mesh_term_list = list(set([
        k for m in self.full_metadata['article_metadata']
        if 'mesh_terms' in m and m['mesh_terms'] is not None
        for k in m['mesh_terms']
    ]))
    self.age_list = list(set([c['age'] for case_list in self.full_cases['cases'] for c in case_list]))
    self.gender_list = list(set([c['gender'] for case_list in self.full_cases['cases'] for c in case_list]))
    self.image_label_list = [l for l in self.full_image_metadata_df.generic_label.explode().unique() if type(l) == str]
    self.normalized_extraction_list = {column: [l for l in self.full_image_metadata_df[f"{column}"].explode().unique() if type(l) == str] for column in self.full_image_metadata_df.columns[8:-1]}
    print('Done!')

  def _download_dataset(self):

    '''Method used to download and unzip the MultiCaRe Dataset from Zenodo.'''

    response = requests.get("https://zenodo.org/api/records/10079370/files-archive", stream=True)

    if response.status_code == 200:
      # Open the local file for writing in binary mode
      with open("multicare.zip", 'wb') as file:
        # Iterate over the content of the response and write to the file
        for chunk in response.iter_content(chunk_size=128):
          file.write(chunk)
    else:
      print(f"Failed to download file. Status code: {response.status_code}")

    shutil.unpack_archive('multicare.zip', self.whole_dataset_path)
    os.remove('multicare.zip')
    for i in range(9):
      shutil.unpack_archive(f'{self.whole_dataset_path}/PMC{i+1}.zip', self.whole_dataset_path)
      os.remove(f'{self.whole_dataset_path}/PMC{i+1}.zip')

  def create_dataset(self, dataset_name, filter_list, dataset_type = 'multimodal'):

    '''The create_dataset method is used to create a customized subset from the MultiCaRe Dataset.

    - dataset_name (str): Name of the subset. The subset will be created in a folder with this name inside the main directory.
    - filter_list (list): List of dictionaries with the keys below.
      - 'field': Some fields are applied at an article level ('min_year', 'max_year', 'license', 'keywords', 'mesh_terms'), some at a clinical case level ('min_age', 'max_age', 'gender', 'case_strings') and some at an image level ('caption', 'label', 'normalized_extraction').
      - 'string_list': The values that are searched in a given field. For min_year, max_year, min_age, and max_age, the list should include only one value (if there are many, it will pick the min or max among them depending on the case).
      - 'operator': Used for fields with multiple values in the string list. It can be either 'all' (default value), 'any' or 'none'.
      - 'match_type': Used for keywords and mesh_terms. It can be either 'full_match' or 'partial_match'.
        Example: {'field': 'keywords', 'string_list': ['diabetes'], 'match_type': 'partial_match'} will retrieve all the cases with at least one keyword that contains the substring 'diabetes'. If 'full_match' had been used, it would only retrieve cases which include the keyword 'diabetes' (exact match).
      - 'matching_case': Used for captions. It can be either True (if the casing from the search term is relevant) or False (if not).
    - dataset_type (str): Required type of dataset. It can be either 'text', 'image', 'multimodal' or 'case_series'. All the dataset types will include a readme.txt, an article_metadata.json and a reference_list.json (with citation information from case report articles).
      - text: The dataset contains a csv file with case_id, pmcid, case_text, age and gender of the patient.
      - image: The dataset contains a folder with images, and a json file with file_id, file_path, normalized_extractions, labels, caption, raw_image_link (from PMC), case_id, license, split_during_preprocessing (True if the raw image included more than one sub images).
      - multimodal: The dataset contains a combination of the files frmo text and image datasets.
      - case_series: The dataset contains a folder with images (there is one folder per patient), and a csv file with cases including case_id, pmcid, case_text, age, gender, link to the case report article, amount_of_images for the specific case, and image_folder.

    Before creating any filter, you can check the parameters with possible values: year_list, license_list, keyword_list, mesh_term_list, age_list, gender_list, image_label_list, and normalized_extraction_list.
    For example, you can get all the possible keywords present in the whole dataset by using the code below.

    mdc = MedicalDatasetCreator()
    keyword_list = mdc.keyword_list
    '''

    ### Creation of the directory
    self.dataset_name = dataset_name
    self.dataset_type = dataset_type
    if self.dataset_name in os.listdir(self.directory):
      raise ValueError("Dataset name already exists. Please use a different name.")
    os.makedirs(f"{self.directory}/{self.dataset_name}")

    ### Checking for errors in the filters
    if type(filter_list) != list:
      filter_list = [filter_list]
    for filter in filter_list:
      if (type(filter) != dict) or ('field' not in filter.keys()) or ('string_list' not in filter.keys()):
        raise ValueError("Each filter should be a dictionary including the keys 'field' and 'string_list'.")

    ### The filters are split depending on their type (article, case or image):
    self.full_filter_list = filter_list
    self.article_filter_list = []
    self.case_filter_list = []
    self.image_filter_list = []
    for filter in self.full_filter_list:
      if filter['field'] in ['min_year', 'max_year', 'license', 'keywords', 'mesh_terms']:
        self.article_filter_list.append(filter)
      elif filter['field'] in ['min_age', 'max_age', 'gender', 'case_strings']:
        self.case_filter_list.append(filter)
      elif filter['field'] in ['caption', 'label', 'normalized_extractions']:
        self.image_filter_list.append(filter)
      else:
        raise ValueError("Wrong filter 'field'. Possible values are: 'min_year', 'max_year', 'license', 'keywords', 'mesh_terms', 'min_age', 'max_age', 'gender', 'case_strings', 'caption', 'label', 'normalized_extractions'.")

    ### The articles are filtered according to their year, license, keywords and mesh terms
    self.filtered_metadata = self.full_metadata['article_metadata'].copy()
    for dct in self.article_filter_list:
      included_data = []
      if dct['field'] == 'min_year':
        min_year = min([int(y) for y in dct['string_list']])
        for article in self.filtered_metadata:
          if int(article['year']) >= min_year:
            included_data.append(article)
      elif dct['field'] == 'max_year':
        max_year = max([int(y) for y in dct['string_list']])
        for article in self.filtered_metadata:
          if int(article['year']) <= max_year:
            included_data.append(article)
      elif dct['field'] == 'license':
        relevant_license_list = dct['string_list']
        for article in self.filtered_metadata:
          if article['license'] in relevant_license_list:
            included_data.append(article)
      elif dct['field'] in ['keywords', 'mesh_terms']:
        for article in self.filtered_metadata:
          relevant_article = self._get_term_matches(article_dict = article, filter_dict = dct)
          if relevant_article:
            included_data.append(article)
      self.filtered_metadata = included_data

    ### Cases are filtered according to age, gender, and case_strings
    self.filtered_cases = [c for case_list in self.full_cases['cases'] for c in case_list]
    for dct in self.case_filter_list:
      included_data = []
      if dct['field'] == 'min_age':
        min_age = min([int(y) for y in dct['string_list']])
        for case_ in self.filtered_cases:
          if (case_['age'] is not None) and (int(case_['age']) >= min_age):
            included_data.append(case_)
      elif dct['field'] == 'max_age':
        max_age = max([int(y) for y in dct['string_list']])
        for case_ in self.filtered_cases:
          if (case_['age'] is not None) and (int(case_['age']) <= max_age):
            included_data.append(case_)
      elif dct['field'] == 'gender':
        relevant_gender_list = [g.lower() for g in dct['string_list']]
        for case_ in self.filtered_cases:
          if case_['gender'].lower() in relevant_gender_list:
            included_data.append(case_)
      elif dct['field'] == 'case_strings':
        if 'operator' not in dct.keys():
          dct['operator'] = 'all'
        for case_ in self.filtered_cases:
          if dct['operator'] == 'all':
            relevant_case = True
            for string in dct['string_list']:
              if string.lower() not in case_['case_text'].lower():
                relevant_case = False
                break
          elif dct['operator'] == 'any':
            relevant_case = False
            for string in dct['string_list']:
              if string.lower() in case_['case_text'].lower():
                relevant_case = True
                break
          elif dct['operator'] == 'none':
            relevant_case = True
            for string in dct['string_list']:
              if string.lower() in case_['case_text'].lower():
                relevant_case = False
                break
          if relevant_case:
            included_data.append(case_)
      self.filtered_cases = included_data

    ### Images are filtered according to their labels, normalized extractions and the contents of their captions
    self.filtered_image_metadata_df = self.full_image_metadata_df.copy()
    self.filtered_image_metadata_df = self.filtered_image_metadata_df[['file_id', 'file', 'raw_image_link', 'patient_id', 'license', 'caption', 'chunk', 'generic_label', 'normalized_extractions']]
    for dct in self.image_filter_list:
      if 'matching_case' in dct.keys():
        matching_case = dct['matching_case']
      else:
        matching_case = False

      if 'operator' in dct.keys():
        logical_operator = dct['operator']
      else:
        logical_operator = 'all'

      if dct['field'] == 'caption':
        if logical_operator == 'any':
          pattern = '|'.join(dct['string_list'])
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['caption'].str.contains(pattern, case=matching_case, regex=True)]

        elif logical_operator == 'none':
          pattern = '|'.join(dct['string_list'])
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[~self.filtered_image_metadata_df['caption'].str.contains(pattern, case=matching_case, regex=True)]

        elif logical_operator == 'all':
          for substring in caption_substrings['substrings']:
            self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['caption'].str.contains(substring, case=matching_case, regex=True)]

      elif dct['field'] == 'normalized_extractions':
        if logical_operator == 'any':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['normalized_extractions'].apply(lambda x: any(elem.lower() in x for elem in dct['string_list']))]

        elif logical_operator == 'none':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[~self.filtered_image_metadata_df['normalized_extractions'].apply(lambda x: any(elem.lower() in x for elem in dct['string_list']))]

        elif logical_operator == 'all':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['normalized_extractions'].apply(lambda x: all(elem.lower() in x for elem in dct['string_list']))]

      elif dct['field'] == 'label':
        if logical_operator == 'any':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['generic_label'].apply(lambda x: any(elem in x for elem in dct['string_list']))]

        elif logical_operator == 'none':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[~self.filtered_image_metadata_df['generic_label'].apply(lambda x: any(elem in x for elem in dct['string_list']))]

        elif logical_operator == 'all':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['generic_label'].apply(lambda x: all(elem in x for elem in dct['string_list']))]

    self.filtered_image_metadata_df = self.filtered_image_metadata_df.reset_index(drop = True)
    self.filtered_image_metadata_df['file_path'] = self.filtered_image_metadata_df['file'].apply(lambda x: f"{self.directory}/{self.dataset_name}/images/{x[:4]}/{x[:6]}/{x}")

    ### After applying the filters, the resulting id lists are harmonized (e.g. if all the cases from an article were filtered out, then that article is not included in the dataset even if it had not been filtered out by article filters)
    if self.dataset_type in ['text', 'case_series']:
      common_pmcids = set([m['pmcid'] for m in self.filtered_metadata]) & set([c['case_id'].split('_')[0] for c in self.filtered_cases]) & set(list(self.filtered_image_metadata_df['patient_id'].apply(lambda x: x.split('_')[0])))
      common_cases = set([c['case_id'] for c in self.filtered_cases]) & set(list(self.filtered_image_metadata_df['patient_id']))
    else:
      common_pmcids = set([m['pmcid'] for m in self.filtered_metadata]) & set([c['case_id'].split('_')[0] for c in self.filtered_cases])
      common_cases = set([c['case_id'] for c in self.filtered_cases])

    ### Metadata is split into case_report_citations.json and article_metadata.json
    self.filtered_metadata = [m for m in self.filtered_metadata if m['pmcid'] in common_pmcids]

    self.reference_list = []
    for d in self.filtered_metadata:
      self.reference_list.append({key: d[key] for key in ['pmcid', 'doi', 'pmid', 'title', 'year', 'authors', 'journal', 'journal_detail', 'link', 'license']})
    with open(f"{self.directory}/{self.dataset_name}/case_report_citations.json", 'w') as json_file:
        json.dump(self.reference_list, json_file)

    self.article_metadata = []
    for d in self.filtered_metadata:
      self.article_metadata.append({key: d[key] for key in ['pmcid', 'license', 'keywords', 'mesh_terms', 'major_mesh_terms']})
    with open(f"{self.directory}/{self.dataset_name}/article_metadata.json", 'w') as json_file:
        json.dump(self.article_metadata, json_file)

    ### The case.csv file is created
    self.filtered_cases = [c for c in self.filtered_cases if ((c['case_id'].split('_')[0] in common_pmcids) and (c['case_id'] in common_cases))]
    self.case_df = pd.DataFrame(self.filtered_cases)
    self.case_df['pmcid'] = self.case_df['case_id'].apply(lambda x: x.split('_')[0])
    self.case_df = self.case_df[['case_id', 'pmcid', 'gender', 'age', 'case_text']]
    if self.dataset_type in ['text', 'multimodal']:
      self.case_df.to_csv(f"{self.directory}/{self.dataset_name}/cases.csv", index = False)
    elif self.dataset_type == 'case_series':
      self.case_df['link'] = self.case_df['pmcid'].apply(lambda x: f"https://www.ncbi.nlm.nih.gov/pmc/articles/{x}/")
      image_counts = pd.DataFrame(self.filtered_image_metadata_df['patient_id'].value_counts()).reset_index().rename({'count': 'amount_of_images', 'patient_id': 'case_id'}, axis = 1)
      self.case_df = pd.merge(self.case_df, image_counts)
      self.case_df['amount_of_images'].fillna(0, inplace=True)
      self.case_df['image_folder'] = self.case_df.apply(lambda x: f"images/{x['case_id'][:4]}/{x['case_id'][:6]}/{x['case_id']}", axis = 1)
      self.case_df[self.case_df['amount_of_images'] == 0]['image_folder'] = 'None'
      self.case_df.to_csv(f"{self.directory}/{self.dataset_name}/cases.csv", index = False)

    ### The image_metadata.json file is created for image and multimodal dataset, and images are copied to their corresponding paths
    self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['patient_id'].apply(lambda x: x.split('_')[0]).isin(common_pmcids) & self.filtered_image_metadata_df['patient_id'].isin(common_cases)]
    self.filtered_image_metadata_df.reset_index(drop = True, inplace = True)

    if self.dataset_type in ['image', 'multimodal']:
      self.filtered_image_metadata_df.drop(['file', 'chunk'], axis=1, inplace = True)
      self.filtered_image_metadata_df['split_during_preprocessing'] = self.filtered_image_metadata_df.file_path.str.contains('undivided')
      self.filtered_image_metadata_df = self.filtered_image_metadata_df[['file_id', 'file_path', 'normalized_extractions', 'generic_label', 'caption', 'raw_image_link', 'patient_id', 'license', 'split_during_preprocessing']]
      self.filtered_image_metadata_df.rename(columns={'generic_label': 'labels', 'patient_id': 'case_id'}, inplace=True)
      self.filtered_image_metadata_df.to_json(f"{self.directory}/{self.dataset_name}/image_metadata.json", orient='records', lines=True)
      for file_path in self.filtered_image_metadata_df['file_path']:
        new_path = f"{file_path}"
        old_path = f"{self.directory}/whole_multicare_dataset/{'/'.join(file_path.split('/')[3:])}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(old_path, new_path)

    if self.dataset_type == 'case_series':
      for idx in range(len(self.filtered_image_metadata_df)):
        file_path = self.filtered_image_metadata_df['file_path'][idx]
        patient_id = self.filtered_image_metadata_df['patient_id'][idx]
        new_path = f"{file_path.replace(patient_id.split('_')[0] + '_', patient_id + '/')}"
        old_path = f"{self.directory}/whole_multicare_dataset/{'/'.join(file_path.split('/')[3:])}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(old_path, new_path)

    ### Finally, a readme.txt file is created
    readme_string = f'''README

This dataset is based on the MultiCaRe Dataset (1). The citation information from the case reports that were used to create this dataset can be found in the file case_report_citations.json.

Dataset Information:
- Name: {self.dataset_name}
- Type: {self.dataset_type}
- Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
- Filters: {self.full_filter_list}

1. Nievas Offidani, M., & Delrieux, C. (2023). The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10079370
'''
    with open(f"{self.directory}/{self.dataset_name}/readme.txt", 'w') as file:
      file.write(readme_string)

    print(f"The {self.dataset_name} was successfully created!")
    if self.dataset_type in ['image', 'multimodal']:
      comments_string = '''
Suggestions:
- Image captions: If you intend to use them, consider prioritizing images with 'split_during_preprocessing' == False.
  Many captions needed to be split during caption preprocessing, and the resulting strings may have some minor issues such as extra special characters or wrong capitalization.
- Image labels: They were created programatically based on image captions (they were not annotated manually).
  If you intend to use image labels, consider having them manually reviewed by a medical doctor or an SME.'''
      print(comments_string)

  def _get_term_matches(self, article_dict, filter_dict):
    '''This method is used to use match keywords or mesh terms to the string lists from the corresponding filters.'''

    if 'match_type' in dct.keys():
      match_type = dct['match_type']
    else:
      match_type = 'full_match'

    if 'operator' in dct.keys():
      logical_operator = filter_dict['operator']
    else:
      logical_operator = 'all'

    article_inclusion = True
    if article_dict[filter_dict['field']] is None:
      terms = []
    else:
      terms = [t.lower() for t in article_dict[filter_dict['field']]]
    if logical_operator == 'all':
      for string in filter_dict['string_list']:
        if match_type == 'full_match':
          if string.lower() not in terms:
            article_inclusion = False
            break
        elif match_type == 'partial_match':
          if terms == []:
            article_inclusion = False
            break
          else:
            article_inclusion = False
            for term in terms:
              if (string.lower() in term):
                article_inclusion = True
                break
    elif logical_operator == 'any':
      for string in filter_dict['string_list']:
        if terms == []:
          article_inclusion = False
          break
        elif match_type == 'full_match':
          if string.lower() in terms:
            break
        elif match_type == 'partial_match':
          for term in terms:
            if string.lower() in term:
              break
    elif logical_operator == 'none':
      for string in filter_dict['string_list']:
        if match_type == 'full_match':
          if string.lower() in terms:
            article_inclusion = False
            break
        elif match_type == 'partial_match':
          for term in terms:
            if string.lower() in term:
              article_inclusion = False
              break
    return article_inclusion

if __name__ == "__main__":
  # Creating an example multimodal dataset with patients aged 18 to 65 including images with CT scans.
  mdc = MedicalDatasetCreator()
  filter_list = [{'field': 'min_age', 'string_list': ['18']},
                 {'field': 'max_age', 'string_list': ['65']},
                 {'field': 'normalized_extractions', 'string_list': ['ct']}]
  mdc.create_dataset(dataset_name = 'ct_scan_data', filter_list = filter_list, dataset_type = 'multimodal')
