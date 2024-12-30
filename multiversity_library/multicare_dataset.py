import shutil
import os
import pyarrow.parquet as pq
import json
import pandas as pd
import ast
from datetime import datetime
import requests
from IPython.display import Image, Markdown, display
from PIL import Image as PILImage
import io

class MedicalDatasetCreator():

  def __init__(self, directory = 'MultiCaRe', master_dataset_path = '', version = 2):

    '''The MedicalDatasetCreator class is used to create datasets based on the MultiCaRe Dataset given a list of filters and a dataset type.

    - directory (str): Name of the directory that will contain the MultiCaRe Dataset and all the datasets that are created based on it.
    - master_dataset_path (str): Name of the directory that contains the main MultiCaRe Dataset. By default, f'{directory}/whole_multicare_dataset' is considered.
    '''

    ### Creation of the directory
    self.directory = directory
    if master_dataset_path:
      self.whole_dataset_path = master_dataset_path
    else:
      self.whole_dataset_path = f"{self.directory}/whole_multicare_dataset"

    for folder in [self.directory, self.whole_dataset_path]:
      if not os.path.exists(folder):
        os.makedirs(folder)

    self.version = version

    ### The MultiCaRe Dataset is downloaded and unpacked
    if os.listdir(self.whole_dataset_path) == []:
      print('Downloading the MultiCaRe Dataset from Zenodo. This may take approximately 5 minutes.')
      self._download_dataset()
    else:
      print('The MultiCaRe Dataset is already downloaded.')

    ### The dataset files are imported
    print('Importing and pre-processing the main files.')
    self.full_metadata = pq.read_table(f"{self.whole_dataset_path}/metadata.parquet").to_pydict()
    self.full_cases = pq.read_table(f"{self.whole_dataset_path}/cases.parquet").to_pydict()
    self.full_image_metadata_df = pd.read_csv(f"{self.whole_dataset_path}/captions_and_labels.csv", low_memory=False)

    ### The image metadata df is preprocessed by turning string columns with lists into list type.
    for list_column in ['case_substring', 'ml_labels_for_supervised_classification', 'gt_labels_for_semisupervised_classification']:
      self.full_image_metadata_df[list_column] = self.full_image_metadata_df[list_column].apply(ast.literal_eval)

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
    self.image_label_list = [l for l in self.full_image_metadata_df.gt_labels_for_semisupervised_classification.explode().unique() if type(l) == str]
    print('Done!')

  def _download_dataset(self):

    '''Method used to download and unzip the MultiCaRe Dataset from Zenodo. For version 1, use the multicare_v1 module from multiversity.'''

    if int(self.version) == 2:
      zenodo_id = '13936721'

    response = requests.get(f"https://zenodo.org/api/records/{zenodo_id}/files-archive", stream=True)

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

  ### create_dataset method

  def create_dataset(self, dataset_name, filter_list, dataset_type = 'multimodal', image_token = None, label_filter_column = 'both'):

    '''The create_dataset method is used to create a customized subset from the MultiCaRe Dataset.

    - dataset_name (str): Name of the subset. The subset will be created in a folder with this name inside the main directory.
    - filter_list (list): List of dictionaries with the keys below.
      - 'field': Some fields are applied at an article level ('min_year', 'max_year', 'license', 'keywords', 'mesh_terms'), some at a clinical case level ('min_age', 'max_age', 'gender', 'case_strings') and some at an image level ('caption', 'label').
      - 'string_list': The values that are searched in a given field. For min_year, max_year, min_age, and max_age, the list should include only one value (if there are many, it will pick the min or max among them depending on the case).
      - 'operator': Used for fields with multiple values in the string list. It can be either 'all' (default value), 'any' or 'none'.
      - 'match_type': Used for keywords and mesh_terms. It can be either 'full_match' or 'partial_match'.
        Example: {'field': 'keywords', 'string_list': ['diabetes'], 'match_type': 'partial_match'} will retrieve all the cases with at least one keyword that contains the substring 'diabetes'. If 'full_match' had been used, it would only retrieve cases which include the keyword 'diabetes' (exact match).
      - 'matching_case': Used for captions. It can be either True (if the casing from the search term is relevant) or False (if not).
    - dataset_type (str): Required type of dataset. It can be either 'text', 'image', 'multimodal' or 'case_series'. All the dataset types will include a readme.txt, an article_metadata.json and a reference_list.json (with citation information from case report articles).
      - text: The dataset contains a csv file with case_id, pmcid, case_text, age and gender of the patient.
      - image: The dataset contains a folder with images, and a json file with file_id, file_path, labels, caption, main_image_link (from PMC), case_id, license, split_during_preprocessing (True if the raw image included more than one sub images).
      - multimodal: The dataset contains a combination of the files frmo text and image datasets.
      - case_series: The dataset contains a folder with images (there is one folder per patient), and a csv file with cases including case_id, pmcid, case_text, age, gender, link to the case report article, amount_of_images for the specific case, and image_folder.
    - image_token (str): the token used to replace figure references mentioned in case text (e.g. if it is set to <image_token>, 'there is a mass in Figure 1' will become 'there is a mass in <image_token>'). If None (default value), the corresponding image file name is used as an image token.
    - label_filter_column (str): label column used for filters, the possible values are 'gt_labels_for_semisupervised_classification', 'ml_labels_for_supervised_classification', or 'both' (by default).

    Before creating any filter, you can check the parameters with possible values: year_list, license_list, keyword_list, mesh_term_list, age_list, gender_list and image_label_list.
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
    self.image_token = image_token
    self.label_filter_column = label_filter_column

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
      elif filter['field'] in ['caption', 'label']:
        self.image_filter_list.append(filter)
      else:
        raise ValueError("Wrong filter 'field'. Possible values are: 'min_year', 'max_year', 'license', 'keywords', 'mesh_terms', 'min_age', 'max_age', 'gender', 'case_strings', 'caption', 'label'.")

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

    ### Images are filtered according to their labels and the contents of their captions
    self.filtered_image_metadata_df = self.full_image_metadata_df.copy()

    if self.label_filter_column == 'both':
      self.filtered_image_metadata_df['filter_column'] = self.filtered_image_metadata_df['gt_labels_for_semisupervised_classification'] + self.filtered_image_metadata_df['ml_labels_for_supervised_classification']
    else:
      self.filtered_image_metadata_df['filter_column'] = self.filtered_image_metadata_df[self.label_filter_column]

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

      elif dct['field'] == 'label':

        if logical_operator == 'any':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['filter_column'].apply(lambda x: any(elem in x for elem in dct['string_list']))]

        elif logical_operator == 'none':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[~self.filtered_image_metadata_df['filter_column'].apply(lambda x: any(elem in x for elem in dct['string_list']))]

        elif logical_operator == 'all':
          self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['filter_column'].apply(lambda x: all(elem in x for elem in dct['string_list']))]

    self.filtered_image_metadata_df.drop('filter_column', axis = 1, inplace = True)
    self.filtered_image_metadata_df = self.filtered_image_metadata_df.reset_index(drop = True)
    self.filtered_image_metadata_df['file_path'] = self.filtered_image_metadata_df['file'].apply(lambda x: f"{self.directory}/{self.dataset_name}/images/{x[:4]}/{x[:5]}/{x}")

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
      if self.dataset_type == 'multimodal':
        self.case_df = self._incorporate_image_token()
      self.case_df.to_csv(f"{self.directory}/{self.dataset_name}/cases.csv", index = False)
    elif self.dataset_type == 'case_series':
      self.case_df['link'] = self.case_df['pmcid'].apply(lambda x: f"https://www.ncbi.nlm.nih.gov/pmc/articles/{x}/")
      image_counts = pd.DataFrame(self.filtered_image_metadata_df['patient_id'].value_counts()).reset_index().rename({'count': 'amount_of_images', 'patient_id': 'case_id'}, axis = 1)
      self.case_df = pd.merge(self.case_df, image_counts)
      self.case_df['amount_of_images'].fillna(0, inplace=True)
      self.case_df['image_folder'] = self.case_df.apply(lambda x: f"images/{x['case_id'][:4]}/{x['case_id'][:5]}/{x['case_id']}", axis = 1)
      self.case_df[self.case_df['amount_of_images'] == 0]['image_folder'] = 'None'
      self.case_df.to_csv(f"{self.directory}/{self.dataset_name}/cases.csv", index = False)

    ### The image_metadata.json file is created for image and multimodal dataset, and images are copied to their corresponding paths
    self.filtered_image_metadata_df = self.filtered_image_metadata_df[self.filtered_image_metadata_df['patient_id'].apply(lambda x: x.split('_')[0]).isin(common_pmcids) & self.filtered_image_metadata_df['patient_id'].isin(common_cases)]
    self.filtered_image_metadata_df.reset_index(drop = True, inplace = True)

    if self.dataset_type in ['image', 'multimodal']:
      self.filtered_image_metadata_df['split_during_preprocessing'] = self.filtered_image_metadata_df.file_path.str.contains('undivided')
      # self.filtered_image_metadata_df.rename({'label_list': 'label_list_before_postprocessing'}, axis = 1, inplace = True)
      # if not self.keep_label_columns:
      multilabel_columns = ['file_id', 'file', 'file_path', 'main_image', 'patient_id', 'license', 'file_size', 'split_during_preprocessing', 'caption',
                            'image_type', 'image_subtype', 'radiology_region', 'radiology_region_granular', 'radiology_view',
                            'ml_labels_for_supervised_classification', 'gt_labels_for_semisupervised_classification']
      self.filtered_image_metadata_df = self.filtered_image_metadata_df[multilabel_columns].copy()
      self.filtered_image_metadata_df.rename({'patient_id': 'case_id'}, axis = 1, inplace = True)

      self.filtered_image_metadata_df.to_json(f"{self.directory}/{self.dataset_name}/image_metadata.json", orient='records', lines=True)
      for file_path in self.filtered_image_metadata_df['file_path']:
        new_path = f"{file_path}"
        old_path = f"{self.whole_dataset_path}/{'/'.join(file_path.split('/')[-3:])}"
        os.makedirs(os.path.dirname(new_path), exist_ok=True)
        shutil.copy2(old_path, new_path)

    if self.dataset_type == 'case_series':
      for idx in range(len(self.filtered_image_metadata_df)):
        file_path = self.filtered_image_metadata_df['file_path'][idx]
        patient_id = self.filtered_image_metadata_df['patient_id'][idx]
        new_path = f"{file_path.replace(patient_id.split('_')[0] + '_', patient_id + '/')}"
        old_path = f"{self.whole_dataset_path}/{'/'.join(file_path.split('/')[-3:])}"
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
- MultiCaRe version: {self.version}

1. Nievas Offidani, M., & Delrieux, C. (2023). The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10079370
'''
    with open(f"{self.directory}/{self.dataset_name}/readme.txt", 'w') as file:
      file.write(readme_string)

    print(f"The {self.dataset_name} was successfully created!")

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

  def _get_image_token_dicts(self, input_data):
    '''This method is used to turn image references into replacement dictionaries.'''

    # Initialize an empty dictionary
    merged_dict = {}

    # Loop through each dictionary in the list
    for d in input_data:
      for key, value in d.items():
        # If the key is not in the dictionary, add it with a list as the initial value
        if key not in merged_dict:
          merged_dict[key] = []
        # Append the value to the list associated with the key
        merged_dict[key].append(value)

    outcome_dict = {}

    for k in merged_dict.keys():
      if self.image_token:
        outcome_dict[k] = self.image_token * len(merged_dict[k])
      else:
        outcome_dict[k] = '<' + '><'.join(merged_dict[k]) + '>'

    outcome_dict = {key: value for key, value in outcome_dict.items() if not isinstance(key, float)} # To filter out nan keys, present if the figure mention was not identified.

    return outcome_dict

  def _replace_text_with_dict(self, text, replacement_dict):
    '''This method is used to replace, in a case text, all the keys from a given dictionary by their corresponding values.'''

    if type(replacement_dict) == dict:
      for key, value in replacement_dict.items():
        text = text.replace(key, str(value))
    return text

  def _incorporate_image_token(self):
    '''This method is used to replace figure mentions by image tokens in case texts.'''

    image_token_df = self.full_image_metadata_df[['patient_id', 'case_substring', 'file']].copy()
    image_token_df['reference_dict'] = image_token_df.apply(lambda x: {substring: x['file'] for substring in x['case_substring']}, axis = 1)
    image_token_df.drop(['case_substring', 'file'], axis = 1, inplace = True)
    image_token_df = image_token_df.groupby('patient_id').agg(list).reset_index()
    image_token_df['reference_dict'] = image_token_df['reference_dict'].apply(lambda x: self._get_image_token_dicts(x))
    image_token_df.rename({'patient_id': 'case_id'}, axis = 1, inplace = True)
    self.case_df = pd.merge(self.case_df, image_token_df, on='case_id', how='left')
    self.case_df['case_text'] = self.case_df.apply(lambda x: self._replace_text_with_dict(x['case_text'], x['reference_dict']), axis = 1)
    self.case_df.drop('reference_dict', axis = 1, inplace = True)
    return self.case_df

  ### create_image_classification_dataset method

  def create_image_classification_dataset(self, dataset_dict, keep_label_columns = False, filter_column = 'both'):

    '''
    This method is used to create a dataset for image classification, by merging different subsets created by using the create_dataset method. As a result, you will get a csv containing image ids together with their paths and their corresponding label.

    dataset_dicts (dict): a dictionary with the following keys:
      - dataset_name: name of the dataset.
      - common_filter_list: filter lists that are applied to all the subsets.
      - class_subsets: a list of dictionaries, one per subset. Each dictionary should have the following keys:
        - class: name of the class.
        - filter_list: filter list that is applied to the subset.
    keep_label_columns (bool): wether to keep the original label columns from the MultiCaRe dataset or not (False by default).
    filter_column (str): label column used for filters, the possible values are 'gt_labels_for_semisupervised_classification', 'ml_labels_for_supervised_classification', or 'both' (by default).


    For example, if you want to create a dataset of CT scans with two classes ['female' and 'male'], you should provide this dictionary:

    sample_dict = {
        'dataset_name': 'gender_ct_classifier',
        'common_filter_list': [{'field': 'label', 'string_list': ['ct']}],
        'class_subsets': [
            {'class': 'female', 'filter_list': [{'field': 'gender', 'string_list': ['Female']}]},
            {'class': 'male', 'filter_list': [{'field': 'gender', 'string_list': ['Male']}]}
        ]
    }

    As a result, a csv will be created, including only images of CT scans, some of them labeled as 'male' and the rest labeled as 'female'.
    '''

    subset_dict = {}
    for class_ in dataset_dict['class_subsets']:
      subset_dict[class_['class']] = dataset_dict['common_filter_list'] + class_['filter_list']

    for k in subset_dict.keys():
      self.create_dataset(dataset_name = f"{dataset_dict['dataset_name']}_{k}", filter_list = subset_dict[k], dataset_type = 'image', image_token = None, label_filter_column = filter_column)

    dfs = []
    for k in subset_dict.keys():
      df = pd.read_json(f"{self.directory}/{dataset_dict['dataset_name']}_{k}/image_metadata.json", orient = 'records', lines = True)
      df['class'] = k
      col_list = ['file_id', 'file', 'class']
      if keep_label_columns:
        col_list += ['image_type', 'image_subtype', 'radiology_region', 'radiology_region_granular', 'radiology_view',
                     'ml_labels_for_supervised_classification', 'gt_labels_for_semisupervised_classification']
      df = df[col_list]
      dfs.append(df)

    outcome_df = pd.concat(dfs, ignore_index=True)

    outcome_df = outcome_df[~outcome_df['file'].duplicated(keep=False)]
    # outcome_df.drop('file', axis = 1, inplace = True)

    dataset_directory = f"{self.directory}/{dataset_dict['dataset_name']}"
    if not os.path.exists(dataset_directory):
      os.makedirs(dataset_directory)
      
    outcome_df['file_path'] = outcome_df.apply(lambda x: os.path.join(f"{dataset_directory}_{x['class']}", 'images', x['file'][:4], x['file'][:5], x['file']), axis = 1)

    cols = list(outcome_df.columns)
    cols.remove("file_path")
    cols.insert(2, "file_path")  # Insert 'class' as the 3rd column
    outcome_df = outcome_df[cols]

    outcome_df.to_csv(f"{dataset_directory}/{dataset_dict['dataset_name']}.csv", index = False)

    readme_string = f'''README

  This dataset is based on the MultiCaRe Dataset (1). The citation information from the case reports that were used to create this dataset can be found in the file case_report_citations.json.

  Dataset Information:
  - Name: {dataset_dict['dataset_name']}
  - Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}
  - Dict for dataset creation: {dataset_dict}
  - MultiCaRe version: {self.version}

  1. Nievas Offidani, M., & Delrieux, C. (2023). The MultiCaRe Dataset: A Multimodal Case Report Dataset with Clinical Cases, Labeled Images and Captions from Open Access PMC Articles (1.0) [Data set]. Zenodo. https://doi.org/10.5281/zenodo.10079370
  '''
    with open(f"{dataset_directory}/readme.txt", 'w') as file:
      file.write(readme_string)

  ### display_example method

  def display_example(self, image_index = 0):
    '''This method is used to display one example of the created dataset.

    image_index (int, default 0): index of the image in the created dataset.'''

    image_path = self.filtered_image_metadata_df['file_path'][image_index]
    image_labels = self.filtered_image_metadata_df['gt_labels_for_semisupervised_classification'][image_index] + self.filtered_image_metadata_df['ml_labels_for_supervised_classification'][image_index]
    image_caption = self.filtered_image_metadata_df['caption'][image_index]

    case_id = self.filtered_image_metadata_df['case_id'][image_index]

    case_row = self.case_df[self.case_df['case_id'] == case_id]
    gender = case_row['gender'].values[0]
    age = case_row['age'].values[0]
    case_text = case_row['case_text'].values[0]

    pmcid = case_id.split('_')[0]
    for r in self.reference_list:
      if r['pmcid'] == pmcid:
        citation = r
        break

    display(Markdown(f"**Case {case_id}:**"))
    display(f"Gender: {gender}")
    display(f"Age: {age}")
    display(f"Clinical Case:")
    display(case_text)
    display(Markdown(f"**Image:**"))
    # Open the image using Pillow
    img = PILImage.open(image_path)
    # Convert the image to bytes
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG') # Convert to PNG format
    img_byte_arr = img_byte_arr.getvalue()

    display(Image(data=img_byte_arr, format='png')) # Display using PNG data
    display(Markdown(f"**Image Labels:**"))
    display(Markdown(f"{image_labels}"))
    display(Markdown(f"**Image Caption:**"))
    display(Markdown(f"{image_caption}"))
    display(Markdown(f"**Citation Information:**"))
    display(citation)
