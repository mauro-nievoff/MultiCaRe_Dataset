from Bio import Entrez
from bs4 import BeautifulSoup
import requests
import json
import re
import os
from tqdm import tqdm
import warnings
import shutil
import importlib.resources as pkg_resources
import random
import pandas as pd
import numpy as np
import pyarrow.parquet as pq
import cv2
from google.colab.patches import cv2_imshow
from datetime import datetime
from PIL import Image, UnidentifiedImageError
from huggingface_hub import snapshot_download

from multiversity.multicare_dataset import MedicalDatasetCreator
from multiversity.multi_labeler import CaptionLabeler
from multiversity.multiplex_classification import MultiplexDatasetProcessor

  ########### MulticareCreator Class ###########

class MulticareCreator():

  def __init__(self, email, annotated_ngram_path = '', input_owl_path = '', api_key = '', outcome_directory = 'MultiCaRe', delete_previous_data = False, multiplex_format = False,
               max_report_amount = None, search = 'case', data_split = 20, new_dataset = False, gaussian_n = 13, low_canny = 1, high_canny = 20, edge_minimum = 100, min_size = 100, cropping_iterations = 3, image_data_split = 100):

    '''
    The MulticareCreator class is used to download the MultiCaRe dataset from scratch and to update it automatically.

    email (str): Email address used for NCBI account.
    annotated_ngram_path (str): Path to the csv file containing the annotated ngrams. By default, the file included in the multiplex package is used.
    input_owl_path (str): Path to the input taxonomy file with a Decision Rainforest format (used for Multiplex classification). By default, the file included in the multiplex package is used.
    api_key (str, '' by default): API key from NCBI (available at https://www.ncbi.nlm.nih.gov/account/settings/).
    outcome_directory (str, 'MultiCaRe' by default): Path to the folder where that will contain the dataset.
    delete_previous_data (bool, False by default): If true, any file in the main_folder will be deleted before downloading the dataset. If false, the downloading process will continue from the last checkpoint.
    multiplex_format (bool, False by default): if False, there outcome image label dataframe will include only one column with labels, otherwise there will be one column per Basic Classification Task.
    max_report_amount (int, None by default): If not None, the process will stop once this amount of case reports are downloaded.
    search (str, 'case' by default): String used in the search string in PubMed. 'case' will return all the case reports. It can be replaced by other values, such as names of diseases.
    data_split (int, 20 by default): The dataset will be split into different folders during the process, and the files will be merged at the end into one main json file.
    new_dataset (bool, False by default): If True, any saved data will be deleted before starting the process. If False, the process will start from the previous progress.
    gaussian_n (int, 13 by default): Size of the Gaussian filter used for image preprocessing.
    low_canny (int, 1 by default): Low threshold of the Canny filter used for image preprocessing.
    high_canny (int, 20 by default): High threshold of the Canny filter used for image preprocessing.
    edge_minimum (int, 100 by default): Amount of edges used to identify image borders.
    cropping_iterations (int, 3 by default): Number of iterations used in the loop for cropping.
    image_data_split (int, 100 by default): The image folder will be split into this amount of folders.
    '''

    self.email = email
    self.api_key = api_key
    if annotated_ngram_path:
      self.annotated_ngrams_path = annotated_ngram_path
    else:
      self.annotated_ngrams_path = (pkg_resources.files("multiversity.data")/ "annotated_ngrams.csv").as_posix()
    if input_owl_path:
      self.input_owl_path = input_owl_path
    else:
      self.input_owl_path = (pkg_resources.files("multiversity.data")/ "GT_MCR_TX_input_format.owx").as_posix()
    self.main_folder = outcome_directory
    if not os.path.exists(self.main_folder):
      os.makedirs(self.main_folder)

    self.delete_previous_data = delete_previous_data
    self.max_report_amount = max_report_amount
    self.search = search
    self.data_split = data_split
    self.new_dataset = new_dataset
    self.gaussian_n = gaussian_n
    self.low_canny = low_canny
    self.high_canny = high_canny
    self.edge_minimum = edge_minimum
    self.min_size = min_size
    self.cropping_iterations = cropping_iterations
    self.image_data_split = image_data_split
    self.multiplex_format = multiplex_format

    self.case_report_json_path = os.path.join(self.main_folder, 'case_report_dataset.json')
    self.caption_df_path = os.path.join(self.main_folder, 'captions_and_labels.csv')
    self.images_folder = os.path.join(self.main_folder, 'images')

    # Loop through the range of PMC numbers and create folders
    for i in range(10, 100):
      folder_name = f'PMC{str(i)[0]}/PMC{i}'
      folder_path = os.path.join(self.main_folder, folder_name)

      if not os.path.exists(folder_path):
        os.makedirs(folder_path)


  ########### Dataset Creation ###########

  def download_dataset(self, start_year = '1990', end_year = '2030', previous_pmcid_list = [], remove_temp_files = True):
    '''
    Main method used for dataset creation after class instantiation.

    start_year (str, 1990 by default): Start year of the search.
    end_year (str, 2030 by default): End year of the search.
    previous_pmcid_list (list, [] by default): List of the pmcids of the previous version of the dataset (param used for dataset update).
    remove_temp_files (bool, True by default): Wether to remove temporary files or not.
    '''

    self.start_year = start_year
    self.end_year = end_year

    # Case reports are downloaded
    self.crd = CaseReportDownloader(self.email, self.api_key, self.main_folder, self.delete_previous_data, self.max_report_amount, self.search, self.start_year, self.end_year, self.data_split)
    self.crd.create_dataset()

    if previous_pmcid_list:
      self._filter_duplicated_case_reports(previous_pmcid_list)

    # Captions are preprocessed
    self.cp = CaptionPreprocessor(self.case_report_json_path)
    self.cp.create_subcaption_dataset(csv_path = self.caption_df_path)
    self.cp.label_dataset =self._label_dataset_fix(self.cp.label_dataset)
    self.cp.label_dataset.to_csv(self.caption_df_path, index = False)

    # Images are downloaded and preprocessed
    self.ip = ImagePreprocessor(self.caption_df_path, self.images_folder, self.new_dataset, self.gaussian_n, self.low_canny, self.high_canny, self.edge_minimum, self.min_size, self.cropping_iterations, self.image_data_split)
    self.ip.create_image_dataset()

    # Files are reorganized and their contents are completed
    self._manage_dataset_files()
    self._filter_images_dataset_by_download_status()
    self._add_image_metadata()
    self._apply_caption_labeler()
    self.caption_df = self._create_caption_df()

    # The updated df is saved and temporary files are removed
    self.caption_df.to_csv(f'{self.crd.main_folder}/captions_and_labels.csv', index = False)
    if remove_temp_files:
      self._remove_temporary_files()
    self.classify_medical_images()
    self.caption_df['gt_labels_for_semisupervised_classification'] = self.caption_df.apply(lambda x: self._filter_by_incompatibilities(x), axis = 1)
    self.caption_df.to_csv(f'{self.crd.main_folder}/captions_and_labels.csv', index = False)
    print('The dataset was successfully created!')

  ### Auxiliary methods

  def _filter_duplicated_case_reports(self, previous_pmcid_list):

    '''Method used to filter out case reports that were included in the previous version of the dataset.'''

    filtered_cr_list = []
    for cr in self.crd.dataset:
      if cr['article_id'] not in previous_pmcid_list:
        filtered_cr_list.append(cr)

    with open(os.path.join(self.crd.main_folder, 'case_report_dataset.json'), 'w') as f:
      json.dump(filtered_cr_list, f)

  def _manage_dataset_files(self):

    '''Method used to turn the initial case_report_dataset.json file into four parquet files, and to reorganize dataset images.'''

    case_json = []
    metadata_json = []
    abstract_json = []
    case_images_json = []

    for c in self.crd.dataset:
      case_json.append({'article_id': c['article_id'], 'cases': [{key: value for key, value in case_.items() if key != 'case_image_list'} for case_ in c['cases']]})
      metadata_json.append({'article_id': c['article_id'], 'article_metadata': {key: value for key, value in c['article_metadata'].items() if key != 'abstract'}})
      abstract_json.append({'article_id': c['article_id'], 'abstract': c['article_metadata']['abstract']})
      case_images_json.append({'article_id': c['article_id'], 'case_images': [{key: value for key, value in case_.items() if key in ['case_id', 'case_image_list']} for case_ in c['cases']]})

    saving_dicts = [{'file': case_json, 'path': 'cases.json'},
                    {'file': metadata_json, 'path': 'metadata.json'},
                    {'file': abstract_json, 'path': 'abstracts.json'},
                    {'file': case_images_json, 'path': 'case_images.json'},]

    for d in saving_dicts:
      file_path = os.path.join(self.crd.main_folder, d['path'])

      with open(file_path, 'w') as f:
        json.dump(d['file'], f)

      df = pd.read_json(file_path)
      parquet_path = f"{self.crd.main_folder}/{d['path'].split('.')[0]}.parquet"
      df.to_parquet(parquet_path, engine='pyarrow')

    split_folder = os.path.join(self.main_folder, 'images/data_splits')
    for split in tqdm(os.listdir(split_folder)):
      split_path = os.path.join(split_folder, split, 'processed_images')
      for file_ in os.listdir(split_path):
        source_file_path = os.path.join(split_path, file_)
        destination_file_path = os.path.join(self.main_folder, file_[:4], file_[:5], file_)
        if not os.path.exists(destination_file_path):
         shutil.copy(source_file_path, destination_file_path)

  def _filter_images_dataset_by_download_status(self):

    '''Method used to get the download status of images, and to filter out from the caption df any image that was not downloaded.'''

    downloaded_files = []
    for n in tqdm(range(9)):
      folder = f'PMC{n+1}'
      subfolder_list = os.listdir(os.path.join(self.main_folder, folder))
      for subfolder in subfolder_list:
        file_list = os.listdir(os.path.join(self.main_folder, folder, subfolder))
        if file_list:
          for file_ in file_list:
            downloaded_files.append(file_)

    comparison_list = downloaded_files.copy()
    subcaption_ids = list(self.cp.label_dataset['file'])

    outcome = []
    for id in tqdm(subcaption_ids):
      status = 'not-found'
      for element in comparison_list[::-1]:
        if id == element:
          status = 'ok'
          comparison_list.remove(element)
          break
      outcome.append({'file': id, 'status': status})

    status_df = pd.DataFrame(outcome)

    self.cp.label_dataset = pd.merge(self.cp.label_dataset, status_df, on = 'file', how = 'outer')
    self.cp.label_dataset = self.cp.label_dataset[self.cp.label_dataset['status'] == 'ok']
    self.cp.label_dataset.drop('status', axis = 1, inplace = True)

  def _add_image_metadata(self):

    '''Method used to add image metadata to the caption df.'''

    metadata = pd.DataFrame(pq.read_table(f"{self.crd.main_folder}/metadata.parquet").to_pydict())
    metadata['license'] = metadata['article_metadata'].apply(lambda x: x['license'])
    metadata.drop('article_metadata', inplace = True, axis = 1)
    metadata.rename({'article_id': 'pmcid'}, axis = 1, inplace = True)
    self.cp.label_dataset['pmcid'] = self.cp.label_dataset['patient_id'].apply(lambda x: x.split('_')[0])
    self.cp.label_dataset = pd.merge(self.cp.label_dataset, metadata, on = 'pmcid', how = 'left')
    self.cp.label_dataset.drop('pmcid', axis = 1, inplace = True)
    self.cp.label_dataset.reset_index(inplace = True)
    self.cp.label_dataset['index'] = self.cp.label_dataset['index'].apply(lambda x: f"image_{('000000' + str(x))[-6:]}")
    self.cp.label_dataset.rename({'index': 'image_id'}, axis = 1, inplace = True)
    self.cp.label_dataset.to_csv(f'{self.crd.main_folder}/merged_df.csv', index = False)

  def _apply_caption_labeler(self):

    '''Method used to apply all the caption labeler methods to filter labels according to incompatibilities or dependencies.'''

    self.cl = CaptionLabeler(f'{self.crd.main_folder}/merged_df.csv', self.annotated_ngrams_path)
    self.cl.remove_label('_LABEL_BLOCKER')
    self.cl.remove_incompatibilities(group_1 = ['_assertion_absent'], group_2 = ['mass', 'benign', 'malignant'])
    self.cl.remove_label('_assertion_absent')
    anatomical_labels = ['abdomen', 'thorax', 'head', 'shoulder', 'pelvis', 'neck', 'breast', 'axial_region', 'lower_leg',
                         'lower_limb', 'knee', 'thigh', 'abdominopelvic_region', 'dental_view', 'hand', 'forearm', 'wrist', 'upper_limb', 'parasternal', 'upper_arm',
                         'foot', 'appendicular_region', 'frontal', 'hip', 'subcostal']
    radiology_labels = ['scintigraphy', 'nuclear_medicine', 'angiography', 'ct', 'contrast', 'mri', 'ultrasound', 'x_ray', 'arteriogram', 'spect', 'doppler',
                        'radiology', 'flair', 'dwi', 'pet', 't1', 'mip', 'fat_suppression', 'stir', 'venogram', 'lung_window', 'soft_tissue_window', 'without_contrast']
    self.cl.check_requirements(checked_group = anatomical_labels, required_group = radiology_labels)
    laterality_labels = ['left', 'right']
    apendicular_labels = ['lower_leg', 'lower_limb', 'knee', 'thigh', 'hand', 'forearm', 'wrist', 'upper_limb', 'upper_arm', 'foot', 'appendicular_region', 'hip']
    self.cl.check_requirements(checked_group = laterality_labels, required_group = apendicular_labels)
    mass_labels = ['mass', 'benign', 'malignant']
    self.cl.check_requirements(checked_group = mass_labels, required_group = radiology_labels)
    self.cl.caption_df.to_csv(f'{self.crd.main_folder}/labeled_df.csv', index = False)


  def _label_dataset_fix(self, df):
    for col in ['subcaption_amount', 'reference', 'subcaption_order', 'caption']:
      if col in df.columns:
        df.drop(col, axis = 1, inplace = True)
    renaming_dict = {'image_id': 'main_image', 'sub_caption': 'caption', 'subcaption_id': 'file'}
    for k in renaming_dict.keys():
      if (k in df.columns) and (renaming_dict[k] not in df.columns):
        df.rename({k: renaming_dict[k]}, axis = 1, inplace = True)
    df['patient_id'] = df['main_image'].apply(lambda x: '_'.join(x.split('_')[:2]))

    return df.copy()

  def _remove_temporary_files(self):

    '''
    Method used to remove any temporary file.
    '''

    temporary_files = [
        'data_splits',
        'case_report_dataset.json',
        'images',
        'cases.json',
        'metadata.json',
        'abstracts.json',
        'case_images.json',
        'merged_df.csv',
        'labeled_df.csv'
    ]

    for item in temporary_files:
      item_path = os.path.join(self.crd.main_folder, item)
      if os.path.isfile(item_path):
        os.remove(item_path)
      elif os.path.isdir(item_path):
        shutil.rmtree(item_path)

  def _create_caption_df(self):

    '''Method used to create the caption df with its corresponding columns and contents.'''

    # Turning the df into Multiplex format
    self.mdp = MultiplexDatasetProcessor(input_owl_path = self.input_owl_path, input_csv_path = f'{self.crd.main_folder}/labeled_df.csv', exclusion_classes=False, output_format = 'multiplex')
    self.mdp.apply_postprocessing()

    # Adding new columns with file path, size and the image component (e.g. 'a', 'undivided', etc)
    self.mdp.dataset['file_path'] = self.mdp.dataset['file'].apply(lambda x: os.path.join(f'{self.crd.main_folder}', x[:4], x[:5], x))
    self.mdp.dataset['file_size'] = self.mdp.dataset['file_path'].apply(lambda x: os.path.getsize(x))
    self.mdp.dataset['image_component'] = self.mdp.dataset['file'].apply(lambda x: x.split('_')[-3].lower())

    # Adding the corresponding text references for each image (e.g. 'Fig. 1a')
    file_df = self._get_text_references_df()
    self.mdp.dataset = pd.merge(self.mdp.dataset, file_df, on = ['main_image', 'image_component'], how = 'left')
    self.mdp.dataset.rename({'mention': 'case_substring'}, axis = 1, inplace = True)

    # Adding new columns with image id and link to the main image
    self.mdp.dataset['image_id'] = self.mdp.dataset.reset_index()['index'].apply(lambda x: f"file_{('0000000' + str(x))[-7:]}")
    self.mdp.dataset.rename({'image_id': 'file_id'}, axis = 1, inplace = True)
    self.mdp.dataset['main_image_link'] = self.mdp.dataset.main_image.apply(lambda x: f"https://www.ncbi.nlm.nih.gov/pmc/articles/{x.split('_')[0]}/bin/{'_'.join(x.split('_')[2:])}")

    # Sorting the columns

    if self.multiplex_format:
      new_column_order = ['file_id', 'file', 'file_path', 'main_image', 'image_component', 'patient_id', 'license', 'file_size', 'main_image_link',
                          'caption', 'case_substring', 'label_list', 'postprocessed_label_list',
                          'image_type', 'image_type:electrography', 'image_type:endoscopy.egd',
                          'image_type:medical_photograph', 'image_type:ophthalmic_imaging',
                          'image_type:ophthalmic_imaging.anterior_segment_image',
                          'image_type:ophthalmic_imaging.fundus_image',
                          'image_type:ophthalmic_imaging.fundus_image.ophtalmic_angiography',
                          'image_type:ophthalmic_imaging.oct', 'image_type:pathology',
                          'image_type:pathology.immunostaining',
                          'image_type:pathology.immunostaining.ihc', 'image_type:radiology.ct',
                          'image_type:radiology.mri.dwi', 'image_type:radiology.mri.ir',
                          'image_type:radiology.mri.spin_echo',
                          'image_type:radiology.nuclear_medicine',
                          'image_type:radiology~anatomical_region:body_part',
                          'image_type:radiology~anatomical_region:body_part.appendicular_region.lower_limb',
                          'image_type:radiology~anatomical_region:body_part.appendicular_region.upper_limb',
                          'image_type:radiology~anatomical_region:body_part.axial_region',
                          'image_type:radiology~anatomical_region:body_part.axial_region.torso',
                          'image_type:radiology~anatomical_region:body_part.axial_region.torso.abdominopelvic_region',
                          'image_type:radiology~anatomical_region:body_part.axial_region.torso.abdominopelvic_region.abdomen',
                          'image_type:radiology~anatomical_view:region_specific_view',
                          'image_type:radiology~anatomical_view:region_specific_view.cardiac_view.transthoracic',
                          'image_type:radiology~anatomical_view:region_specific_view.dental_view',
                          'image_type:radiology~anatomical_view:standard_view',
                          'image_type:radiology~anatomical_view:standard_view.axial',
                          'image_type:radiology~anatomical_view:standard_view.frontal',
                          'image_type:radiology~anatomical_view:standard_view.sagittal',
                          'image_type:radiology~attribute_angiography:angiography',
                          'image_type:radiology~mass_finding:mass',
                          'image_type:endoscopy(main|attribute_nbi)',
                          'image_type:radiology(attribute_myelogram|main|attribute_3d|mass_finding|anatomical_region|anatomical_view|contrast_status|attribute_mip|attribute_angiography)',
                          'image_type:radiology.mri(attribute_fat_suppression|main)',
                          'image_type:radiology.ultrasound(attribute_m_mode|main)',
                          'image_type:radiology~anatomical_region:body_part.appendicular_region(main|laterality)',
                          'image_type:radiology~anatomical_view:region_specific_view.cardiac_view(cardiac_chambers|cardiac_axis|main)']
    else:
      new_column_order = ['file_id', 'file', 'file_path', 'main_image', 'image_component', 'patient_id', 'license', 'file_size', 'main_image_link',
                          'caption', 'case_substring', 'postprocessed_label_list']

    cols = []
    for col in new_column_order:
      if col in self.mdp.dataset.columns:
        cols.append(col)

    self.mdp.dataset = self.mdp.dataset[cols]
    self.mdp.dataset.rename({'postprocessed_label_list': 'gt_labels_for_semisupervised_classification'}, axis = 1, inplace = True)

    self.mdp.dataset['case_substring'] = self.mdp.dataset['case_substring'].apply(lambda x: self._filter_out_incompatible_strings(x)) # Incompatible strings are filtered out (e.g. if an image has the strings 'Fig 1' and 'Fig 2', they are both removed).

    return self.mdp.dataset.copy()

  def _get_text_references_df(self):

    '''Method used to get the reference to figures from the contents of the text (e.g. 'Fig 1A', etc).'''

    case_images_file = pd.DataFrame(pq.read_table(f"{self.crd.main_folder}/case_images.parquet").to_pydict())
    text_references_df = pd.json_normalize(case_images_file['case_images'].apply(lambda x: [{'main_image': im['image_id'], 'text_references': im['text_references']} for c in x for im in c['case_image_list']]).explode())
    text_references_df.dropna(inplace = True)
    text_references_df = text_references_df[text_references_df['text_references'].apply(lambda x: len(x) > 0)]
    text_references_df['figure_mentions'] = text_references_df['text_references'].apply(lambda x: self._get_figure_mentions(x))
    text_references_df.drop('text_references', axis = 1, inplace = True)
    text_references_df = text_references_df.explode('figure_mentions').explode('figure_mentions')
    text_references_df.dropna(inplace = True)
    text_references_df.reset_index(drop = True, inplace = True)
    text_references_df = pd.concat([text_references_df.drop('figure_mentions', axis = 1), pd.json_normalize(text_references_df['figure_mentions'])], axis=1)
    text_references_df = text_references_df.explode('letters').reset_index(drop = True)
    text_references_df = text_references_df.fillna('undivided')
    text_references_df.rename({'letters': 'image_component'}, axis = 1, inplace = True)

    file_df = self.mdp.dataset[['main_image', 'image_component']].copy()
    file_df = pd.merge(file_df, text_references_df, on = ['main_image', 'image_component'], how = 'left')
    undivided_references_df = text_references_df[text_references_df['image_component'] == 'undivided'].drop('image_component', axis = 1).copy()
    file_df = pd.merge(file_df, undivided_references_df, on = ['main_image'], how = 'left')
    file_df = file_df.groupby(['main_image', 'image_component']).agg(list).reset_index()
    file_df['mention'] = file_df.apply(lambda x: list(set([element for lst in [x['mention_x'], x['mention_y']] for element in lst if type(element) == str])), axis = 1)
    file_df.drop(['mention_x', 'mention_y'], axis = 1, inplace = True)

    return file_df

  def _get_figure_mentions(self, input_string_list):

    '''Method used to identify any figure mentioned in a given string.'''

    outcome_list = []

    for input_string in input_string_list:

      patterns = r"(?:fig |figs |fig\.|figs\.|figure|figures)"

      s = input_string

      fig_mention_start = []

      search_end = 0
      text_search = re.search(patterns, s.lower())

      while text_search:
        fig_mention_start.append(text_search.span()[0] + search_end)
        search_end = text_search.span()[-1]
        s = s[search_end:]
        text_search = re.search(patterns, s.lower())
      if text_search:
        fig_mention_start.append(text_search.span()[0] + search_end)

      substrings = []
      for start in fig_mention_start:
        substring = input_string[start:].split(')')[0].split(']')[0]
        if any(char.isdigit() for char in ' '.join(substring.split()[:3])): # Checking if there is any digit in the substring.
          substrings.append(substring)

      substring_split = []
      for substring in substrings:
        char_type_split = re.findall(r'\d+|[a-zA-Z]+|[^a-zA-Z\d]+', substring.lower())
        max_considered_split = 20
        char_type_split = [s for s in char_type_split[:max_considered_split] if s.isalnum()]
        substring_split.append(char_type_split)

      letters = []
      for ss in substring_split:
        ss_letters = []
        digit_condition = False
        for p in ss:
          if digit_condition:
            if p.isalpha() and len(p) == 1:
              ss_letters.append(p)
            elif p in ['and', 'to']:
              pass
            else:
              break
          if p.isdigit():
            digit_condition = True
        letters.append(ss_letters)

      letter_list = []
      for lst in letters:
        if lst:
          letter_list.append([chr(i) for i in range(ord(lst[0]), ord(lst[-1]) + 1)])
        else:
          letter_list.append([])

      fig_mentions = []

      for i, substring in enumerate(substrings):
        mention_start = input_string.find(substring)
        char_type_list = re.findall(r'\d+|[a-zA-Z]+|[^a-zA-Z\d]+', input_string[mention_start:].lower())
        new_substring = ''
        if letter_list[i]:
          last_figure_letter = letter_list[i][-1]
          for chars in char_type_list:
            if chars != last_figure_letter:
              new_substring += chars
            else:
              new_substring += chars
              break
        else:
          for chars in char_type_list:
            if chars.isdigit():
              new_substring += chars
              break
            else:
              new_substring += chars
        mention_end = mention_start + len(new_substring)
        fig_mentions.append({'mention': input_string[mention_start:mention_end], 'letters': letter_list[i]})
      outcome_list.append(fig_mentions)
    return outcome_list

  def _filter_out_incompatible_strings(self, string_list):

    '''Method used to remove strings from a list if there is any incompatibility (e.g. 'Fig 1' and 'Fig 2' cannot be assigned to the same image).'''

    conditions = []
    if len(string_list) > 1:
      fig_numbers = set(re.findall(r'\d+', string_list[0]))
      number_condition = False
      for number in fig_numbers:
        for string in string_list[1:]:
          if number in set(re.findall(r'\d+', string)):
            number_condition = True
          else:
            number_condition = False
            break
        conditions.append(number_condition)
    else:
      conditions.append(True)
    if (len(conditions) > 0) and (any(conditions)):
      return string_list
    else:
      return []


  def classify_medical_images(self, df_path = ''):

    '''Method used to add to a dataframe the predictions from the MultiCaReClassifier.
    df_path (str): Path to the dataframe with image paths ('file_path' column). If it is not provided, then the path to self.caption_df is used by default.
    '''

    if df_path:
      self.caption_df = pd.read_csv(df_path)
    else:
      df_path = os.path.join(self.main_folder, 'captions_and_labels.csv')

    snapshot_download(repo_id='mauro-nievoff/MultiCaReClassifier',local_dir=os.getcwd())

    from pipeline import MultiCaReClassifier

    ml_cols = ['ml_labels_for_supervised_classification', 'image_type', 'image_subtype', 'radiology_region', 'radiology_region_granular', 'radiology_view']
    for col in ml_cols:
      if col not in self.caption_df.columns:
        self.caption_df[col] = np.nan

    self.temp_folder = os.path.join(self.main_folder, 'temp_folder')

    while True:
      input_paths = self.caption_df[self.caption_df['ml_labels_for_supervised_classification'].isna()]['file_path'][:100].tolist()
      if len(input_paths) == 0:
        if os.path.exists(self.temp_folder):
          shutil.rmtree(self.temp_folder)
        break
      else:
        if os.path.exists(self.temp_folder):
          shutil.rmtree(self.temp_folder)
        os.makedirs(self.temp_folder, exist_ok=True)

        for image_path in input_paths:
          shutil.copy(image_path, self.temp_folder)

        predictions = MultiCaReClassifier(image_folder = self.temp_folder, models_root = 'MultiCaReClassifier/models', add_multiclass_columns = True)
        predictions.data.rename({'image_path': 'file_path', 'label_list': 'ml_labels_for_supervised_classification'}, axis = 1, inplace = True)
        predictions.data['file'] = predictions.data['file_path'].apply(lambda x: str(x).split('/')[-1])
        predictions.data.drop('file_path', axis = 1, inplace = True)

        self.caption_df = pd.merge(self.caption_df, predictions.data, on='file', how='left')

        for col in ml_cols:
          self.caption_df[col] = self.caption_df[col + '_x'].combine_first(self.caption_df[col + '_y'])
          self.caption_df.drop([col + '_x', col + '_y'], axis = 1, inplace = True)
        self.caption_df.to_csv(os.path.join(self.main_folder, 'captions_and_labels.csv'), index = False)

  def _filter_by_incompatibilities(self, row):

    '''Method used to filter out ground truth label lists that have any incompatibility with a ML label.
    '''

    image_types = ['chart', 'radiology', 'pathology', 'medical_photograph', 'ophthalmic_imaging', 'endoscopy', 'electrography']
    image_subtypes = ['chart',
                      'ct', 'mri', 'x_ray', 'pet', 'spect', 'scintigraphy', 'ultrasound', 'tractography',
                      'acid_fast', 'alcian_blue', 'congo_red', 'fish', 'giemsa', 'gram', 'h&e', 'immunostaining', 'masson_trichrome', 'methenamine_silver', 'methylene_blue', 'papanicolaou', 'pas', 'van_gieson',
                      'skin_photograph', 'oral_photograph', 'other_medical_photograph',
                      'b_scan', 'autofluorescence', 'fundus_photograph', 'gonioscopy', 'oct', 'ophthalmic_angiography', 'slit_lamp_photograph',
                      'gi_endoscopy', 'airway_endoscopy', 'other_endoscopy', 'arthroscopy',
                      'eeg', 'emg', 'ekg']
    anatomical_regions = ['abdomen', 'breast', 'head', 'neck', 'pelvis', 'thorax',
                          'lower_limb', 'upper_limb', 'whole_body']
    granular_anatomical_regions = ['abdomen', 'breast', 'head', 'neck', 'pelvis', 'thorax',
                                  'ankle', 'foot', 'hip', 'knee', 'lower_leg', 'thigh',
                                  'elbow', 'forearm', 'hand', 'shoulder', 'upper_arm', 'wrist',
                                  'whole_body']
    anatomical_views = ['axial',  'frontal', 'sagittal', 'oblique', 'occlusal', 'panoramic', 'periapical', 'intravascular', 'transabdominal', 'transesophageal', 'transvaginal', 'transthoracic']

    incompatible_lists = [image_types,
                          image_subtypes,
                          anatomical_regions,
                          granular_anatomical_regions,
                          anatomical_views]
    condition = True

    labels = set(row['ml_labels_for_supervised_classification'] + row['gt_labels_for_semisupervised_classification'])

    for incompatibility_list in incompatible_lists:
      if len(labels.intersection(set(incompatibility_list))) > 1:
        condition = False
        break

    if condition:
      return row['gt_labels_for_semisupervised_classification']
    else:
      return []

  ########### Dataset Update ###########

  def update_dataset(self, previous_version_directory = '', negative_year_correction = 0):

    '''Method used to update the dataset, based on a previous version. If a previous version directory is not provided, the lastest version available on Zenodo will be downloaded and it will be considered to be the previous version.
    previous_version_directory (str): path to the directory containing the previous version of the dataset. If an empty string is provided (default value), then the previous version is downloaded from Zenodo.
    negative_year_correction (int): negative numb
    '''


    # Previous data is imported
    if previous_version_directory != '':
      self.previous_version_directory = previous_version_directory
    else:
      self.previous_version_directory = os.path.join(self.main_folder, 'previous_version_dataset')

    self.mdc = MedicalDatasetCreator(directory = 'MultiCaRe', master_dataset_path = self.previous_version_directory)

    self.pmcid_list = self.mdc.full_metadata['article_id']
    self.start_year = str(int(sorted(self.mdc.year_list)[-1]) + negative_year_correction)
    self.end_year = str(int(datetime.now().date().strftime("%Y")) + 1)

    # The new data is downloaded
    self.download_dataset(start_year = self.start_year, end_year = self.end_year, previous_pmcid_list = self.pmcid_list, remove_temp_files = True)

    # Files are updated
    self._update_parquet_files()

    previous_caption_df = pd.read_csv(f'{self.previous_version_directory}/captions_and_labels.csv')

    self.caption_df = pd.concat([previous_caption_df, self.caption_df], ignore_index=True)
    self.caption_df['new_path'] = self.caption_df['file'].apply(lambda x: os.path.join(f'{self.crd.main_folder}', x[:4], x[:5], x).replace('.jpg', '.webp'))

    tqdm.pandas()

    self.caption_df.progress_apply(lambda x: self._update_images(x), axis = 1)

    self.caption_df.drop('file_path', axis = 1, inplace = True)
    self.caption_df.rename({'new_path': 'file_path'}, axis = 1, inplace = True)
    if self.caption_df.columns[-1] == 'file_path':
      self.caption_df = self.caption_df[list(self.caption_df.columns[:2]) + ['file_path'] + list(self.caption_df.columns[2:-1])]

    # The final file is exported
    self.caption_df['file_id'] = self.caption_df.reset_index()['index'].apply(lambda x: f"file_{('0000000' + str(x))[-7:]}")
    self.caption_df['file'] = self.caption_df['file'].str.replace('.jpg', '.webp')
    self.caption_df.to_csv(f'{self.main_folder}/captions_and_labels.csv', index = False)

  ### Auxiliary Methods

  def _update_parquet_files(self):

    '''Method used to merge the previous data with the new data and create the corresponding parquet files.'''

    for file_name in ['abstracts', 'metadata', 'case_images', 'cases']:
      v1_file = pq.read_table(f"{self.previous_version_directory}/{file_name}.parquet").to_pydict()
      update_file = pq.read_table(f"{self.main_folder}/{file_name}.parquet").to_pydict()
      v2_file = self._merge_dicts(v1_file, update_file)
      self._save_parquet(v2_file, file_name = file_name, folder = self.main_folder)

  def _merge_dicts(self, previous_dict, dict_update):

    '''Method used to merge dicts.'''

    merged_dict = {key: previous_dict.get(key, []) + dict_update.get(key, []) for key in previous_dict.keys() | dict_update.keys()}
    return merged_dict

  def _save_parquet(self, dct, file_name, folder):

    '''Method used to save a dict as a parquet file.'''

    df = pd.DataFrame(dct)
    parquet_file = f"{folder}/{file_name}.parquet"
    df.to_parquet(parquet_file, engine='pyarrow')

  def _update_images(self, row):

    '''Method used to update the images.'''

    subfolders = ['PMC' + str(n) for n in range(1, 100)]

    for folder in subfolders:
        parent_folder = folder[:4] if int(folder[3:]) >= 10 else ''
        folder_path = os.path.join(self.crd.main_folder, parent_folder, folder)
        os.makedirs(folder_path, exist_ok=True)

    if not os.path.exists(row['new_path']):
      try:
        old_path = row['file_path']
        with Image.open(old_path) as img:
          img.save(row['new_path'], 'webp')
      except UnidentifiedImageError:
        blank_image = Image.new('RGB', (1, 1), color=(255, 255, 255))  #  If there is an image error for some reason (very infrequent), a white image is saved instead.
        blank_image.save(row['new_path'], 'webp')

  ########### CaseReportDownloader Class ###########

class CaseReportDownloader():

  def __init__(self, email, api_key = '', main_folder = '/content/drive/MyDrive/MultiCaRe_dataset', delete_previous_data = False,
               max_report_amount = None, search = 'case', start_year = '1990', end_year = '2030', data_split = 20):

    '''
    CaseReportDownloader is a class used to download open access case reports from PubMed Central.
    email (str): email address used for NCBI account.
    api_key (str): API key from NCBI (available at https://www.ncbi.nlm.nih.gov/account/settings/).
    main_folder (str): path to the folder where that will contain the dataset.
    delete_previous_data (bool): If true, any file in the main_folder will be deleted before downloading the dataset. If false, the downloading process will continue from the last checkpoint.
    max_report_amount (int): If not None, the process will stop once this amount of case reports are downloaded.
    search (str): String used in the search string in PubMed. 'case' will return all the case reports. It can be replaced by other values, such as names of diseases.
    start_year (str): The dataset will not include case reports that were published earlier than this start year.
    end_year (str): The dataset will not include case reports that were published later than this end year.
    data_split (int): The dataset will be split into different folders during the process, and the files will be merged at the end into one main json file.
    '''

    ## Creating the directory

    self.main_folder = main_folder
    self.data_split = data_split

    if os.path.exists(self.main_folder):
      if delete_previous_data:
        for filename in os.listdir(self.main_folder):
          file_path = os.path.join(self.main_folder, filename)
          if os.path.isfile(file_path):
            os.remove(file_path)
          elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
        print('Previous data was deleted successfully.')
    else:
      os.makedirs(self.main_folder)

    if isinstance(self.data_split, int) and data_split > 0:
      self.partial_data_folders = []
      for s in range(self.data_split):
        split_path = f"{self.main_folder}/data_splits/split_{s+1}_of_{self.data_split}"
        if not os.path.exists(split_path):
          os.makedirs(split_path)
          os.makedirs(f"{split_path}/dataset_jsons")
          os.makedirs(f"{split_path}/other_jsons")
        self.partial_data_folders.append(split_path)
    else:
      raise TypeError("data_split must be a positive integer.")

    ## Setting up Entrez

    if email is None:
      raise TypeError("Email is not set. You should use the email from your NCBI account (www.ncbi.nlm.nih.gov).")
    else:
      Entrez.email = email
    if api_key:
      Entrez.api_key = api_key
    else:
      warnings.warn("api_key was not set. You can get your API key at https://www.ncbi.nlm.nih.gov/account/settings/.")

    ## Creating search params

    self.search_string = self._create_search_string(search)
    self.start_year = start_year
    self.end_year = end_year
    self.max_report_amount = max_report_amount

  ### Methods for dataset creation

  def create_dataset(self):

    '''Method used to create the dataset.
    If the process is stopped at any time, the following time it will recover the progress that had been made (it doesn't start from scratch every time).'''

    pmid_list_path = f'{self.main_folder}/pmid_list.json'
    if os.path.exists(pmid_list_path):
      print('The PMID list was found in the dataset directory.')
      with open(pmid_list_path, 'r') as f:
        self.pmid_list = json.load(f)
    else:
      print('Getting article PMIDs.')
      self.pmid_list = self._get_pmids()
      with open(pmid_list_path, 'w') as f:
        json.dump(self.pmid_list, f)

    print(f'A total of {len(self.pmid_list)} relevant article IDs were found.')

    split_len = (len(self.pmid_list) // self.data_split)
    start_index = 0
    end_index = split_len
    for folder in self.partial_data_folders:
      if end_index > len(self.pmid_list):
        end_index = len(self.pmid_list)
      pmid_split = self.pmid_list[start_index:end_index]
      pmid_split_path = f"{folder}/partial_pmid.json"
      if not os.path.exists(pmid_split_path):
        with open(pmid_split_path, 'w') as f:
          json.dump(pmid_split, f)
      start_index = end_index
      end_index = end_index + split_len

    case_report_amount = 0
    for folder in self.partial_data_folders:
      if not os.path.exists(f"{folder}/partial_dataset.json"):
        with open(f"{folder}/partial_pmid.json", 'r') as f:
          pmid_split = json.load(f)
        folder_pmid_amount = len(pmid_split)
        dataset_jsons = [j for j in os.listdir(f"{folder}/dataset_jsons") if (j[-4:] == 'json')]
        other_jsons = [j for j in os.listdir(f"{folder}/other_jsons") if (j[-4:] == 'json')]
        processed_jsons = (len(dataset_jsons) + len(other_jsons))
        remaining_jsons = folder_pmid_amount - processed_jsons
        if remaining_jsons:
          self._download_data(pmid_split[processed_jsons:], folder, case_report_amount)
        partial_json = []
        for j in os.listdir(f"{folder}/dataset_jsons"):
          with open(f"{folder}/dataset_jsons/{j}", 'r') as f:
            case_ = json.load(f)
          partial_json.append(case_)
        with open(f"{folder}/partial_dataset.json", "w") as f:
          json.dump(partial_json, f)
        case_report_amount += len([j for j in os.listdir(f"{folder}/dataset_jsons") if (j[-4:] == 'json')])
        if (self.max_report_amount) and (case_report_amount >= self.max_report_amount):
          break
      else:
        case_report_amount += len([j for j in os.listdir(f"{folder}/dataset_jsons") if (j[-4:] == 'json')])

    main_dataset_path = f"{self.main_folder}/case_report_dataset.json"
    if not os.path.exists(main_dataset_path):
      self.dataset = []
      for folder in self.partial_data_folders:
        if os.path.exists(f"{folder}/partial_dataset.json"):
          with open(f"{folder}/partial_dataset.json", 'r') as f:
            partial_dataset = json.load(f)
          for case_ in partial_dataset:
            self.dataset.append(case_)
      with open(main_dataset_path, 'w') as f:
        json.dump(self.dataset, f)
    else:
      with open(main_dataset_path, 'r') as f:
        self.dataset = json.load(f)

    print(f'Case report dataset creation is complete, including data from {len(self.dataset)} case reports.')

  def _download_data(self, pmid_list, partial_data_folder, case_report_amount):

    '''This method is used to create a json for each PMID in pmid_list, by using the rest of the methods included in this class.
    The name of the json file has the following format: f'{pmcid}_{pmid}_{status_string}.json'.
    The PMID is always present. If there is no PMCID for a certain PMID, then 'x' is used as PMCID.
    The status_string is either 'ok', 'er' (if there was an error when using the API) or 'empty' (if the article doesn't include cases).'''

    print(f"Downloading data from data split number {partial_data_folder.split('/')[-1].split('_')[1]} out of {partial_data_folder.split('/')[-1].split('_')[-1]}.")
    for pmid in tqdm(pmid_list):
      if (self.max_report_amount) and (case_report_amount >= self.max_report_amount):
        break
      else:
        id_mapping = self._get_id_mapping(pmid)
        pmcid = 'x'
        if 'pmcid' in id_mapping.keys():
          pmcid = id_mapping['pmcid']

        metadata = None
        case_report = None
        status_string = 'ok'
        outcome_json = {'pmcid': pmcid}

        if pmcid != 'x':
          try:
            metadata = self._get_article_metadata(id_mapping)
          except:
            status_string = 'er'

          if metadata:
            try:
              cr = self._get_case_report(pmcid)
              if cr:
                case_report = cr
              else:
                status_string = 'er'
            except:
              status_string = 'er'

            if case_report:
              outcome_json = self._combine_jsons(metadata, case_report)
              if outcome_json['article_metadata']['case_amount'] == 0:
                status_string = 'empty'
              else:
                self._extract_demographics(outcome_json)
                for c in outcome_json['cases']:
                  c['figs_in_text'] = self._get_fig_numbers(c['case_text'])
                self._assign_images(outcome_json)
                for c in outcome_json['cases']:
                  for l in c['case_image_list']:
                    l['image_id']  = f"{c['case_id']}_{l['file']}"
                  self._assign_text_references(c)
                outcome_json.pop('captions')
                for c in outcome_json['cases']:
                  if 'figs_in_text' in c.keys():
                    c.pop('figs_in_text')
                  for l in c['case_image_list']:
                    if 'caption_order' in l.keys():
                      l.pop('caption_order')
                case_report_amount += 1
        else:
          status_string = 'er'

        if status_string == 'ok':
          json_folder = f"{partial_data_folder}/dataset_jsons"
        else:
          json_folder = f"{partial_data_folder}/other_jsons"

        file_path = f"{json_folder}/{pmcid}_{pmid}_{status_string}.json"
        with open(file_path, 'w') as f:
          json.dump(outcome_json, f)

  ### Auxiliary Methods

  def _create_search_string(self, search):

    '''The search string contains all the combinations of search terms and filters to try to retrieve as many case reports as possible.'''

    cr_filter_search_string = f"({search}[All Fields] AND case reports[Publication Type] NOT animal[filter])"

    case_synonyms = ['case study', 'case studies', 'case series', 'case report', 'case reports', 'clinical case', 'clinical cases', 'case presentation', 'case presentations']
    case_search_string = '('
    for idx, synonym in enumerate(case_synonyms):
      case_search_string += synonym + '[Title/Abstract]'
      if idx != len(case_synonyms) -1:
        case_search_string += ' OR '
      else:
        case_search_string += ')'

    cr_term_search_string = f"(({search}[All Fields]) AND {case_search_string} NOT case reports[Publication Type] NOT animal[filter])"

    search_string = f"({cr_filter_search_string} OR {cr_term_search_string}) AND ffrft[Filter]"

    return search_string

  def _get_unique_in_order(self, original_list):

    '''Method used to remove duplicates from a list without affecting the order of the elements.'''

    unique_list = []
    for item in original_list:
      if item not in unique_list:
        unique_list.append(item)
    return unique_list

  ########### Getting PMID List ###########

  def _get_pmids(self):

    '''This method is used to get the list of PMIDs returned by PubMed when querying using the search string.
    As there is a limitation in the API (max 10k results per query), the whole search is divided into multiple searches by splitting the temporal range.
    The full temporal range is first divided into years. If that year contains more than 10k articles, then it is divided into months. Those months are also eventually divided into days.
    Days cannot be split, so if a day contains more than 10k resuts, all the results after the 10kth are lost. This happens in a few dates (January 1st only in specific years).'''

    # dict used to map months to their lenghts.
    mon_len_dic = {'01': '31', '02': '28', '03': '31', '04': '30',
                  '05': '31', '06': '30', '07': '31', '08': '31',
                  '09': '30', '10': '31', '11': '30', '12': '31'}

    pmid_list = []
    api_limitation_string = ''

    for year in tqdm(range(int(self.start_year), int(self.end_year)+1)):
      date_string = str(year)
      if self._get_result_amount(self.search_string, year) < 9999:
        pmid_list += self._get_pmid_list(self.search_string, date_string)
      else:
        for month in range(12):
          month = ('0'+str(month+1))[-2:]
          date_string = date_string + '/' + month
          if self._get_result_amount(self.search_string, year, month) < 9999:
            pmid_list += self._get_pmid_list(self.search_string, date_string)
          else:
            mon_len = mon_len_dic[month]
            for day in range(int(mon_len)):
              day = ('0'+str(day+1))[-2:]
              date_string = date_string+'/'+day
              result_amount = self._get_result_amount(self.search_string, year, month, day)
              pmid_list += self._get_pmid_list(self.search_string, date_string)
              if result_amount > 9999:
                if api_limitation_string:
                  api_limitation_string += ", "
                api_limitation_string += f"{year}.{month}.{day}"
    if api_limitation_string:
      print(f'Due to API limitation, only 10k PMIDs were extracted for the following dates: {api_limitation_string}')

    pmid_list = list(set(pmid_list))
    random.shuffle(pmid_list) # The final PMID list is sorted randomly to avoid using always earlier articles first.
    return pmid_list

  def _get_result_amount(self, search, year, month = None, day = None):

    '''Method used to get the amount of articles returned by a particular query.'''

    date_string = str(year)
    if month:
      date_string += f'/{str(month)}'
    if day:
      date_string += f'/{str(day)}'
    search_string = search + " AND " + date_string +"[Date - Publication]"
    handle = Entrez.esearch(db="pubmed", term = search_string)
    record = Entrez.read(handle)
    return int(record['Count'])

  def _get_pmid_list(self, search_string, date_string):

    '''Method used to get the PMID list corresponding to a particular search string and a date string.'''

    query = search_string + ' AND ' + date_string + '[Date - Publication]'
    handle = Entrez.esearch(db="pubmed", term=query, retmode="xml", retmax= 10000)
    record = Entrez.read(handle)
    return record["IdList"]

  #### Methods to Map PMIDs to PMCIDs

  def _get_id_mapping(self, pmid):

    '''Method used to get the PMCID of an article given its PMID (PMC's API requires PMCIDs instead of PMIDs).
    It was decided to get PMIDs and then map them to the corresponding PMCIDs instead of searching for the PMCIDs using PMC's API because the search engine from PubMed is more suitable for the task than the one from PMC.'''

    try:
      pmid_handle = Entrez.efetch(db="pubmed", id=pmid, rettype="xml")
      pmid_record = Entrez.read(pmid_handle)
      mapping = {e.attributes['IdType'].replace('pubmed', 'pmid').replace('pmc', 'pmcid'): str(e) for e in pmid_record['PubmedArticle'][0]['PubmedData']['ArticleIdList']}
      return mapping
    except:
      return {"pmid": pmid, "comment": 'Mapping error.'}

  ### Methods to Get Article Metadata

  def _get_article_metadata(self, id_dict):

    '''Method used to get the article metadata, including all the data points for citation.'''

    article = id_dict['pmcid']
    url = f"https://pubmed.ncbi.nlm.nih.gov/?term={article}&format=abstract&size=10"
    response = requests.get(url)
    article_soup = BeautifulSoup(response.content, 'html.parser')
    article_dict = {}
    article_dict['title'] = article_soup.find("h1", class_="heading-title").text.replace('\n', '').strip()
    article_dict['authors'] = self._get_authors(article_soup)
    article_dict['journal'] = article_soup.find("div", class_="journal-actions dropdown-block").text.split("\n")[1].strip()
    article_dict['journal_detail'] = article_soup.find("span", class_="cit").text
    article_dict['year'] = article_dict['journal_detail'][:4]
    article_dict['doi'] = id_dict.get('doi')
    article_dict['pmid'] = id_dict.get('pmid')
    article_dict['pmcid'] = id_dict.get('pmcid')
    mesh_term_dict = self._get_mesh_terms(article_soup)
    article_dict['mesh_terms'] = mesh_term_dict['mesh_terms']
    article_dict['major_mesh_terms'] = mesh_term_dict['major_mesh_terms']
    try:
      article_dict['keywords'] = self._get_keywords(article_soup)
    except:
      article_dict['keywords'] = None
    try:
      article_dict['abstract'] = re.sub(' +', ' ', article_soup.find('div', class_="abstract-content selected").text.replace(' \n ', '').replace('\n\n\n', '').strip()).replace('\n ', '\n')
    except:
      article_dict['abstract'] = None
    article_dict['link'] = article_soup.find('meta', {'name': 'citation_abstract_html_url'})['content']
    return article_dict

  def _get_authors(self, article_soup):

    '''Method used to get the authors from the article metadata.'''

    names = []
    divs = article_soup.find_all('div', {'class': 'authors-list'})
    for div in divs:
      name_list = div.find_all('a', {'class': 'full-name'})
      for name in name_list:
        names.append(name.text)
    return self._get_unique_in_order(names)

  def _get_mesh_terms(self, article_soup):

    '''Method used to get the mesh terms from the article metadata.'''

    mesh_dict = {}
    mesh_dict['major_mesh_terms'] = [term.text.replace('\n', '').strip().replace('*', '') for term in article_soup.find_all('button', class_='keyword-actions-trigger trigger keyword-link major')]
    mesh_dict['minor_mesh_terms'] = [term.text.replace('\n', '').strip() for term in article_soup.find_all('button', class_='keyword-actions-trigger trigger keyword-link')]
    mesh_dict['mesh_terms'] = mesh_dict['major_mesh_terms'] + mesh_dict['minor_mesh_terms']
    return mesh_dict

  def _get_keywords(self, article_soup):

    '''Method used to get the keywords from the article metadata.'''

    tags = article_soup.find_all('p')
    keywords = None
    for tag in tags:
      strong_tag = tag.find('strong', {'class': 'sub-title'})
      if strong_tag and strong_tag.text.strip() == 'Keywords:':
        keywords = [kw.strip().lower() for kw in tag.text.split(':')[1].replace('\n', '').replace('.', '').split(';')]
    return keywords

  ### Methods used to Get the Case Reports

  def _get_case_report(self, article_id):

    '''Method used to get the text from the article content given an article id.'''

    url = f"https://www.ncbi.nlm.nih.gov/research/bionlp/RESTful/pmcoa.cgi/BioC_xml/{article_id}/ascii"
    r = requests.get(url)
    if str(r) != '<Response [200]>':
      pass
    else:
      soup = BeautifulSoup(r.content, 'xml')
      article_dict = {}
      article_dict['pmcid'] = 'PMC' + soup.find('id').text
      article_dict['title'] = soup.find('infon', {'key': 'section_type'}, string = 'TITLE').parent.find('text').text
      article_dict['cases'] = self._get_cases(soup)
      article_dict['cases'] = self._explode_list(article_dict['cases'])
      article_dict['cases'] = self._merge_paragraphs(article_dict['cases'])
      article_dict['cases'] = self._reduce_noise_by_title(title = article_dict['title'], cases = article_dict['cases'])
      article_dict['case_amount'] = len(article_dict['cases'])
      article_dict['caption_dicts'] = [{'tag': caption_soup.parent.find('infon', {'key':'id'}).text,
                                        'caption': caption_soup.parent.find('text').text,
                                        'file': caption_soup.parent.find('infon', {'key': 'file'}).text,
                                        'caption_order': i+1} for i, caption_soup in enumerate(soup.find_all('infon', {'key': 'type'}, string = re.compile(r'fig_')))]
      article_dict['caption_dicts'] = self._merge_captions(article_dict['caption_dicts'])
      article_dict['caption_dicts'] = self._reduce_caption_noise(cases = article_dict['cases'], caption_dicts = article_dict['caption_dicts'])
      article_dict['license'] = soup.find('infon', {'key': 'license'}).text
      try:
        article_dict['keywords'] = soup.find('infon', {'key': 'kwd'}).text.split(', ')
      except:
        article_dict['keywords'] = None
      return article_dict

  def _merge_paragraphs(self, paragraph_list):

    '''Method used to merge paragraphs from the article content when they refer to the same case.
    To do so, all consecutive paragraphs are considered to belong to the same case until a certain age or gender is mentioned, because those mentions are usually included right at the beginning of the cases.'''

    cases = []
    new_case = []
    age_pattern = r'[\s-](year|yr|month|day)s?(\s?-+|-?\s+)old'
    gender_pattern = r' (man|woman|male|female|gentleman|lady|boy|girl|child|baby|patient)( |,)'

    ## Defining split pattern to use (if age is mentioned in text, gender pattern should be ignored).

    n = 0
    for paragraph in paragraph_list:
      age_mention = re.search(age_pattern, paragraph.lower())
      if (age_mention is not None):
        n += 1

    if n > 0:
      split_pattern = age_pattern
    else:
      split_pattern = gender_pattern

    for idx, paragraph in enumerate(paragraph_list):
      relevant_mention = re.search(split_pattern, paragraph.lower())
      if (new_case != []) and (relevant_mention is not None):
        cases.append('\n'.join(new_case))
        new_case = [paragraph]
      elif (new_case == []) and (relevant_mention is not None):
        new_case.append(paragraph)
      elif (new_case != []) and (relevant_mention is None):
        new_case.append(paragraph)
      if (idx == len(paragraph_list) -1) and new_case:
        cases.append('\n'.join(new_case))
    return cases

  def _get_cases(self, soup):

    '''Method used to get the cases from the article content.
    Such case texts can be included in specific 'CASE' sections (option 1), in other sections with 'case' in the header (option 2), or in other paragraphs which include age mentions (option 3).'''

    ### Option 1: Articles with specific section type ('CASE'):

    paragraphs = [case_section.parent.find('text').text for case_section in soup.find_all('infon', {'key': 'section_type'}, string = 'CASE') if (case_section.find('infon', {'key': 'type'}, string = 'paragraph')) or (case_section.parent.find('infon', {'key': 'type'}, string = 'paragraph'))]
    if paragraphs:
      return self._merge_paragraphs(paragraphs)
    elif paragraphs == []:

      ### Option 2: Articles with titles or subtitles including the word 'case':
      case_titles = [case_section for case_section in soup.find_all('infon', {'key': 'type'}, string = re.compile(r'(?i)title')) if case_section.parent.find('text', string = re.compile(r'(?i)case'))]
      cases = []
      if case_titles:
        for title in case_titles:
          title_case = []
          next_sibling = title.parent
          while True:
            next_sibling = next_sibling.find_next_sibling("passage")
            paragraphs = next_sibling.find_all("infon", {'key': 'type'}, string = 'paragraph')
            title = next_sibling.find_all('infon', {'key': 'type'}, string = re.compile(r'(?i)title'))
            if paragraphs:
              title_case.append('\n'.join([p.parent.find('text').text for p in paragraphs]))
            elif title:
              cases.append('\n'.join(title_case))
              title_case = []
              break
        if title_case:
          cases.append('\n'.join(title_case))
        if cases == ['']:
          cases = []

      if cases:
        return cases
      else:

        ### Option 3: Articles with paragraphs that mention ages.
        age_pattern = r'[\s-](year|yr|month|day)s?[\s-]old'
        sections = [s.parent.find('infon', {'key': 'section_type'}).text for s in soup.find_all('infon', {'key': 'type'}, string = 'paragraph') if s.parent.find('text', string = re.compile(age_pattern))]
        try:
          case_section = [s for s in sections if s.lower() != 'abstract'][0]
          paragraphs = [s.parent.find('text').text for s in soup.find_all('infon', {'key': 'type'}, string = 'paragraph') if s.parent.find('infon', {'key': 'section_type'}, string = case_section)]
          return self._merge_paragraphs(paragraphs)
        except:
          return []

  def _explode_list(self, lst):

    '''Method used to explode a list of lists into a list.'''

    new_list = []
    for element in lst:
      if type(element) != list:
        new_list.append(element)
      else:
        for e in element:
          new_list.append(e)
    return new_list

  def _merge_captions(self, caption_dicts):

    '''Method used to merge caption dicts that refer to the same image.'''

    file_list = []
    for dct in caption_dicts:
      file_list.append(dct['file'])
    file_list = self._get_unique_in_order(file_list)

    new_dct_list = []
    for file_ in file_list:
      new_dict = {'tag': [t['tag'] for t in caption_dicts if t['file'] == file_][0],
                  'caption': '. '.join([t['caption'] for t in caption_dicts if t['file'] == file_]).strip(),
                  'file': file_,
                  'caption_order': [t['caption_order'] for t in caption_dicts if t['file'] == file_][0]}
      new_dct_list.append(new_dict)

    ### Fixing caption_order after merging captions:

    order_numbers = []
    for dct in new_dct_list:
      order_numbers.append(dct['caption_order'])

    order_numbers.sort()

    for dct in new_dct_list:
      dct['caption_order'] = order_numbers.index(dct['caption_order'])+1

    return new_dct_list

  def _reduce_noise_by_title(self, title, cases):

    '''Method used to ignore certain articles by the content of their titles.'''

    if ('review' in title.lower() or 'study' in title.lower()) and not ('case' in title.lower()):
      return []
    else:
      return cases

  def _reduce_caption_noise(self, cases, caption_dicts):

    if len(cases) == 0:
      return []
    else:
      return caption_dicts

  ### Methods for Combining JSONs with Metadata and Content

  def _combine_jsons(self, metadata, article_content):

    '''Method used to combine jsons with metadata and article content that belong to the same article.'''

    article_dict = metadata
    id = article_dict['pmcid']
    if article_dict['keywords'] == []:
      article_dict['keywords'] = article_content['keywords']
    article_dict['license'] = article_content['license']
    article_dict['case_amount'] = article_content['case_amount']

    cases = []
    if 'cases' in article_content.keys():
      for i, case_ in enumerate(article_content['cases']):
        cases.append({'case_id': f"{id}_{('0' + str(i+1))[-2:]}", 'case_text': case_.strip()})

    caption_dicts = []
    if 'caption_dicts' in article_content.keys():
      caption_dicts = article_content['caption_dicts']

    return {'article_id': id, 'article_metadata': article_dict, 'cases': cases, 'captions': caption_dicts}

  ### Methods used to Extract Demographic Information from Cases

  def _extract_demographics(self, combined_json):

    '''Method used to get the age and the gender of the patient mentioned in a case.
    Age extraction includes numerical and textual values (e.g. '16' and 'sixteen'). It returns 0 for ages lower than 1 year.
    Gender extraction returns either Female, Male or Transgender.'''

    for c in combined_json['cases']:
      c['age'] = self._get_age(c['case_text'])
      c['gender'] = self._get_gender(c['case_text'])

  def _get_age(self, text):

    '''Method used to get the age of the patient from the content of the clinical case.'''

    age_pattern_1 = r'(one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|\d+)(\s?-+|-?\s+)(year|yr| yo |month|week|day)s?([^.]+)?(\s?-+|-?\s+)old'
    match_1 = re.search(age_pattern_1, text.lower())
    if match_1:
      age_string = match_1.group()
    else:
      age_pattern_2 = r'( aged| age of) (one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|thirteen|fourteen|fifteen|sixteen|seventeen|eighteen|nineteen|twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|\d+)'
      match_2 = re.search(age_pattern_2, text.lower())
      if match_2:
        age_string = match_2.group()
      else:
        age_string = ''
    if re.search(r'(age|year| yo |yr)', age_string):
      age_match = re.search('\d+', age_string)
      if age_match:
        age = int(age_match.group())
      else:
        age = self._string_to_number(age_string)
    elif re.search(r'(month|week|day)', age_string):
      age = 0
    else:
      age = None
    if age and age > 130:
      age = None
    return age

  def _string_to_number(self, text):

    '''Method used to convert text to number (e.g. 'two' into '2').'''

    text2int = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10,
                "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
                "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90, "hundred": 100}

    number = 0

    new_string = 'y'.join(text.split('y')[:-1])

    for text_number in list(text2int.keys())[::-1]:
        numeric_word = re.search(text_number, new_string)
        if numeric_word:
            number = number + text2int[numeric_word.group()]
            new_string = new_string[:numeric_word.span()[0]] + new_string[numeric_word.span()[1]:]
    return number

  def _get_gender(self, text):

    '''Method used to get the gender of the patient from the content of the clinical case.'''

    gender_class = self._string_to_gender_class(text.split('.')[0])

    if gender_class == 'Unknown':
      gender_class = self._string_to_gender_class(text)

    return gender_class

  def _string_to_gender_class(self, text):

    '''Method used to get the gender class of the patient from the content of the clinical case.'''

    specific_transgender_pattern = r'(female-to-male|male-to-female|female to male|male to female|transgender|transexual)'
    other_transgender_pattern = r'( mtf | ftm )'

    specific_female_pattern = r'\s(woman|girl|mistress|lady|female|puerpera|pregnant)[s\s.,:]'
    other_female_pattern = r"(\s(she|her|herself|hers)[s\s.,']| f )"

    specific_male_pattern = r'\s(man|boy|gentleman|mister|male|puerpera|pregnant)[s\s.,:]'
    other_male_pattern = r"(\s(he|his|him|himself)[s\s.,']| m )"

    # Accronyms are searched for only in the first sentence of the case, to make sure not to get accronyms with a different meaning.
    if re.search(specific_transgender_pattern, text.lower()) or re.search(other_transgender_pattern, text.lower().split('.')[0]):
      return 'Transgender'
    elif re.search(specific_female_pattern, text.lower()):
      return 'Female'
    elif re.search(specific_male_pattern, text.lower()):
      return 'Male'
    elif re.search(other_female_pattern, text.lower()):
      return 'Female'
    elif re.search(other_male_pattern, text.lower()):
      return 'Male'
    else:
      return 'Unknown'

  ### Methods used to Assign Images to Cases

  def _assign_images(self, article_dict):

    '''Method used to assign caption dicts to cases.
    In a few cases, some captions dicts may be missing, so 'caption_order' cannot be used, because it's not possible to know which is the actual caption that is missing.
    For these cases, if any number is present in the caption tag, that will be considered as the caption_order.'''

    max_fig = self._get_max_figure(article_dict)
    if max_fig:
      if len(article_dict['captions']) < max_fig:
        for case_ in article_dict['cases']:
          case_captions = []
          for c in article_dict['captions']:
            if re.search('\d+', c['tag']):
              fig_order = int(re.search('\d+', c['tag']).group())
              c['caption_order'] = fig_order
              if fig_order in case_['figs_in_text']:
                case_captions.append(c)
          case_['case_image_list'] = case_captions
      else:
        for case_ in article_dict['cases']:
          case_captions = []
          for c in article_dict['captions']:
            if c['caption_order'] in case_['figs_in_text']:
              case_captions.append(c)
          case_['case_image_list'] = case_captions
    else:
      for case_ in article_dict['cases']:
        case_['case_image_list'] = []

  def _get_fig_numbers(self, input_string):

    '''Method used to get all the fig numbers mentioned in a string.'''

    ### Splitting the string into substrings that start with mentions of a figure (such as "fig." or "Figures") and end with a dot.

    patterns = r"(?:fig |figs |fig\.|figs\.|figure|figures|\(fig |\(figs |\(fig\.|\(figs\.|\(figure|\(figures)(?:[^.]+)\."
    matches = re.findall(patterns, input_string.lower())

    ### Splitting substrings that include more than one figure mention.

    substrings = []
    for m in matches:
      fig_substrings = m.split('fig')
      for s in fig_substrings:
        if s:
          substrings.append(s)

    ### Tokenization of substrings using blank spaces.

    token_lists = []
    for s in substrings:
      token_lists.append(re.split(r'\s+', s))

    ### Adding fig numbers found in tokens to the fig_list object.

    fig_list = []
    for l in token_lists:

      previous_number_idx = None
      fig_number = None
      l_fig_list = []
      range_token = False
      break_id = None # This is used to ignore numbers that appear after closing parenthesis, semi-colons or other similar special characters.

      for idx, token in enumerate(l):

        if re.search(r"[^a-zA-Z0-9\(,.\-]", token):
          break_id = idx +1

        if idx == break_id:
          break

        if 'year' not in token: ### to ignore tokens with /d+-year-old format.

          ### If the first number appears more than 5 tokens away from the figure mention, it's likely to be referring to something else, so the loop is broken.

          if previous_number_idx is None:
            if idx > 5:
              break

          ### If any number appears more than 5 tokens away from the previous number, it's likely to be referring to something else, so the loop is broken.

          else:
            if (idx - previous_number_idx) > 5:
              break

            ### If '-', 'to' or 'through' come between two numbers, it is in fact a range so range_token is True. This is used, for example, to be able to extract '4-7' as [4, 5, 6, 7].

            if re.search(r'-|to|through', token) and (previous_number_idx  is not None) and (l_fig_list != []):
              range_token = True
              range_start = l_fig_list[-1]

          ### All the numbers present in the token are extracted. If the token is not a range, only the first number is added to l_fig_list.

          fig_number = re.findall(r'\d+', token)
          if fig_number:
            previous_number_idx = idx
            if re.search(r'-', token) and len(fig_number) > 1:
              f_range = list(range(int(fig_number[0]), int(fig_number[-1])+1))
              for e in f_range:
                l_fig_list.append(int(e))
            else:
              l_fig_list.append(int(fig_number[0]))

      ### If l_fig_list includes more than one number and it is a range (range_token == True), then all the numbers from that range are added to the l_fig_list.

      if range_token and (len(l_fig_list) > 1):
        range_end_index = l_fig_list.index(range_start) + 1
        if len(l_fig_list) > range_end_index:
          l_fig_list = list(range(range_start, l_fig_list[range_end_index] +1))

      ### The elements from l_fig_list are added to the outcome fig_list. If the number is higher than 16, it's unlikely to be a figure number, so its filtered out.

      for element in l_fig_list:
        if element < 16:
          fig_list.append(element)

    return self._get_unique_in_order(fig_list)

  def _get_max_figure(self, article_dict):

    '''Method used to get the maximum figure number.'''

    figs = []
    for c in article_dict['cases']:
      for f in c['figs_in_text']:
        figs.append(f)
    if figs:
      return max(figs)
    else:
      return None

  ### Methods to Assign Text References to Images

  def _assign_text_references(self, case_dict):

    '''Method used to assign to each figure, the part of the case that mention such figure (e.g. sentences that contain '(see fig 1)').'''

    paragraphs = re.split('\n', case_dict['case_text'])
    sentences = []

    for p in paragraphs:
      sentence_limits = re.findall(r'\. [A-Z]', p)

      for i, l in enumerate(sentence_limits):
        sentence_end_index = re.search(l, p).span()[0]+1
        sentences.append(p[:sentence_end_index])
        p = p[sentence_end_index + 1:]
        if i == len(sentence_limits)-1:
          sentences.append(p)

    fig_references = []
    for s in sentences:
      patterns = r"(?:fig |figs |fig\.|figs\.|figure|figures|\(fig |\(figs |\(fig\.|\(figs\.|\(figure|\(figures)(?:[^.]+)\."
      if re.findall(patterns, s.lower()):
        fig_references.append(s)

    for c in case_dict['case_image_list']:
      c['text_references'] = []
      for r in fig_references:
        mentioned_figs = self._get_fig_numbers(r)
        if c['caption_order'] in mentioned_figs:
          c['text_references'].append(r)

  ########### CaptionPreprocessor Class ###########

class CaptionPreprocessor():

  def __init__(self, case_report_json):

    '''This class is used to create a dataset of image labels from a json with case report data.

    case_report_json: either path to a json file or a list of dictionaries created using the CaseReportDownloader class.'''

    if type(case_report_json) == str:
      with open(case_report_json, 'r') as f:
        self.case_reports = json.load(f)
    else:
      self.case_reports = case_report_json

    try:
      self.captions = []
      for j in self.case_reports:
        for c in j['cases']:
          for i in c['case_image_list']:
            self.captions.append({'image_id': i['image_id'], 'caption': i['caption']})
    except:
      raise TypeError("The format of the case report json file is not correct. case_report_json should be either path to a json file or a list of dictionaries created using the CaseReportDownloader class.")

    # Tokens and regex patterns used to identify image reference mentioned in text are set:

    self.reference_pattern = r'([;:./(/),]|-| and | to )'
    self.uppercase_reference_tokens = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L']
    self.lowercase_reference_tokens = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l']
    self.position_reference_tokens = ['upper', 'lower', 'left', 'right', 'top', 'bottom', 'above', 'below', 'center', 'centre', 'middle']

  ### Methods for string processing

  def _preprocess_string(self, input_string):

    '''Method used to remove dots that may interfere with other methods ('e.g.', 'i.e.' and decimal dots).'''

    string = input_string.replace("i.e.", "ie").replace("e.g.", "eg")
    output_string = ""

    # Decimal dots are turned into exclamation marks. They will be turned back to dots at the end of the process.
    for i in range(len(string)):
        if i < len(string) - 1 and string[i+1].isdigit() and string[i] == '.':
            output_string += '!'
        else:
            output_string += string[i]

    return output_string

  def _postprocess_string(self, string):

    '''Method used to return the decimal dots that were replaced by exclamation marks during the preprocessing, and to fix other parts of the string.'''

    output_string = ""
    for i in range(len(string)):
        if i < len(string) - 1 and string[i+1].isdigit() and string[i] == '!':
            output_string += '.'
        else:
            output_string += string[i]

    new_string = ''
    while new_string != output_string:
      new_string = output_string
      output_string = output_string.replace(' , and ', '').replace('(. ', '').replace('(.', '').replace(',,', ',').replace(',;', '').replace('..', '.').replace('  ', '. ').strip()
    if output_string.startswith('. ') or output_string.startswith('( ') or output_string.startswith(') '):
      output_string = output_string[2:]
    if output_string.endswith(' ('):
      output_string = output_string[:-2]
    if not output_string.endswith('.'):
      output_string += '.'

    return output_string

  def _add_reference_splitter(self, text):

    '''This method is used to identify references when there is no specific special character (e.g. 'a Chest CT').
    A semi colon is added ('a; Chest CT'), so that later on in the process the reference 'a' is correctly identified.'''

    letter_references = self.lowercase_reference_tokens + self.uppercase_reference_tokens
    if (len(text)>2) and (text[0] in letter_references) and (text[1] == ' ') and ((text.split()[1].istitle()) or ((text[0] != 'a') and text.split()[1].isupper())):
      return f'{text[:1]};{text[1:]}'
    else:
      return text

  def _split_sentences(self, text):

    '''Method used to split a paragraph in sentences.'''

    # Undesired dtos are removed, and the text is split using dots.
    preprocessed_text = self._preprocess_string(text)
    sentences = preprocessed_text.split('.')

    # Substrings with length lower than 5 are combined (they are likely abbreviations)
    combined_sentences = []
    for i in range(len(sentences) - 1):
        if len(sentences[i]) <= 4:
            combined_sentence = f"{self._add_reference_splitter(sentences[i].strip())}. {self._add_reference_splitter(sentences[i + 1].strip())}"
            sentences[i + 1] = combined_sentence
        else:
            combined_sentences.append(self._add_reference_splitter(sentences[i].strip()))

    # The last sentence is added (no combining needed)
    if sentences[-1]:
      combined_sentences.append(self._add_reference_splitter(sentences[-1].strip()))

    return combined_sentences

  def _remove_special_character(self, text):

    '''Method used to remove special characters from the beginning or end of a string.'''

    end_pattern = r'\W$'
    if re.search(end_pattern, text):
      text = text[:-1]

    start_pattern = r'^\W'
    if re.search(start_pattern, text):
      text = text[1:]
    return text

  ### Methods for caption split

  def _get_chunk_dict(self, split_text, reference_tokens):

    '''This method is used to identify reference tokens present in text (such as 'C' or 'd').'''

    chunk_dicts = []
    for chunk in split_text:
      if chunk.strip() in reference_tokens:
        chunk_dicts.append({'chunk': chunk, 'token_type': 'reference'})
      elif chunk.strip() in [';', ':', '(', ')', ',', '.', 'and', 'to', '-']:
        chunk_dicts.append({'chunk': chunk, 'token_type': 'split'})
      else:
        chunk_dicts.append({'chunk': chunk, 'token_type': 'other'})
    return chunk_dicts

  def _get_caption_references(self, text, reference_tokens):

    '''This method is used to get caption sections considering the reference tokens.'''

    split_text = re.split(self.reference_pattern, text)
    chunk_dicts = self._get_chunk_dict(split_text, reference_tokens)

    caption_sections = []
    section_string = ''
    reference_string = ''

    for i, chunk in enumerate(chunk_dicts):
      if chunk['token_type'] == 'other':
        if reference_string != '':
          caption_sections.append({'string': reference_string, 'type': 'reference'})
          reference_string = ''
        section_string += chunk['chunk']
      elif chunk['token_type'] == 'split':
        if reference_string != '':
          reference_string += chunk['chunk']
        else:
          section_string += chunk['chunk']
      elif chunk['token_type'] == 'reference':
        if section_string != '':
          caption_sections.append({'string': section_string, 'type': 'caption'})
          section_string = ''
        reference_string += chunk['chunk']

    if reference_string:
      caption_sections.append({'string': reference_string, 'type': 'reference'})
    if section_string:
      caption_sections.append({'string': section_string, 'type': 'caption'})

    for dct in caption_sections:
      dct['tidy_string'] = self._remove_special_character(dct['string']).strip()
      if (len(dct['tidy_string']) > 0) and (dct['type'] == 'caption'):
        dct['tidy_string'] += '.'

    # This part of the code is used to turn range references (such as 'a-d') to list references (such as 'a, b, c, d').
    pattern_1 = r'(,|;| and )'
    pattern_2 = r'(-| to )'
    list_of_letters = self.uppercase_reference_tokens + self.lowercase_reference_tokens

    for dct in caption_sections:
      if dct['type'] == 'reference':
        dct['tidy_refs'] = []
        refs = re.split(pattern_1, dct['tidy_string'])
        for element in refs:
          if element != ' and ':
            consecutive_refs = re.split(pattern_2, element)
            if len(consecutive_refs) == 1:
              dct['tidy_refs'].append(consecutive_refs[0].strip())
            if len(consecutive_refs) > 1:
              if (consecutive_refs[0].strip() in list_of_letters) and (consecutive_refs[-1].strip() in list_of_letters):
                dct['tidy_refs'].append(consecutive_refs[0].strip())
                reduced_list_of_letters = list_of_letters[list_of_letters.index(consecutive_refs[0].strip())+1:list_of_letters.index(consecutive_refs[-1].strip())]
                for letter in reduced_list_of_letters:
                  dct['tidy_refs'].append(letter)
                dct['tidy_refs'].append(consecutive_refs[-1].strip())
              else:
                dct['tidy_refs'] = ['manual_review_needed']

    return caption_sections

  def _get_reference_tokens(self, text, reference_tokens):

    '''This method is used to test which reference tokens are included in a specific caption (either uppercase letter, lowercase letter or position token).'''

    split_text = re.split(self.reference_pattern, text)
    is_reference = False
    for chunk in split_text:
      if chunk.strip() in reference_tokens:
        is_reference = True
    return is_reference

  def _organize_captions(self, caption_paragraph):

    '''This method is used to map the parts of the full caption to each subcaption reference ('a', 'b', 'c', etc.).'''

    # Only one list of reference tokens is selected. The most prioritary is uppercase tokens, then lowercase tokens and finally position tokens.
    reference_tokens = None
    for ref in [self.uppercase_reference_tokens, self.lowercase_reference_tokens, self.position_reference_tokens]:
      for sentence in self._split_sentences(caption_paragraph):
        if self._get_reference_tokens(sentence, ref):
          reference_tokens = ref
          break
      if reference_tokens:
        break
    if not reference_tokens:
      reference_tokens = self.uppercase_reference_tokens

    # Some parts of the full caption are considered to be common to all the subcaptions. (e.g. in 'Chest CT scan. A. Lung Fields, B. Heart.', 'Chest CT scan' will be included in subcaptions A and B).
    organized_captions = []
    last_ref = ['common_string']
    for sentence in self._split_sentences(caption_paragraph):
      caption_refs = self._get_caption_references(sentence, reference_tokens)
      refs = [ref for ref in caption_refs if ref['type'] == 'reference']
      caps = [ref for ref in caption_refs if ref['type'] == 'caption']
      if (len(refs) == 0):
        organized_captions.append({'sentence': sentence, 'reference':last_ref})
      elif (len(refs) == 1):
        last_ref = refs[-1]['tidy_refs']
        organized_captions.append({'sentence': sentence, 'reference':last_ref})
      else:
        if len(refs) == len(caps):
          for i, r in enumerate(refs):
            last_ref = refs[i]['tidy_refs']
            organized_captions.append({'sentence': caps[i]['string'], 'reference': r['tidy_refs']})
        else:
          for i, c in enumerate(caps):
            if i != len(caps)-1:
              split_c = re.split(r'(,|;| and )', c['string'])
              last_ref = refs[i]['tidy_refs']
              if (len(split_c) == 1) or (i==0):
                organized_captions.append({'sentence': c['string'], 'reference': refs[i]['tidy_refs']})
              else:
                organized_captions.append({'sentence': ','.join(split_c[:-1]), 'reference': refs[i-1]['tidy_refs']})
                organized_captions.append({'sentence': split_c[-1], 'reference': refs[i]['tidy_refs']})
            else:
              organized_captions.append({'sentence': c['string'], 'reference': last_ref})
    return organized_captions

  def _get_caption_mapping(self, preprocessed_captions):

    '''This method is used to include in the same subcaption parts of the full caption that have the same reference, even in those parts are in different parts of the subcaption.'''

    refs = self._get_unique_values_with_order([r for c in preprocessed_captions for r in c['reference'] if r != 'common_string'])
    mapping_dicts = []
    if refs:
      for ref in refs:
        mapping_dicts.append({'reference': ref, 'caption': '. '.join([c['sentence'] for c in preprocessed_captions if ((c['reference'] == ['common_string']) or (ref in c['reference']))])})
    else:
      mapping_dicts.append({'reference': 'undivided_caption', 'caption': '. '.join([c['sentence'] for c in preprocessed_captions])})
    return mapping_dicts

  def _get_unique_values_with_order(self, input_list):

    '''This method is used to make sure not to change the order of references when getting their unique values.'''

    unique_values = []
    seen_values = set()
    for item in input_list:
      if item not in seen_values:
        seen_values.add(item)
        unique_values.append(item)
    return unique_values

  def _preprocess_caption(self, caption):

    '''This method is used to get the list of references and subcaptions given a caption as an input.'''

    subcaptions = self._organize_captions(caption)
    c_mapping = self._get_caption_mapping(subcaptions)
    # Incorrect comma reference is filtered out.
    c_mapping = [c for c in c_mapping if c['reference'] != ',']
    return c_mapping

  def _fix_start(self, string):

    '''Method used to fix the start of a string.'''

    if len(string) > 0:
      if string[0].isalnum() == False:
        if (string[0] != '(') or (string[1].isalnum() == False):
          return string[1:]
        else:
          return string[:]
      elif string[:4].lower() == 'and ':
        return string[4:]
      else:
        return string[:]
    else:
      return ''

  def _fix_end(self, string):

    '''Method used to fix the end of a string.'''

    if string[-1].isalnum() == False:
      if (string[-1] != ')') or (string[-2].isalnum() == False):
        return string[:-1]
      else:
        return string[:]
    elif string[-4:] == ' and':
      return string[:-4]
    else:
      return string[:]

  def _fix_string(self, string):

    '''Method used to fix a string.'''

    new_string = self._fix_start(string)
    while new_string != string:
      temp_string = new_string[:]
      new_string = self._fix_start(new_string)
      string = temp_string[:]
    string = new_string[:1].upper() + new_string[1:]

    new_string = self._fix_end(string)
    while new_string != string:
      temp_string = new_string[:]
      new_string = self._fix_end(new_string)
      string = temp_string[:]
    string = new_string[:] + '.'

    return string[:]

  def _capitalize_after_period(self, string):

    '''Method used to fix capitalization.'''

      # Capitalize the first character after ". "
    return re.sub(r'(?<=\.\s)([a-z])', lambda x: x.group().upper(), string)

  def _fix_caption_string(self, string):

    '''Method used to fix a caption.'''


    if not any(char.isalnum() for char in string):
      return 'Medical image.'
    else:
      new_string = self._fix_string(string)
      new_string = self._capitalize_after_period(new_string)
      if len(new_string) < 5:
        return 'Medical image.'
      else:
        return new_string

  ### Methods to create the subcaption dataset

  def create_subcaption_dataset(self, csv_path = None):

    '''This method is used to loop through all the captions to create the output dataframe.
    csv_path (str): path where the output dataframe will be saved.'''

    self.subcaption_dicts = []
    for c in tqdm(self.captions):
      dct = dict()
      dct['image_id'] = c['image_id']
      dct['caption'] = c['caption']
      preprocessed_captions = self._preprocess_caption(c['caption'])
      preprocessed_captions = [pc for pc in preprocessed_captions if ' ' not in pc['reference']] # To filter a few edge cases
      dct['subcaption_amount'] = len(preprocessed_captions)
      for i, pc in enumerate(preprocessed_captions):
        subcaption_dict = {}
        subcaption_dict = dct.copy()
        subcaption_dict['sub_caption'] = self._postprocess_string(pc['caption'])
        subcaption_dict['reference'] = pc['reference']
        try:
          subcaption_dict['subcaption_order'] = self.uppercase_reference_tokens.index(dct['reference'].upper()) + 1
        except:
          subcaption_dict['subcaption_order'] = i+1
        article_id = subcaption_dict['image_id'].split('_')[0]
        file_name = '_'.join(subcaption_dict['image_id'].split('_')[2:])
        subcaption_dict['subcaption_id'] = f"{article_id}_{'.'.join(file_name.split('.')[:-1])}_{subcaption_dict['reference'].split('_')[0]}_{subcaption_dict['subcaption_order']}_{subcaption_dict['subcaption_amount']}.{file_name.split('.')[1]}"
        self.subcaption_dicts.append(subcaption_dict)

    self.label_dataset = pd.DataFrame(self.subcaption_dicts)
    self.label_dataset = self.label_dataset.drop_duplicates(ignore_index = True)
    self.label_dataset['caption'] = self.label_dataset.apply(lambda row: self._fix_caption_string(row['caption']) if 'undivided' not in row['subcaption_id'] else row['caption'], axis=1)
    if csv_path:
      self.label_dataset.to_csv(csv_path, index = False)

  ##### ImagePreprocessor Class #####

class ImagePreprocessor():

  def __init__(self, input_csv_path, output_folder = 'image_dataset', new_dataset = False,
               gaussian_n = 13, low_canny = 1, high_canny = 20, edge_minimum = 100, min_size = 100, cropping_iterations = 3, data_split = 100):

    '''
    Class instatiantion.
    input_csv_path (str): path to file with caption data.
    output_folder (str): path to the folder that will be used to save the dataset.
    new_dataset (bool): if True, any saved data will be deleted before starting the process. If False, the process will start from the previous progress.
    gaussian_n (int): size of the Gaussian filter used for image preprocessing.
    low_canny (int): low threshold of the Canny filter used for image preprocessing.
    high_canny (int): high threshold of the Canny filter used for image preprocessing.
    edge_minimum (int): amount of edges used to identify image borders.
    min_size (int): any image with fewer pixels (in any axis) than this min_size will be disregarded.
    cropping_iterations (int): number of iterations used in the loop for cropping.
    '''

    # Creation of the direction for the dataset.
    self.output_folder = output_folder
    self.data_split = data_split

    if not os.path.exists(self.output_folder):
      os.makedirs(self.output_folder)
    else:
      if new_dataset: # if new_dataset, previous files are deleted.
        shutil.rmtree(self.output_folder)
        os.makedirs(self.output_folder)

    if isinstance(self.data_split, int) and self.data_split > 0:
      self.partial_data_folders = []
      for s in range(self.data_split):
        split_path = f"{self.output_folder}/data_splits/split_{s+1}_of_{self.data_split}"
        if not os.path.exists(split_path):
          os.makedirs(split_path)
          os.makedirs(f"{split_path}/processed_images")
          os.makedirs(f"{split_path}/progress")
        self.partial_data_folders.append(split_path)
    else:
      raise TypeError("data_split must be a positive integer.")

    # Creation of the object with image data.
    self.input_csv_path = input_csv_path
    self.caption_df = pd.read_csv(self.input_csv_path)
    self.caption_df['pmcid'] = self.caption_df['file'].apply(lambda x: x.split('_')[0])

    self.caption_df_split = np.array_split(self.caption_df.groupby('pmcid').agg(list), self.data_split)
    for i, df_split in enumerate(self.caption_df_split):
      df_split_path = f"{self.partial_data_folders[i]}/labeled_images.csv"
      if not os.path.exists(df_split_path):
        df_split.explode(list(df_split.columns)).reset_index().to_csv(df_split_path)

    # Params for image preprocessing.
    self.gaussian_n = gaussian_n
    self.low_canny = low_canny
    self.high_canny = high_canny
    self.edge_minimum = edge_minimum
    self.min_size = min_size
    self.cropping_iterations = cropping_iterations

  def create_image_dataset(self):

    '''
    Method used to create the image dataset.
    '''

    print("Creating image dataset.")
    for i, folder in enumerate(self.partial_data_folders):
      print(f"Batch {i+1} out of {len(self.partial_data_folders)}.")
      self._run_image_processing(folder)

  ##### Methods for Data Preprocessing Methods

  def _get_subimage_data(self, subcaption_id):

    '''
    This method is used to turn a caption_df row into a dictionary with subcaption data.
    '''

    dct = {}
    subcaption_id_parts = subcaption_id.split('_')
    dct['pmcid'] = subcaption_id_parts[0]
    if 'undivided' in subcaption_id:
      dct['image_type'] = 'single'
    else:
      dct['image_type'] = 'multiple'
    dct['file_name'] = '_'.join(subcaption_id_parts[1:-3]) + '.jpg' # In case the name of the file includes underscores.
    dct['subcaption_id'] = subcaption_id
    dct['reference'] = subcaption_id_parts[-3]
    dct['image_order'] = subcaption_id_parts[-2]
    dct['image_amount'] = subcaption_id_parts[-1].split('.')[0]

    return dct

  def _get_image_data(self, image_dicts, processed_pmcid_list):

    '''
    This method is used to turn a list with subcaption data into the suitable format to run the process.
    '''

    if image_dicts:
      image_df = pd.DataFrame(image_dicts)
      image_df['subimage_dict'] = image_df.apply(lambda x: {'subcaption_id': x['subcaption_id'],
                                                            'reference': x['reference'],
                                                            'image_order': x['image_order']}, axis = 1)
      image_df.drop(['subcaption_id', 'reference', 'image_order'], axis = 1, inplace = True)
      image_df = image_df.groupby(['pmcid', 'image_type', 'file_name', 'image_amount']).agg(list).reset_index()
      image_df['image_dict'] = image_df.apply(lambda x: {'file_name': x['file_name'],
                                                        'image_type': x['image_type'],
                                                        'image_amount': x['image_amount'],
                                                        'subimage_dicts': x['subimage_dict']}, axis = 1)
      image_df.drop(['file_name', 'image_type', 'image_amount', 'subimage_dict'], axis = 1, inplace = True)
      image_df = image_df.groupby(['pmcid']).agg(list).reset_index()
      image_df['data_dicts'] = image_df.apply(lambda x: {'pmcid': x['pmcid'], 'image_dicts': x['image_dict']}, axis = 1)
      image_df.drop(['pmcid', 'image_dict'], axis = 1, inplace = True)
      image_data = list(image_df['data_dicts'])
    else:
      image_data = []

    if processed_pmcid_list: # PMCIDs that were already processed are not included in the list.
      new_list = []
      n = 0
      processed_n = len(set(processed_pmcid_list))
      for element in image_data:
        n+=1
        if n > processed_n:
          new_list.append(element)
    else:
      new_list = image_data.copy()

    return new_list

  ### Methods for image processing

  def _find_transition_indices(self, boolean_array):

      '''Method used to get the indices in which True transitions to False or False to True in a boolean array.'''

      transition_indices = [0]
      prev_value = None

      for idx, value in enumerate(boolean_array):
          if prev_value is not None and value != prev_value:
              transition_indices.append(idx)
          prev_value = value
      transition_indices.append(idx)
      return transition_indices

  def _find_image_limits(self, index_list):

    '''Method used to filter in transition indices according to a size threshold.'''

    image_limits = []
    for i, pixel_index in enumerate(index_list):
      if i != len(index_list)-1:
        next_pixel_index = index_list[i+1]
        if (next_pixel_index - pixel_index) >= self.min_size:
          image_limits.append((pixel_index, next_pixel_index))

    return image_limits

  def _crop_image(self, image, origin, axis):

    '''Method used to crop an edge image considering a thershold for size and edge sum.'''

    axis_sums = np.sum(image, axis=axis)
    bool_axis_sum = axis_sums >= self.edge_minimum
    transition_indices = self._find_transition_indices(bool_axis_sum)
    image_limits = self._find_image_limits(transition_indices)

    im_list = []
    for limit in image_limits:
      if axis == 0:
        new_img = image[:,limit[0]:limit[1]]
        new_origin = (origin[0] + limit[0], origin[1])
      elif axis == 1:
        new_img = image[limit[0]:limit[1],:]
        new_origin = (origin[0], origin[1] + limit[0])
      im_list.append({'image': new_img, 'origin': new_origin})

    return im_list

  def _get_output_images(self, input_list):

    '''Method used to get the output list of images given an input image, by cropping in both axis.'''

    ax = 0
    output_0 = []
    for element in input_list:
      cropped_images = self._crop_image(element['image'], element['origin'], axis = ax)
      for im in cropped_images:
        output_0.append(im)

    ax = 1
    output_1 = []
    for element in output_0:
      cropped_images = self._crop_image(element['image'], element['origin'], axis = ax)
      for im in cropped_images:
        output_1.append(im)

    return output_1

  def _sort_by_origin(self, image_list):

    '''
    This method is used to sort subimages (left to right, top to bottom), so that the second image is always the one to the right from the left image.
    '''
    if image_list:
      for im in image_list:
        im['distance_from_left'] = im['origin'][0]//50
        im['distance_from_top'] = im['origin'][1]//50
      im_df = pd.DataFrame(image_list).sort_values(by=['distance_from_top', 'distance_from_left'])
      im_df['dicts'] = im_df.apply(lambda x: {'image': x['image'], 'origin': x['origin']}, axis=1)
      sorted_image_list = list(im_df['dicts'])
    else:
      sorted_image_list = []
    return sorted_image_list

  def _filter_empty_images(self, output_list):

    '''
    Method used to filter out images that are only black or white.
    '''

    filtered_list = []
    for l in output_list:
      if (l['image'].min() < 230) and (l['image'].max() > 25):
        filtered_list.append(l)
    return filtered_list

  def _process_image(self, input_path, output_paths, subimage_amount, progress_folder, processed_image_folder):

    '''
    Method used to process an image and save the corresponding subimages.
    '''

    # The original image is read, changed to gray, blurred, and edges are identified.
    image = cv2.imread(input_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred_image = cv2.GaussianBlur(gray, (self.gaussian_n, self.gaussian_n), 0)
    edges = cv2.Canny(blurred_image, self.low_canny, self.high_canny)

    # The image is cropped.
    input_list = [{'image': edges, 'origin': (0,0)}]
    output_list = []
    i = 0
    while i<self.cropping_iterations:
      output_list = self._get_output_images(input_list)
      if len(input_list) == len(output_list):
        i+=1
      input_list = output_list[:]

    # The subimages are sorted by their position.
    output_list = self._sort_by_origin(output_list)

    # If there are extra subimages, black or white images are filtered out.
    if len(output_list) > int(subimage_amount):
      output_list = self._filter_empty_images(output_list)

    pmcid = input_path.split('/')[-1].split('_')[0]

    # Subimages are included in the processed_image_folder only if the number of subimages is the same as the subimage_amount (amount of subcaptions).
    if output_list:
      for ix, im in enumerate(output_list):
        if len(output_list) == int(subimage_amount):
          new_path = f"{processed_image_folder}/{output_paths[ix].split('/')[-1]}"
          if ix == 0:
            with open(f"{progress_folder}/{pmcid}_{'_'.join(input_path.split('/')[-1].split('_')[1:])}_ok.txt", 'w') as file:
              file.write('')
          new_image = image[im['origin'][1]:im['origin'][1] + im['image'].shape[0],im['origin'][0]:im['origin'][0] + im['image'].shape[1]]
          cv2.imwrite(new_path, new_image)
        else:
          if ix == 0:
            with open(f"{progress_folder}/{pmcid}_{'_'.join(input_path.split('/')[-1].split('_')[1:])}_incorrect-output-amount.txt", 'w') as file:
              file.write('')
    else:
      with open(f"{progress_folder}/{pmcid}_{'_'.join(input_path.split('/')[-1].split('_')[1:])}_zero-output-amount.txt", 'w') as file:
        file.write('')

  def _run_image_processing(self, folder):

    '''
    This method is used to get the image files from the API from Europe PMC, and then process them.
    '''

    # Getting checkpoints.
    progress_folder = f"{folder}/progress"
    processed_image_folder = f"{folder}/processed_images"
    processed_file_list = ['_'.join(f.split('_')[:2]) for f in os.listdir(progress_folder)]
    processed_pmcid_list = [f.split('_')[0] for f in processed_file_list]

    # Getting image data.
    split_df = pd.read_csv(f"{folder}/labeled_images.csv")
    # image_dicts = list(split_df['subcaption_id'].apply(self._get_subimage_data))
    image_dicts = list(split_df['file'].apply(self._get_subimage_data))
    image_data = self._get_image_data(image_dicts, processed_pmcid_list)

    for img in tqdm(image_data):
      pmcid = img['pmcid']
      url = f"https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/supplementaryFiles?includeInlineImage=true"
      r = requests.get(url)
      file_folder = 'supplementary_files'
      if str(r) != "<Response [200]>": # In case there is any API error.
        for file_ in img['image_dicts']:
          with open(f"{progress_folder}/{pmcid}_{file_['file_name']}_api-error.txt", 'w') as file:
            file.write('')
      else:
        try:
          with open(f'{file_folder}.zip', 'wb') as f:
            f.write(r.content)
          shutil.unpack_archive(f'{file_folder}.zip', file_folder)
          file_list = os.listdir(file_folder)

          for file_ in img['image_dicts']:
            if file_['file_name'] in file_list:
              old_file_name = f"{file_folder}/{file_['file_name']}"
              input_path = f"{file_folder}/{pmcid}_{file_['file_name']}"
              os.rename(old_file_name, input_path)
              output_paths = [f"{processed_image_folder}/{f['subcaption_id']}" for f in file_['subimage_dicts']]
              subimage_amount = file_['image_amount']
              self._process_image(input_path, output_paths, subimage_amount, progress_folder, processed_image_folder)
            else: # In case the name of an image is not found in the list of files corresponding to a certain PMCID.
              with open(f"{progress_folder}/{pmcid}_{file_['file_name']}_image-not-found.txt", 'w') as file:
                file.write('')

          if os.listdir(file_folder): # Files are removed from the temporary folder.
            for f in os.listdir(file_folder):
              os.remove(f"{file_folder}/{f}")

        except: # In case there is any API error (exceptional articles that are not open access).
          for file_ in img['image_dicts']:
            try:
              with open(f"{progress_folder}/{pmcid}_{file_['file_name']}_api-error.txt", 'w') as file:
                file.write('')
            except:
              with open(f"{progress_folder}/{pmcid}_{file_['file_name'][:5]}_api-error.txt", 'w') as file: ## to avoid an error in one particular case.
                file.write('')
