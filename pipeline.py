from fastai.vision.all import *
image_extensions.add('.webp')

class MultiCaReClassifier():

  def __init__(self, image_folder, models_root = 'MultiCaReClassifier/models', save_path = '', add_multiclass_columns = False):

    '''Class used to classify medical images considering their types (such as ultrasound or MRI), and the corresponding anatomical region and view (for radiology images only).
    image_folder (str): folder containing all the input images.
    models_root (str): folder containing the image classification models.
    save_path (str): path to save the inference table.
    add_multiclass_columns (bool): if True, multiclass columns will be added to the dataframe based on the multilabel column ('label_list').'''

    self.image_folder = os.path.join(image_folder, '')
    self.models_root = models_root
    self.save_path = save_path
    self.add_multiclass_columns = add_multiclass_columns

    # List of possible labels per model.
    self.label_dict = {
        "image_type:radiology~anatomical_region:axial_region": ["abdomen", "breast", "head", "neck", "pelvis", "thorax"],
        "image_type:radiology~anatomical_region:lower_limb": ["ankle", "foot", "hip", "knee", "lower_leg", "thigh"],
        "image_type:radiology~anatomical_view": ["axial", "frontal", "intravascular", "oblique", "occlusal", "panoramic", "periapical", "sagittal", "transabdominal", "transesophageal", "transthoracic", "transvaginal"],
        "image_type:endoscopy": ["airway_endoscopy", "arthroscopy", "ig_endoscopy", "other_endoscopy"],
        "image_type:electrography": ["eeg", "ekg", "emg"],
        "image_type:ophthalmic_imaging": ["autofluorescence", "b_scan", "fundus_photograph", "gonioscopy", "oct", "ophtalmic_angiography", "slit_lamp_photograph"],
        "image_type:radiology~anatomical_region:upper_limb": ["elbow", "forearm", "hand", "shoulder", "upper_arm", "wrist"],
        "image_type:radiology~anatomical_region": ["axial_region", "lower_limb", "upper_limb", "whole_body"],
        "image_type:radiology~main": ["ct", "mri", "pet", "scintigraphy", "spect", "tractography", "ultrasound", "x_ray"],
        "image_type:pathology": ["acid_fast", "alcian_blue", "congo_red", "fish", "giemsa", "gram", "h&e", "immunostaining", "masson_trichrome", "methenamine_silver", "methylene_blue", "papanicolaou", "pas", "van_gieson"],
        "image_type:radiology~anatomical_region:axial_region.thorax": ["cardiac_image", "other_thoracic_image"],
        "image_type:medical_photograph": ["oral_photograph", "other_medical_photograph", "skin_photograph"],
        "image_type": ["chart", "electrography", "endoscopy", "medical_photograph", "ophthalmic_imaging", "pathology", "radiology"]
        }

    # The outcome dataframe is created.
    self.image_paths = get_image_files(self.image_folder)
    self.data = pd.DataFrame(columns=[name for name in self.label_dict.keys() if os.path.isdir(os.path.join(models_root, name.replace(':', '_')))])
    self.data['image_path'] = self.image_paths
    self.predict_image_classes()

  ### Main Methods ###

  def predict_image_classes(self):

    '''Method used to get the predictions for each image.'''

    # Models are ran one level of hierarchy at a time.
    model_order = 1
    while True:
      order_count = 0
      for model_name in self.label_dict.keys():
        if len(re.split(r'[:.]', model_name)) == model_order: # The count of ':' and '.' in a model name is equivalent to its level of hierarchy.
          self._add_predictions(model_name)
          order_count += 1
      if order_count == 0: # If a level of hierarchy is empty, then the process is finished.
        break
      model_order += 1
    
    # Postprocessing is applied.
    self.apply_postprocessing()
    if self.save_path:
      self.data.to_csv(self.save_path, index=None)

  def apply_postprocessing(self):

    '''Method used to postprocess the predictions.'''

    # All predictions are merged in a single column as a list.
    columns_to_flatten = [c for c in self.data.columns if c.startswith('image_type')]
    self.data['label_list'] = self.data[columns_to_flatten].values.tolist()
    self.data['label_list'] = self.data['label_list'].apply(lambda x: [item for item in x if isinstance(item, (str, np.str_))])
    self.data.drop(columns_to_flatten, axis = 1, inplace = True)

    # Class typos are fixed and certain classes with low accuracy are merged.
    replacement_dict = {'transesophageal': 'ultrasound_view', 'transthoracic': 'ultrasound_view', 'transabdominal': 'ultrasound_view',
                        'transvaginal': 'ultrasound_view', 'ophtalmic_angiography': 'ophthalmic_angiography', 'ig_endoscopy': 'gi_endoscopy'}

    self.data['label_list'] = self.data['label_list'].apply(lambda x: [replacement_dict.get(item, item) for item in x])

    # Compound classes are added if there corresponding component classes are present.
    self.data['label_list'] = self.data['label_list'].apply(lambda x: self._add_compound_classes(x))

    # If multiclass columns are required, they are added.
    if self.add_multiclass_columns:
      self._generate_multiclass_columns()

    # Auxiliary classes are removed from the label list. This were added for the sake of the class structure of the taxonomy, but they do not add value to the list of predictions.
    auxiliary_classes = ['axial_region', 'cardiac_image', 'other_thoracic_image', 'intravascular', 'ultrasound_view']
    self.data['label_list'] = self.data['label_list'].apply(lambda x: [item for item in x if item not in auxiliary_classes])

  ### Auxiliary Methods ###

  def _identify_upper_model(self, model_name):

    '''Method used to identify the corresponding upper model of a given model.'''

    colon_index = search_last_match(model_name, ':')
    dot_index = search_last_match(model_name, '.')
    index = max(colon_index, dot_index)
    if index != -1:
      return model_name[:index]
    else:
      return None  

  def _search_last_match(self, string, character):

    '''Method used to find the last mention of a character in a string.'''

    if character in string:
      return string.rindex(character)
    else:
      return -1

  def _add_predictions(self, model_name):

    '''Method used to add all the predictions of a given model to the outcome dataframe.'''

    upper_model = self._identify_upper_model(model_name)

    # Models are used depending on the outcome of models from a higher hierarchy.
    if upper_model is not None:
      condition_class = model_name.split(':')[-1].split('~')[0].split('.')[-1]
      condition = self.data[model_name].isnull() & (self.data[upper_model] == condition_class)
    else:
      condition = self.data[model_name].isnull()
    imgs = self.data[condition].image_path.values

    labels = np.array(self.label_dict[model_name])

    # Models are ran.
    if len(imgs) > 0:
      device = 'cpu'
      checkpoint_file = os.path.join(self.models_root, model_name.replace(':', '_'), 'model')
      dls = ImageDataLoaders.from_path_func('', imgs, lambda x: '0', item_tfms=Resize((224,224), method='squish'))
      learn = vision_learner(dls, resnet50, n_out=len(labels)).to_fp16()
      learn.load(checkpoint_file, device=device)
      test_dl = learn.dls.test_dl(imgs, device=device)
      probs, _ = learn.get_preds(dl=test_dl)
      self.data.loc[condition, model_name] = labels[probs.argmax(axis=1)]
    
  def _add_compound_classes(self, input_class_list):

    '''This method is used to add compound classes to the label list if the corresponding component classes are present.'''

    compound_class_dicts = [
        {'compound_class': 'echocardiogram', 'components': ['ultrasound', 'cardiac_image']},
        {'compound_class': 'ivus', 'components': ['ultrasound', 'intravascular']},
        {'compound_class': 'mammography', 'components': ['x_ray', 'breast']}
        ]

    for dct in compound_class_dicts:
      condition = True
      for cls in dct['components']:
        if cls not in input_class_list:
          condition = False
          break
      if condition:
        if dct['compound_class'] not in input_class_list:
          input_class_list.append(dct['compound_class'])

    return input_class_list

  def _generate_multiclass_columns(self):

    '''Method used to generate the multiclass columns based on the label list.'''

    image_types = ['chart', 'radiology', 'pathology', 'medical_photograph', 'ophthalmic_imaging', 'endoscopy', 'electrography']
    self.data['image_type'] = self.data['label_list'].apply(lambda x: self._get_column_label(x, image_types))

    image_subtypes = ['chart',
                      'ct', 'mri', 'x_ray', 'pet', 'spect', 'scintigraphy', 'ultrasound', 'tractography',
                      'acid_fast', 'alcian_blue', 'congo_red', 'fish', 'giemsa', 'gram', 'h&e', 'immunostaining', 'masson_trichrome', 'methenamine_silver', 'methylene_blue', 'papanicolaou', 'pas', 'van_gieson',
                      'skin_photograph', 'oral_photograph', 'other_medical_photograph',
                      'b_scan', 'autofluorescence', 'fundus_photograph', 'gonioscopy', 'oct', 'ophthalmic_angiography', 'slit_lamp_photograph',
                      'gi_endoscopy', 'airway_endoscopy', 'other_endoscopy', 'arthroscopy',
                      'eeg', 'emg', 'ekg']

    self.data['image_subtype'] = self.data['label_list'].apply(lambda x: self._get_column_label(x, image_subtypes))

    anatomical_regions = ['abdomen', 'breast', 'head', 'neck', 'pelvis', 'thorax',
                          'lower_limb', 'upper_limb', 'whole_body']

    self.data['radiology_region'] = self.data['label_list'].apply(lambda x: self._get_column_label(x, anatomical_regions))

    granular_anatomical_regions = ['abdomen', 'breast', 'head', 'neck', 'pelvis', 'thorax',
                                  'ankle', 'foot', 'hip', 'knee', 'lower_leg', 'thigh',
                                  'elbow', 'forearm', 'hand', 'shoulder', 'upper_arm', 'wrist',
                                  'whole_body']

    self.data['radiology_region_granular'] = self.data['label_list'].apply(lambda x: self._get_column_label(x, granular_anatomical_regions))

    anatomical_view = ['axial', 'frontal', 'sagittal', 'oblique',
                      'occlusal', 'panoramic', 'periapical', 'intravascular', 'ultrasound_view']

    self.data['radiology_view'] = self.data['label_list'].apply(lambda x: self._get_column_label(x, anatomical_view))

  def _get_column_label(self, column_list, label_list):

    '''Method used to get the label from a relevant list that is present in the predictions of a given image.'''

    label = ''
    for column in column_list:
      if column in label_list:
        label = column
    return label
