from owlready2 import *
import pandas as pd
pd.options.mode.chained_assignment = None
from datetime import datetime
import warnings
import re
import ast
import numpy as np

class MultiplexTaxonomyProcessor():

  def __init__(self, input_owl_path, refresh_ontology = True, output_owl_path = 'my_taxonomy.owl', root_class = 'root_class', additional_comment = 'No additional comment.'):

    '''
    The MultiplexTaxonomyProcessor class is used to turn an owl file with a Decision Rainforest format (used for Multiplex classification) into a Disjoint-Union-based Tree format (DUBT), and to do all the necessary pre- and post-processing.
    input_owl_path (str): Path to the input taxonomy file.
    refresh_ontology (bool, default True): When more than one taxonomy is used in the same runtime, previous uploaded ontology objects need to be destroyed.
    output_owl_path (str, default 'my_taxonomy.owl'): Path to the taxonomy file.
    root_class (str, default 'root_class'): Root class of the taxonomy.
    additional_comment (str, default 'No additional comment.'): Additional comment to be included in the output taxonomy file.
    '''

    self.input_owl_path = input_owl_path
    self.output_owl_path = output_owl_path
    self.root_class = root_class
    self.additional_comment = additional_comment
    self.taxonomy_errors = {'single_child_class': False, 'repeated_class_name': False, 'graph_structure': False, 'recursive_relation': False, 'empty_trees': False}
    self.rainforest_format_taxonomy = self._get_taxonomy()
    if refresh_ontology:
      try:
        while True:
          self.rainforest_format_taxonomy.destroy(update_relation = True, update_is_a = True)
      except:
        self.rainforest_format_taxonomy = self._get_taxonomy()
    self.class_map = self._create_class_map()
    self.postprocessing_class_map = self.class_map[self.class_map['class_path'].str.startswith('postprocessing')].reset_index(drop=True)
    self.class_map = self.class_map[~self.class_map['class_path'].str.startswith('postprocessing')].reset_index(drop=True)
    self.class_map['column_name'] = self.class_map.apply(lambda x: x['class_path'][:-(len(x['class_name'])+1)], axis = 1)
    self.class_map['conditioning_class'] = self.class_map.apply(lambda x: self._get_conditioning_class(x['column_name']), axis = 1)
    subsidiary_tree_classes = self.class_map.groupby('conditioning_class')['column_name'].agg(lambda x: x.nunique() > 1).reset_index().rename({'column_name': 'has_subsidiary_tree'}, axis = 1).copy()
    self.class_map = pd.merge(self.class_map, subsidiary_tree_classes, on='conditioning_class', how='left')
    self.class_map['column_name'] = self.class_map.apply(lambda x: self._fix_column_name(x), axis = 1)
    self.class_map.drop('has_subsidiary_tree', axis = 1, inplace = True)

    try:
      self._find_class_with_multiple_parents()
    except RecursionError:
      self.taxonomy_errors['recursive_relation'] = 'Cyclic Hierarchy. To fix this error, make sure that no subclass is an ancestor of its parent class.'
    if all(not value for value in self.taxonomy_errors.values()):
      self.preprocessing_mapping_dict = self._get_preprocessing_dict()
      self.onto_dicts = self._create_taxonomy_dicts()
      self._create_output_taxonomy()
    else:
      error_message = 'The taxonomy errors below were identified. If you used different taxonomies in the same runtime, try restarting the session before debugging the input owl file.\n'
      for key in self.taxonomy_errors.keys():
        if self.taxonomy_errors[key]:
          error_message += self.taxonomy_errors[key] + '\n'

      class TaxonomyError(Exception):
        pass
      raise TaxonomyError(error_message)

  ### Methods for Taxonomy Import

  def _get_taxonomy(self):

    owlready2.onto_path = []
    owlready2.onto_path.append(self.input_owl_path)
    onto = owlready2.get_ontology(self.input_owl_path).load()
    sync_reasoner()
    return onto

  ### Methods for Class Map Creation

  def _get_subclasses(self, node, subsidiary_relation, subsidiary_tree_dict = {}, class_path = ''):

    '''Given the initial node of a given tree, this method provides as an outcome a dictionary with the name of the tree,
    the names of its subsidiary trees, and its corresponding class tree.'''

    tree_name = node.label[0]

    if class_path == '':
      class_path = node.label[0] + ':'

    lst = []
    for child in node.subclasses():
      dct = dict()
      dct['class_name'] = child.label[0]
      if class_path[-1] == ':':
        special_char = ''
      else:
        special_char = '.'
      dct['class_path'] = class_path + special_char + child.label[0]
      if subsidiary_relation:
        dct['subsidiary_trees'] = [att.label[0] for att in subsidiary_relation[child]]
      else:
        dct['subsidiary_trees'] = []
      try:
        dct['subclasses'] = self._get_subclasses(child, subsidiary_relation, subsidiary_tree_dict, dct['class_path'])['classes']
      except RecursionError:
        dct['subclasses'] = []
        self.taxonomy_errors['recursive_relation'] = 'Cyclic Hierarchy. To fix this error, make sure that no subclass is an ancestor of its parent class.'
      lst.append(dct)

      if dct['subsidiary_trees']:
        for at in dct['subsidiary_trees']:
          subsidiary_tree_dict[at] = dct['class_path']

    return {'classes': lst, 'subsidiary_trees': subsidiary_tree_dict, 'tree_name': tree_name}

  def _extract_classes(self, classes, flat_dict):

    '''This method is used to get a flat dictionary with all the classes and their paths.'''

    for cls in classes:
      flat_dict[cls['class_name']] = cls['class_path']
      if cls.get('subclasses'):
        self._extract_classes(cls['subclasses'], flat_dict)

  def _find_taxonomy_errors(self, data):

    '''Method used to identify two types of taxonomy errors: Parent classes with only one child class, and two classes with the same name.'''

    subclasses_with_len_1 = []
    class_names = []
    class_name_counts = {}

    def recurse(tree):
      if isinstance(tree, dict):
        for key, value in tree.items():
          if key == 'class_name':
            class_names.append(value)
            if value in class_name_counts:
              class_name_counts[value] += 1
            else:
              class_name_counts[value] = 1
          elif key == 'subclasses' and len(value) == 1:
            subclasses_with_len_1.append(value[0])
          if isinstance(value, (list, dict)):
            recurse(value)
      elif isinstance(tree, list):
        for item in tree:
          recurse(item)

    try:
      recurse(data)
    except RecursionError:
      self.taxonomy_errors['recursive_relation'] = 'Cyclic Hierarchy. To fix this error, make sure that no subclass is an ancestor of its parent class.'

    repeated_class_names = [name for name, count in class_name_counts.items() if count > 1]

    if subclasses_with_len_1:
      self.taxonomy_errors['single_child_class'] = 'Single Child Class: ' + ', '.join(list(set([d['class_name'] for d in subclasses_with_len_1]))) + '. To fix this error, remove any single child class or add the corresponding siblings.'
    if repeated_class_names:
      self.taxonomy_errors['repeated_class_name'] = 'Repeated Class Name: ' + ', '.join(repeated_class_names) + '. To fix this error, rename the corresponding class to avoid class name repetition.'

  def _get_postprocessing_map(self):

    '''This method is used to get compound classes (aka pseudo classes), which are classes that are not used to train machine learning models,
    they are added during postprocessing if combinations of other classes are present.'''

    posprocessing_class = self.rainforest_format_taxonomy.search_one(label='postprocessing')
    composition_relation = self.rainforest_format_taxonomy.search_one(label="is_composed_by")

    if posprocessing_class and composition_relation:
      children = list(posprocessing_class.subclasses())

      compound_classes = []
      for child in children:
        related_classes = composition_relation[child]
        if related_classes:
          dct = {'class_name':child.label[0]}
          lst = []
          for related_class in related_classes:
            lst.append(related_class.label[0])
          dct['class_path'] = 'postprocessing:' + '+'.join(lst)
          compound_classes.append(dct)
      return compound_classes

  def _find_class_with_multiple_parents(self):

    '''
    This method is used to spot wrong taxonomy structures which include at least one class with multiple parents (something that is not allowed in Multiplex classification).
    '''
    class_names = []

    def extract_class_names(d):
      if isinstance(d, dict):
        for key, value in d.items():
          if key == 'class_name':
            class_names.append(value)
          elif isinstance(value, (dict, list)):
            extract_class_names(value)
      elif isinstance(d, list):
        for item in d:
          extract_class_names(item)

    extract_class_names(self.class_trees)

    seen = set()
    repeated = []
    for item in class_names:
      if item in seen:
        if item not in repeated:
          repeated.append(item)
      else:
        seen.add(item)

    if repeated:
      self.taxonomy_errors['graph_structure'] = 'Wrong Taxonomy Structure. Classes with multiple parents: ' + ', '.join(repeated) + '. To fix this error, make sure that each class has only one parent class.'

  def _create_class_map(self):

    '''This method creates the class maps for each class. The format includes dots (.) to separate between parent classes and children classes,
    colons (:) to separate between tree names and a class name, and tildes (~) to split between a class and an subsidiary tree.
    For example, the path 'main_tree:class_1.class_1_4~subsidiary_tree:class_c' (for class 'class_c') means that class_c belongs to subsidiary_tree,
    which is an subsidiary tree from class_1_4, which is a child from class_1, which belongs to main_tree.'''

    subsidiary_relation = self.rainforest_format_taxonomy.search_one(label="has_subsidiary_tree")

    self.class_trees = []
    for tree in list(self.rainforest_format_taxonomy.search_one(label='taxonomy').subclasses()):
      tree_subclasses = self._get_subclasses(tree, subsidiary_relation)
      self.class_trees.append(tree_subclasses)
      if tree_subclasses['classes'] == []:
        self.taxonomy_errors['empty_trees'] = f"Empty Class Tree: {tree_subclasses['tree_name']}"
      elif len(tree_subclasses['classes']) == 1: # Added in case the only-child class is the first class from the tree.
        self.taxonomy_errors['single_child_class'] = 'Single Child Class: ' + tree_subclasses['classes'][0]['class_name'] + '. To fix this error, remove any single child class or add the corresponding siblings.'
      self._find_taxonomy_errors(tree_subclasses)

    subsidiary_trees = {k:v for ct in self.class_trees for k, v in ct['subsidiary_trees'].items() if ct['subsidiary_trees']}

    tree_names = []
    subsidiary_tree_mapping = {}
    for class_tree in self.class_trees:
      tree_names.append(class_tree['tree_name'])
      for subsidiary_tree in class_tree['subsidiary_trees'].keys():
        subsidiary_tree_mapping[subsidiary_tree] = class_tree['subsidiary_trees'][subsidiary_tree]

    tree_mapping = {}

    while tree_names:

      main_trees = []
      for tree_path in tree_names:
        if tree_path.split('~')[-1] not in subsidiary_tree_mapping.keys():
          main_trees.append(tree_path)

      subsidiary_trees = []
      for tree in main_trees:
        for st in subsidiary_tree_mapping.keys():
          if subsidiary_tree_mapping[st].split('~')[-1].startswith(tree.split('~')[-1]):
            subsidiary_trees.append(st)

      for mt in main_trees:
        tree_mapping[mt.split('~')[-1]] = mt

      for at in subsidiary_trees:
        for mapping in subsidiary_tree_mapping.keys():
          if subsidiary_tree_mapping[mapping].startswith(at):
            prior_mapping = subsidiary_tree_mapping[mapping]
            subsidiary_tree_mapping[mapping] = subsidiary_tree_mapping[at] + '~' + prior_mapping
        for i, tree in enumerate(tree_names):
          if tree.split('~')[-1] == at:
            tree_names[i] = subsidiary_tree_mapping[at] + '~' + at
            del subsidiary_tree_mapping[at]
            break

      names = tree_names.copy()
      tree_names = []
      for tree in names:
        if tree not in main_trees:
          tree_names.append(tree)

    class_dict = {}
    for item in self.class_trees:
      self._extract_classes(item['classes'], class_dict)

    for key in class_dict.keys():
      tree_name = class_dict[key].split(':')[0]
      class_dict[key] = tree_mapping[tree_name] + class_dict[key][len(tree_name):]

    self.class_map = pd.DataFrame.from_dict(class_dict, orient='index', columns=['class_path']).reset_index().rename(columns={'index': 'class_name'})

    compound_classes = self._get_postprocessing_map()

    return pd.concat([self.class_map, pd.DataFrame(compound_classes)], ignore_index=True)

  def _get_preprocessing_dict(self):

    '''This method is used to create a dictionary for preprocessing.'''

    preprocessing_class = self.rainforest_format_taxonomy.search_one(label='preprocessing')
    maps_to_relation = self.rainforest_format_taxonomy.search_one(label="maps_to")

    if preprocessing_class and maps_to_relation:
      children = list(preprocessing_class.subclasses())
    else:
      children = []

    mapping_dict = {}
    for child in children:
      related_classes = maps_to_relation[child]

      if related_classes:
        mapping_dict[child.label[0]] = []
        for related_class in related_classes:
          mapping_dict[child.label[0]].append(related_class.label[0])
    return mapping_dict

  def _get_conditioning_class(self, column_name):

    '''This method is used to identify the conditioning class given a class path.'''

    colon_idx = max([column_name.rfind(':'), column_name.rfind('.')])
    class_separator_idx = max([column_name.rfind('~')])
    if colon_idx > class_separator_idx:
      return column_name
    elif class_separator_idx != -1:
      return column_name[:class_separator_idx]
    else:
      return self.root_class

  def _fix_column_name(self, row):

    '''This method is used to add '~main' to column names if a there are subsidiary trees for the same conditioning class.'''

    if (row['column_name'] == row['conditioning_class']) and row['has_subsidiary_tree']:
      return row['column_name'] + '~main'
    else:
      return row['column_name']

  ### Methods for Creating Dicts with Class Information

  def _adapt_preprocessing_dict(self):

    '''This method is used to adapt the preprocessing dictionary so that it can be used for self._create_taxonomy_dict().'''

    preprocessing_dict = {}
    for key in self.preprocessing_mapping_dict.keys():
      for value in self.preprocessing_mapping_dict[key]:
        if value not in preprocessing_dict.keys():
          preprocessing_dict[value] = [key]
        else:
          preprocessing_dict[value].append(key)
    return preprocessing_dict

  def _adapt_postprocessing_dict(self):

    '''This method is used to adapt the postprocessing dictionary so that it can be used for self._create_taxonomy_dict().'''

    initial_dicts_df = self.postprocessing_class_map.apply(lambda x: {cls:x['class_name'] for cls in x['class_path'].split(':')[-1].split('+')}, axis = 1)
    if not initial_dicts_df.empty:
      initial_dicts = initial_dicts_df.tolist()
    else:
      initial_dicts = []
    postprocessing_mapping_dict = {}
    for dct in initial_dicts:
      for key in dct.keys():
        if key not in postprocessing_mapping_dict.keys():
          postprocessing_mapping_dict[key] = [dct[key]]
        else:
          postprocessing_mapping_dict[key].append(dct[key])
    return postprocessing_mapping_dict

  def _get_parent_class(self, class_path):

    '''This method is used to identify the parent class of a given class considering its path.'''

    class_split = class_path.split(':')
    if '.' in class_split[-1]:
      parent_class = class_split[-1].split('.')[-2]
    elif '~' in class_split[-2]:
      parent_class = class_split[-2].split('.')[-1].split('~')[-2]
    else:
      parent_class = self.root_class
    return parent_class

  def _create_taxonomy_dicts(self):

    '''This method is used to create the dicts with all the necessary information to generate the new owl file.'''

    classes = self.class_map[~self.class_map['class_path'].str.startswith('postprocessing:')]
    preprocessing_dict = self._adapt_preprocessing_dict()
    postprocessing_mapping_dict = self._adapt_postprocessing_dict()
    self.property_df = self._create_property_df()
    classes.loc[:, 'tree_name'] = classes['class_path'].apply(lambda x: x.split(':')[-2].split('~')[-1])
    classes.loc[:, 'parent_class'] = classes['class_path'].apply(self._get_parent_class)
    classes.loc[:, 'associated_compound_classes'] = classes['class_name'].apply(lambda x: ', '.join(postprocessing_mapping_dict[x]) if x in postprocessing_mapping_dict.keys() else '')
    classes.loc[:, 'preprocessed_from'] = classes['class_name'].apply(lambda x: ', '.join(preprocessing_dict[x]) if x in preprocessing_dict.keys() else '')
    classes = pd.merge(classes, self.property_df, on = 'class_name', how = 'left')
    return classes.apply(lambda x: {'class_name': x['class_name'], 'tree_name': x['tree_name'], 'parent_class': x['parent_class'], 'class_path': x['class_path'],
                                    'associated_compound_classes': x['associated_compound_classes'], 'preprocessed_from': x['preprocessed_from'],
                                    'SNOMED_ID': x['SNOMED_ID'], 'synonyms': x['Synonym'], 'auxiliary_class': x['auxiliary_class'], 'description': x['Description']}, axis = 1).tolist()

  def _create_property_df(self):

    '''This method is used to create a df with the class properties.'''

    self.property_df = pd.DataFrame()
    for class_name in list(self.class_map['class_name']):
      dct = {}
      dct['class_name'] = class_name
      cls = self.rainforest_format_taxonomy.search_one(label = class_name)
      props = cls.get_properties(cls)
      for prop in props:
        if prop.label:
          dct[prop.label[0]] = ', '.join([str(element) for element in prop[cls]])
      self.property_df = pd.concat([self.property_df, pd.DataFrame(dct, index = [len(self.property_df)])])
    self.property_df.fillna('', inplace = True)
    return self.property_df

  ### Methods for creating the new taxonomy

  def _create_output_taxonomy(self):

    '''This method is used to generate the new owl file.'''

    readme_message = f'''This is taxonomy was created using the Multiplex Classification Framework. The structure of the taxonomy is hierarchical, where all classes are grouped into disjoint unions, except for the root class ({self.root_class}).

Meaning of class annotations:
- class_path: String with a unique identifier used in the original file, which includes the information about all the ascendent classes from the given class together with their corresponding trees. ':' is used to separate a tree and its classes, '.' is used to separate a class from its subclass, and '~' is used to separate a class and its subsidiary tree.
- associated_compound_classes: List of compound classes associated with the given class ('_none' is the default value if there is no associated class). An instance belongs to a given associated class if it belongs to all the classes that have such class under this data field. For example, if the classes 'heart' and 'ultrasound' have the associated class 'echocardiogram', then any instance that belongs both to 'heart' and 'ultrasound' should be classified also as 'echocardiogram'.
- preprocessed_from: List of all the classes present in the initial taxonomy, that were merged into the given class during preprocessing ('_none' by default). For example, if the class 'ct_scan' has 'ct' and 'computed_tomography' as 'preprocessed_from', it means that the original taxonomy included such classes, and their names were merged into 'ct_scan' (which may also have been present originally).
- tree_name: This taxonomy was created from a file including different class trees. This field refers to the tree_name present in such original file.'''

    sample_owl_path = "http://multiplex_example.org/my_taxonomy"
    self.output_onto = owlready2.get_ontology(sample_owl_path)
    # The ontology is destroyed and recreated, otherwise there can be errors if the notebook is ran multiple times.
    self.output_onto.destroy(update_relation = True, update_is_a = True)
    self.output_onto = owlready2.get_ontology(sample_owl_path)

    # Properties are created in the OWL file.

    with self.output_onto:
        class tree_name(AnnotationProperty):
            pass

        class class_path(AnnotationProperty):
            pass

        class description(AnnotationProperty):
            pass

        class synonyms(AnnotationProperty):
            pass

        class auxiliary_class(AnnotationProperty):
            pass

        class SNOMED_ID(AnnotationProperty):
            pass

        class associated_compound_classes(AnnotationProperty):
            pass

        class preprocessed_from(AnnotationProperty):
            pass

        class readme(AnnotationProperty):
            pass

        class date(AnnotationProperty):
            pass

        class taxonomy_comment(AnnotationProperty):
            pass

    # The class tree is created. Metadata information is included in the root class. The Thing class is not used as root class because any union assigned to it by using equivalent_to is not reflected in the output owl file.

    with self.output_onto:
      created_classes = {}
      created_classes[self.root_class] = types.new_class(self.root_class, (Thing,))
      created_classes[self.root_class].readme = readme_message
      created_classes[self.root_class].date = str(datetime.today().date())
      created_classes[self.root_class].taxonomy_comment = self.additional_comment
      previous_classes = [{'class_name': self.root_class, 'parent_class': '', 'tree_name': '', 'associated_compound_classes': '', 'preprocessed_from': ''}]
      while True:
        new_classes = []
        for dct in self.onto_dicts:
          if dct['parent_class'] in [dct['class_name'] for dct in previous_classes]:
            new_classes.append(dct)

        previous_classes = new_classes

        for class_dct in new_classes:
          parent_class = created_classes[class_dct['parent_class']]
          created_classes[class_dct['class_name']] = types.new_class(class_dct['class_name'], (parent_class,))
          created_classes[class_dct['class_name']].description = class_dct['description']
          created_classes[class_dct['class_name']].SNOMED_ID = class_dct['SNOMED_ID']
          created_classes[class_dct['class_name']].synonyms = class_dct['synonyms']
          created_classes[class_dct['class_name']].auxiliary_class = class_dct['auxiliary_class']
          created_classes[class_dct['class_name']].tree_name = class_dct['tree_name']
          created_classes[class_dct['class_name']].class_path = class_dct['class_path']
          if class_dct['associated_compound_classes']:
            created_classes[class_dct['class_name']].associated_compound_classes = class_dct['associated_compound_classes']
          else:
            created_classes[class_dct['class_name']].associated_compound_classes = '_none'
          if class_dct['preprocessed_from']:
            created_classes[class_dct['class_name']].preprocessed_from = class_dct['preprocessed_from']
          else:
            created_classes[class_dct['class_name']].preprocessed_from = '_none'

        if len(created_classes) == len(self.onto_dicts)+1:
          break

    # Classes that have the same parent and belong to the same tree are defined as disjoint unions.

    disjoint_union_dicts = pd.DataFrame(self.onto_dicts).groupby(['parent_class', 'tree_name']).agg(list).reset_index()[['parent_class', 'class_name']].apply(lambda x: {'parent': x['parent_class'], 'disjoint_union_of': x['class_name']}, axis = 1).tolist()

    with self.output_onto:

      for dct in disjoint_union_dicts:

        disjoint_list = []
        for cls in dct['disjoint_union_of']:
          disjoint_list.append(created_classes[cls])

        AllDisjoint(disjoint_list)

        if dct['parent'] == 'owl:Thing':
          parent_class = Thing
        else:
          parent_class = created_classes[dct['parent']]

        union_expression = created_classes[dct['disjoint_union_of'][0]]
        for cls in dct['disjoint_union_of'][1:]:
          union_expression |= created_classes[cls]

        parent_class.equivalent_to.append(union_expression)

    # The taxonomy is saved.

    self.output_onto.save(self.output_owl_path)

class MultiplexDatasetProcessor():

  def __init__(self, input_owl_path, input_csv_path, input_label_column = 'label_list', output_csv_path = '', output_format = 'multiplex',
               exclusion_classes = False, refresh_ontology = True, output_owl_path = 'my_taxonomy.owl', root_class = 'root_class', additional_comment = 'No additional comment.'):

    '''
    The MultiplexDatasetProcessor class is used to adapt a csv with annotated data considering an owl file containing a multiplex taxonomy. In the process, the quality of annotated data is improved by removing any incompatible labels.

    input_owl_path (str): Path to the taxonomy file.
    input_csv_path (str): Path to the input dataset file.
    input_label_column (str, default 'label_list'): Name of the column containing the list of labels per row.
    output_csv_path (str, default 'dataset_for_multiplex_classification.csv'): Path to the output dataset file.
    output_format (str, default 'multiplex'): Format of the output dataset.
      - 'multiplex': One column per classification task. Tasks that correspond to the same conditioning class are merged into a single column (to train multitask classifiers).
      - 'multiplex_without_merging': One column per classification task (no column merging is applied).
      - 'multilabel': All labels are included in one single column.
    exclusion_classes (bool or list, default False): If a list of labels is provided, they will be considered to be the default value for the corresponding classification task if a certain classified row has no specific label available.
        For example, if exclussion_classes = ['cat'] is provided, then in the classification task that contains the class 'cat' that class will be used for any instance that does not have any label for that task.
        Default labels are applied only to rows that belong to the parent class from the given default class.
        If True, any label called 'exclusion_class' or any class that starts with 'no_' or 'other_' will be considered to be an exclusion class.
    refresh_ontology (bool, default True) When more than one taxonomy is used in the same runtime, previous uploaded ontology objects need to be destroyed.
    output_owl_path (str, default 'my_taxonomy.owl'): Path to the output taxonomy file.
    root_class (str, default 'root_class'): Root class of the taxonomy.
    additional_comment (str, default 'No additional comment.'): Additional comment to be included in the output taxonomy file.
    '''

    self.mtp = MultiplexTaxonomyProcessor(input_owl_path = input_owl_path, output_owl_path = output_owl_path, root_class = root_class, additional_comment = additional_comment, refresh_ontology = refresh_ontology)
    self.input_csv_path = input_csv_path
    self.input_label_column = input_label_column
    if output_csv_path:
     self.output_csv_path = output_csv_path
    else:
      self.output_csv_path = 'dataset_for_multiplex_classification.csv'
    self.output_format = output_format
    self.exclusion_classes = exclusion_classes

    self.dataset = self._create_dataset()
    if self.output_csv_path:
      self.dataset.to_csv(self.output_csv_path, index = False)

  ### Main Method for Dataset Creation

  def _create_dataset(self):

    '''This method creates the dataset.'''

    # Checking if the format of the input dataset is correct.

    data = pd.read_csv(self.input_csv_path)
    if self.input_label_column not in data.columns:
      raise ValueError(f"Input label column '{self.input_label_column}' not found in dataframe columns: {data.columns}")
    if not data[self.input_label_column].apply(lambda x: isinstance(x, list)).all():
      try: # In case label lists are in string format.
        data[self.input_label_column] = data[self.input_label_column].apply(ast.literal_eval)
      except:
        raise ValueError(f"The format of the input label column is incorrect (it should contain lists of strings).")

    data = data.explode(self.input_label_column)

    # Data preprocessing according to the preprocessing class mapping included in the .owl file.
    data['label'] = data[self.input_label_column].map(self._preprocess)
    data.drop(self.input_label_column, axis = 1, inplace = True)
    data = data.explode('label')

    data = data.merge(self.mtp.class_map[['class_name', 'class_path']], left_on='label', right_on='class_name', how = 'left').drop(['label', 'class_name'], axis = 1)

    data = data.drop_duplicates()
    data.fillna('', inplace=True)

    # # If the same instance has multiple label, this part of the code makes sure that they are compatible.
    data = data.groupby([c for c in data.columns if c != 'class_path']).agg(list).reset_index()

    data['class_path'] = data['class_path'].apply(self._get_compatible_values)
    self.initial_columns = [c for c in data.columns if c != 'class_path']

    column_replacement_dict = {new_name[:-(len('~main'))]: new_name for new_name in self.mtp.class_map[self.mtp.class_map['column_name'].str.endswith('~main')]['column_name'].unique()}

    data['class_path'] = data['class_path'].apply(lambda x: self._replace_keys(x, column_replacement_dict))

    column_names = sorted(self.mtp.class_map.column_name.unique())
    data[column_names] = None

    data = data.apply(lambda x: self._add_dict_values(x), axis = 1)

    data.drop('class_path', axis = 1, inplace = True)

    # Missing labels are replaced by exclusion classes.
    if self.exclusion_classes:
      exclusion_dicts = self._get_exclusion_classes()
      data = data.apply(lambda x: self._replace_by_exclusion_class(x, exclusion_dicts), axis = 1)

    # Output format selection.
    if self.output_format == 'multiplex':
      outcome_df = self._merge_columns(data).copy()
    elif self.output_format == 'multiplex_without_merging':
      outcome_df = data.copy()
    elif self.output_format == 'multilabel':
      label_columns = [c for c in data.columns if c not in self.initial_columns]
      data['label_list'] = data.apply(lambda x: self._create_multilabel_column(x, label_columns), axis=1)
      data.drop(label_columns, axis=1, inplace=True)
      outcome_df = data.copy()
    else:
      outcome_df = self._merge_columns(data).copy()

    self.label_columns = [c for c in outcome_df.columns if c not in self.initial_columns]
    return outcome_df

  ### Auxiliary Methods

  def _preprocess(self, value):

    ''' This method replaces labels considering the preprocessing mapping dict.'''

    if value in self.mtp.preprocessing_mapping_dict:
      return self.mtp.preprocessing_mapping_dict[value]
    else:
      return [value]

  def _path_to_dict(self, string):

    '''Given a string with a class_path, this method creates a dictionary with the form {'classification_task': 'corresponding_class'}.'''

    keys = []
    values = []

    for i, char in enumerate(string):
      if char == ':':
        keys.append(string[:i])
        value_start = i+1
      elif char == '.':
        keys.append(string[:i])
        if value_start:
          values.append(string[value_start:i])
        value_start = i+1
      elif char == '~':
        values.append(string[value_start:i])
        value_start = None
      if i+1 == len(string):
        values.append(string[value_start:])

    return dict(zip(keys, values))

  def _get_compatible_values(self, path_list):

    '''Given a list of class_paths, this method returns a filtered dictionary (without incompatibilities).'''

    path_dicts = []
    for path in path_list:
      path_dicts.append(self._path_to_dict(path))

    merged_dict = {}
    for dct in path_dicts:
      for key in dct.keys():
        if key not in merged_dict.keys():
          merged_dict[key] = set()
        merged_dict[key].add(dct[key])

    error_tasks = []
    for key in merged_dict.keys():
      if len(merged_dict[key]) > 1:
        error_tasks.append(key)

    correct_keys = []
    for key in merged_dict.keys():
      errors = 0
      for error_task in error_tasks:
        if key.startswith(error_task):
          errors += 1
      if errors == 0:
        correct_keys.append(key)

    filtered_dict = {k:list(v)[0] for k,v in merged_dict.items() if k in correct_keys}

    return filtered_dict

  def _replace_keys(self, dict_a, dict_b):

    '''This method is used to replace the keys of one dictionary considering the keys of another one.'''

    return {dict_b.get(k, k): v for k, v in dict_a.items()}

  def _add_dict_values(self, row):

    '''This method is used to add the values from the class_path dictonaries to the corresponding dataframe columns.'''

    for k, v in row['class_path'].items():
      row[k] = v
    return row

  def _create_exclusion_dictionary(self, class_path):

    '''Given the class_path from an exclusion class, this method returns a dictionary to use such class as a default value.'''

    dct = {}
    dct['exclusion_class'] = re.split('~|:|\.', class_path)[-1]
    dct['column_name'] = class_path[:-(len(dct['exclusion_class'])+1)]
    auxiliary_string = dct['column_name'].replace(':','.')
    if '.' in auxiliary_string:
      dct['conditioning_column'] = dct['column_name'][:auxiliary_string.rindex('.')]
      dct['conditioning_class'] = re.split('~|:|\.', dct['column_name'][len(dct['conditioning_column'])+1:])[0]
    else:
      dct['conditioning_column'] = ''
      dct['conditioning_class'] = ''
    return dct

  def _get_exclusion_classes(self):

    '''This method is used to get all the exclusion dictionaries.'''

    if isinstance(self.exclusion_classes, list):
      filtering_conditions = (self.mtp.class_map.class_name in self.exclusion_classes)
    else:
      filtering_conditions = (self.mtp.class_map.class_name == 'exclusion_class')|(self.mtp.class_map.class_name.str.startswith('other_'))|(self.mtp.class_map.class_name.str.startswith('no_'))
    exclusion_class_paths = []
    for class_path in self.mtp.class_map[filtering_conditions]['class_path']:
      exclusion_class_paths.append(self._create_exclusion_dictionary(class_path))
    return exclusion_class_paths

  def _replace_by_exclusion_class(self, row, exclusion_dicts):

    '''This method is used to replace missing labels by the corresponding exclusion label.'''

    for dct in exclusion_dicts:
      if (row[dct['column_name']] == '') or (row[dct['column_name']] is np.nan):
        if (dct['conditioning_column'] == '') or ((dct['conditioning_column'] in row.keys()) and (row[dct['conditioning_column']] == dct['conditioning_class'])):
          row[dct['column_name']] = dct['exclusion_class']
    return row

  def _create_merging_dict(self, columns):

    '''This method is used to create a dictionary with the values needed to merge columns that correspond to the same conditioning class.'''

    conditioning_class_df = self.mtp.class_map[['conditioning_class', 'column_name']].groupby('conditioning_class').agg(set).reset_index()
    conditioning_class_df = conditioning_class_df[conditioning_class_df['column_name'].apply(lambda x: len(x) > 1)]
    dictionary = dict(zip(conditioning_class_df['conditioning_class'], conditioning_class_df['column_name'].apply(lambda x: list(x))))

    outcome_dict = {}
    for key in dictionary.keys():
      strings = []
      for col in dictionary[key]:
        strings.append(col[len(key)+1:])
      new_key = key + '(' + '|'.join(strings) + ')'
      outcome_dict[new_key] = dictionary[key]

    return outcome_dict

  def _merge_columns(self, df):

    '''This method is used to merge columns that correspond to the same conditioning class for multitask classifiers.'''

    merging_dict = self._create_merging_dict(df.columns)

    for new_column, old_columns in merging_dict.items():
      df[new_column] = df[old_columns].apply(list, axis=1)

    for key in merging_dict.keys():
      df.drop(merging_dict[key], axis=1, inplace=True)

    return df

  def _create_multilabel_column(self, row, label_columns):

    '''This method is used to merge all the columns with labels in a single multilabel column.'''

    col_list = [row[col] for col in label_columns]
    outcome_list = []
    for l in col_list:
      if isinstance(l, list):
        for element in l:
          if element:
            outcome_list.append(element)
      elif l:
        outcome_list.append(l)

    return outcome_list

  def _add_compound_classes(self, label_list, postprocessing_dict):

    '''This method is used to add compound classes to a label list if all the corresponding component labels are present.'''

    outcome_list = label_list.copy()
    for key in postprocessing_dict.keys():
      condition = True
      for value in postprocessing_dict[key]:
        if value not in label_list:
          condition = False
      if condition:
        outcome_list.append(key)
    return outcome_list

  def apply_postprocessing(self):

    '''This method is used to add compound classes to the dataset. The label_list column is created if absent, and a new column with compound classes is added (postprocessed_label_list).'''

    self.dataset['label_list'] = self.dataset.apply(lambda x: self._create_multilabel_column(x, self.label_columns), axis=1)
    postprocessing_dict = {key: value for d in self.mtp.postprocessing_class_map.apply(lambda x: {x['class_name']: x['class_path'].split(':')[-1].split('+')}, axis = 1) for key, value in d.items()}
    self.dataset['postprocessed_label_list'] = self.dataset['label_list'].apply(lambda x: self._add_compound_classes(x, postprocessing_dict))
