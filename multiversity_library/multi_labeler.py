import re
import pandas as pd

class CaptionNGrammer():
  def __init__(self, caption_csv_path, caption_column = 'caption', output_path = '', text_segmentation_characters = ['!', '(', ')', ',', '.', ':', ';', '?', '[', ']','{', '}'], token_segmentation_characters = [' ', '\n'],
               boundary_stop_words = ['the', 'of', 'and', 'in', 'a', 'to', 'on', 'after', 'is', 'for', 'from', 'by', 'were', 'are', 'an', 'was'], any_stop_word_capitalization = True, max_n = 5, minimum_ngram_count = 25):

    '''
    This class is used to turn an input dataframe with captions into a dataframe with all their ngrams. For example, given the caption "The sky is blue", the output will include the following ngrams: "The", "sky", "is", "blue", "The sky", "sky is", "is blue", "The sky is", "sky is blue" and "The sky is blue".

    params:
    caption_csv_path (str): path to the csv file containing the captions.
    caption_column (str): name of the column containing the captions. By default, it's 'caption'.
    output_path (str): path to the output csv file with the ngrams.
    text_segmentation_characters (list): list of characters that will be used to split the text into segments. By default, it's ['!', '(', ')', ',', '.', ':', ';', '?', '[', ']','{', '}'].
      The concept of text segmentation is similar to sentence splitting, but broader. Text is split using separators that are never included inside relevant ngrams (e.g. '?' or '{' are never found in extractions related to clinical findings).
      If numerical values such as '2.5' are relevant to a given use case, '.' can be replaced by '. '.
    token_segmentation_characters (list): list of characters that will be used to split the text into tokens (grams). By default, it's [' ', '\n'].
    boundary_stop_words (list): words that will be used to filter out ngrams that start or end with them. By default, it's ['the', 'of', 'and', 'in', 'a', 'to', 'on', 'after', 'is', 'for', 'from', 'by', 'were', 'are', 'an', 'was'].
      ngrams starting or ending with these stop words are filtered out as a way to reduce the amount of irrelevant ngrams from the final output.
    any_stop_word_capitalization (bool): If True, stop words are considered disregard their capitalization. By default, it's True.
    max_n (int): maximum length of the ngrams. By default, it's 5.
    minimum_ngram_count (int): minimum number of times an ngram must appear to be included in the final output. By default, it's 25.
    '''

    self.caption_df = pd.read_csv(caption_csv_path)
    self.caption_column = caption_column
    self.output_path = output_path
    self._full_text = ' '.join(self.caption_df[self.caption_column].unique()) # All the captions are merged into the same string.
    self.special_characters = set(re.findall(r"[^a-zA-Z0-9\s]", self._full_text))  # All the special characters present in the full string are identified.
    self.text_segmentation_characters = text_segmentation_characters
    self.token_segmentation_characters = token_segmentation_characters
    if any_stop_word_capitalization:
      self.boundary_stop_words = list(set([word.lower() for word in boundary_stop_words] + [word.capitalize() for word in boundary_stop_words] + [word.upper() for word in boundary_stop_words]))
    else:
      self.boundary_stop_words = boundary_stop_words
    self.max_n = max_n
    self.minimum_ngram_count = minimum_ngram_count

    self.token_df = self._create_count_df(char_list_for_segmentation = self.text_segmentation_characters + self.token_segmentation_characters, column_name = 'token') # A df with all the tokens is created.
    self.segment_df = self._create_count_df(char_list_for_segmentation = self.text_segmentation_characters, column_name = 'text_segment') # A df with all the segments is created.
    self.ngram_df = self._create_ngram_df()
    if self.output_path:
      self.ngram_df.to_csv(self.output_path, index=False)

  ### Auxiliary Methods

  def _create_count_df(self, char_list_for_segmentation, column_name = 'text'):

    '''
    Method used to split a text considering a list of segmentation chars, and then create a df with value counts.
    char_list_for_segmentation (list): list of characters that will be used to split the text into segments.
    column_name (str): name of the column in the output df.
    '''

    text_parts = self._split_text_by_chars(self._full_text, char_list_for_segmentation)
    count_df = pd.DataFrame(text_parts, columns=[column_name])
    count_df['count'] = 1
    count_df = count_df.groupby(column_name).count().reset_index().sort_values(by='count', ascending=False)
    return count_df

  def _split_text_by_chars(self, text, char_list):

    '''
    Method used to split a text considering a list of segmentation chars.
    text (str): text to be split.
    char_list (list): list of characters that will be used to split the text.
    '''
    text_parts = []
    new_part = ''

    for char in text:
      if char in char_list:
        text_parts.append(new_part.strip())
        new_part = ''
      else:
        new_part += char
    text_parts.append(new_part.strip())

    return text_parts

  def _generate_ngrams(self, text):

    '''
    Method used to generate all the ngrams for a given text.
    text (str): text for which the ngrams will be generated.
    '''

    text_tokens = self._split_text_by_chars(text, self.token_segmentation_characters)
    amount_of_tokens = len(text_tokens)

    ngram_dicts = []
    for i in range(amount_of_tokens):
      for j in range(amount_of_tokens):
        if i<=j and j-i<self.max_n:
          ngram_tokens = text_tokens[i:j+1]
          if ngram_tokens[0] not in self.boundary_stop_words and ngram_tokens[-1] not in self.boundary_stop_words:
            ngram_dicts.append({'ngram': ' '.join(text_tokens[i:j+1]), 'n': j-i+1})
    return ngram_dicts

  def _create_ngram_df(self):

    '''
    Method used to create a df with ngrams and their counts.
    '''

    ngram_df = self.segment_df.copy()
    ngram_df['ngrams'] = ngram_df.apply(lambda x: self._generate_ngrams(x['text_segment']), axis = 1)
    ngram_df = ngram_df.explode('ngrams')
    ngram_df = ngram_df[~ngram_df['ngrams'].isna()]
    ngram_df['ngram'] = ngram_df['ngrams'].apply(lambda x: x['ngram'])
    ngram_df['n'] = ngram_df['ngrams'].apply(lambda x: x['n'])
    ngram_df.drop(columns=['text_segment', 'ngrams'], inplace=True)
    ngram_df = ngram_df.groupby(['ngram', 'n']).sum().reset_index().sort_values(by='count', ascending=False).copy()
    ngram_df = ngram_df[ngram_df['count']>self.minimum_ngram_count].copy()
    return ngram_df


class CaptionLabeler():

  def __init__(self, caption_csv_path, annotated_ngrams_csv_path, caption_column = 'caption', label_columns = [], ngram_column = 'ngram', block_label = '_LABEL_BLOCK'):

    '''
    Class used to label captions based on a list of annotated ngrams.

    params:
    caption_csv_path (str): path to the csv file containing the captions.
    annotated_ngrams_csv_path (str): path to the csv file containing the annotated ngrams.
    caption_column (str): name of the column containing the captions. By default, it's 'caption'.
    label_columns (list): list of columns containing the labels. By default, it's [] (all the columns that start with 'label' will be considered).
    ngram_column (str): name of the column containing the ngrams. By default, it's 'ngram'.
    block_label (str): label used to indicate that a label block should be removed from the final output. By default, it's '_LABEL_BLOCK'.
      This label is used in order to avoid labeling irrelevant ngrams. For instance, if 'mass' is labeled as Clinical_Finding, and 'no mass' is labeled as '_LABEL_BLOCK', then 'no mass' will not be labeled as Clinical_Finding.
    '''

    self.caption_df = pd.read_csv(caption_csv_path)
    self.caption_column = caption_column

    self.annotated_ngrams_df = pd.read_csv(annotated_ngrams_csv_path)
    if label_columns:
      self.label_columns = label_columns
    else:
      self.label_columns = [col for col in self.annotated_ngrams_df.columns if col.lower().startswith('label')] # If no label columns are defined, any column starting with 'label' is considered.
    self.annotated_ngrams_df['labels'] = self.annotated_ngrams_df.apply(lambda x: self._get_label_list(x, self.label_columns), axis=1)
    self.annotated_ngrams_df.drop(self.label_columns, axis = 1, inplace = True)
    self.annotated_ngrams_df = self.annotated_ngrams_df[self.annotated_ngrams_df['labels'].apply(lambda x: len(x) > 0)]
    self.annotation_dict = dict(zip(self.annotated_ngrams_df[ngram_column], self.annotated_ngrams_df['labels']))

    self.all_chars = list(set(''.join(set(self.caption_df[self.caption_column].tolist())))) # List with all chars.
    self.special_characters = [char for char in self.all_chars if char.isalnum() == False] # List with all special chars.
    self.text_segmentation_characters = self._filter_characters(string_list = self.annotation_dict.keys(), char_list = self.special_characters)

    # Chars present in the boundary of extractions are considered when defining the list of characters used for token segmentation.
    self.boundary_chars = [item for sublist in [[k[0], k[-1]] for k in self.annotation_dict.keys()] for item in sublist]
    self.token_segmentation_characters = self._filter_characters(string_list = self.boundary_chars, char_list = list(set(self.all_chars) - set(self.text_segmentation_characters)))

    self.caption_df['label_list'] = self.caption_df[self.caption_column].apply(lambda x: self._extract_labels(x))

    self.labels = self.caption_df.label_list.explode().unique()

  ### Auxiliary Methods

  def _get_label_list(self, row, label_columns):

    '''
    Method used to merge all the labels present in different columns into a single list.
    row (pd.Series): row of the dataframe.
    label_columns (list): list of columns containing the labels.
    '''

    label_list = []
    for label in label_columns:
      if type(row[label]) is str:
        label_list.append(row[label])
    return label_list

  def _filter_characters(self, string_list, char_list = [], alnum_condition = True):

    '''
    Method used to filter out characters included in a given list of characters.
    string_list (list): list of strings.
    char_list (list): list of characters to be filtered.
    alnum_condition (bool): if True (default value), only non-alphanumeric characters will be considered.
    '''

    char_set = set(''.join(set(string_list)))

    if char_list == []:
      char_list = list(char_set)

    filtered_characters = []
    for char in char_list:
      if char not in char_set:
        filtered_characters.append(char)

    if alnum_condition:
      filtered_characters = [char for char in filtered_characters if char.isalnum() == False]

    return filtered_characters

  def _extract_labels(self, caption, filter_overlapping_extractions = True, ignore_uppercase = True):

    '''
    Method used to assign labels to a given caption considering its contents.
    caption (str): caption to be labeled.
    filter_overlapping_extractions (bool): if True (default value), overlapping extractions will be filtered out.
    ignore_uppercase (bool): if True (default value), text casing will be ignored.
    '''

    extraction_list = []
    for key in self.annotation_dict.keys():
      if ignore_uppercase:
        caption = caption.lower()
        key = key.lower()
      indices = self._find_all_indices(caption, extraction = key, labels = self.annotation_dict[key])
      for index in indices:
        extraction_list.append(index)

    if filter_overlapping_extractions:
      valid_extraction_list = self._filter_extraction_list(caption, extraction_list)
    else:
      valid_extraction_list = extraction_list.copy()

    caption_extractions = set()
    caption_labels = set()
    for extraction in valid_extraction_list:
      caption_extractions.add(extraction['extraction'])
      for label in extraction['labels']:
        caption_labels.add(label)

    return list(caption_labels)

  def _find_all_indices(self, input_text, extraction, labels, start=0, indices=None):

    '''
    Method used to find the indices of a substring (extraction) in a given input text.
    input_text (str): main string where the substring will be found.
    extraction (str): substring to be found.
    labels (list): label associated with a given extraction.
    start (int): start index, 0 by default.
    indices (list): list of indices, None by default.
    '''

    if indices is None:
      indices = []
    start_index = input_text.find(extraction, start)
    if start_index == -1:
      return indices
    end_index = start_index + len(extraction)
    if ((start_index == 0) or (input_text[start_index-1] in self.text_segmentation_characters + self.token_segmentation_characters)) and ((end_index == len(input_text)) or (input_text[end_index] in self.text_segmentation_characters + self.token_segmentation_characters)):
      indices.append({'extraction': extraction, 'start': start_index, 'end': end_index, 'labels': labels})
    return self._find_all_indices(input_text, extraction, labels, end_index, indices)

  def _filter_extraction_list(self, input_text, extraction_list):

    '''
    Method used to filter out overlapping extractions, so that each token in the string is assigned only one label. Longer extractions are preferred over shorter ones.
      For example, if the extraction 'lung' is considered as Body_Part, and 'lung cancer' is considered as Clinical_Finding, only 'lung cancer' will be considered.
    input_text (str): main string where the substring will be found.
    extraction_list (list): list of extractions to be filtered.
    '''

    reduced_extraction_list = extraction_list.copy()
    last_end_index = 0
    valid_extraction_list = []

    for char_i in range(len(input_text)):
      if char_i < last_end_index:
        pass
      else:
        current_extraction_list = []
        remaining_extraction_list = []
        for extraction in reduced_extraction_list:
          if extraction['start'] == char_i:
            current_extraction_list.append(extraction)
          elif extraction['start'] > char_i:
            remaining_extraction_list.append(extraction)

        if current_extraction_list:
          last_end_index = max([e['end'] for e in current_extraction_list])
          valid_extraction_list.append([e for e in current_extraction_list if e['end'] == last_end_index][0])
        reduced_extraction_list = remaining_extraction_list.copy()
    return valid_extraction_list

  ### Label Postprocessing Methods

  def _remove_block_label(self, label_list, block_label):

    '''
    Method used to filter out any extraction that has the block label assigned.
    label_list (list): list of labels.
    block_label (str): label that will be removed.
    '''

    new_list = []
    for label in label_list:
      if label != block_label:
        new_list.append(label)
    return new_list

  def _remove_incompatible_labels(self, label_list, label_group_1 = [], label_group_2 = []):

    '''
    Method used to remove labels that are incompatible. If a label from group 1 and a label from group 2 are present in a given label list, they are both removed.
    label_list (list): list of labels.
    label_group_1 (list): list of labels from group 1.
    label_group_2 (list): list of labels from group 2.
    '''

    new_list = []

    incompatibility_bool_1 = False
    incompatibility_bool_2 = False

    for l in label_list:
      if l in label_group_1:
        incompatibility_bool_1 = True
      elif l in label_group_2:
        incompatibility_bool_2 = True

    if incompatibility_bool_1 and incompatibility_bool_2:
      for l in label_list:
        if l not in label_group_1 and l not in label_group_2:
          new_list.append(l)
    else:
      new_list = label_list.copy()

    return new_list

  def _check_label_requirements(self, label_list, checked_label_group = [], required_label_group = []):

    '''
    Method used to remove labels if a given label list does not include any of the labels from the required group.
    label_list (list): list of labels.
    checked_label_group (list): list of labels that will be checked.
    required_label_group (list): list of labels that must be included (at least one) in order to keep the checked label.
    '''
    new_list = []

    met_requirement = False
    for l in label_list:
      if l in required_label_group:
        met_requirement = True

    for l in label_list:
      if (l not in checked_label_group) or met_requirement:
        new_list.append(l)

    return new_list

  def remove_label(self, label):

    '''
    Method used to apply _remove_block_label to a Pandas series.
    '''

    self.caption_df['label_list'] = self.caption_df['label_list'].apply(lambda x: self._remove_block_label(x, label))

  def remove_incompatibilities(self, group_1, group_2):

    '''
    Method used to apply _remove_incompatible_labels to a Pandas series.
    '''

    self.caption_df['label_list'] = self.caption_df['label_list'].apply(lambda x: self._remove_incompatible_labels(x, group_1, group_2))

  def check_requirements(self, checked_group, required_group):

    '''
    Method used to apply _check_label_requirements to a Pandas series.
    '''

    self.caption_df['label_list'] = self.caption_df['label_list'].apply(lambda x: self._check_label_requirements(x, checked_group, required_group))
