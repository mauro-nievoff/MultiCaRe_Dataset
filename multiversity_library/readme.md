# The `multiversity` package

This folder contains all the code from the `multiversity` Python package, which was used for the creation of the MultiCaRe dataset:
- multi_labeler.py: code to create labels based on captions.
- multiplex_classification.py: code for handling the class taxonomy and for working with complex classification problem that involve both sequential and simultanous classification tasks.
- multicare_creator.py: code to create or upload the MultiCaRe dataset.
- multicare_dataset.py: code to create subsets based on the MultiCaRe dataset, according to specific use cases.
- multicare_v1: code to use the previous version of the dataset (V1.0).

The whole dataset can be downloaded running the cell below. The process takes some days (if it is stopped and resumed, the process starts from the last checkpoint).

```python
!pip install multiversity

from multiversity.multicare_creator import *

mc = MulticareCreator(email = 'sample@email.com', # NCBI account email
                      api_key = 'sample_key') # NCBI account API key

mc.download_dataset()
```
