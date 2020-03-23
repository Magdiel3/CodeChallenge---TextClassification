# CodeChallenge - TextClassification

Classify a single str entry or a JSON formated data with labels
  1. soft: for soft skills like 'team player'
  2. tech: for tech skills like'python experience'
  3. none: for all other type of sentences

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for evaluation purposes. If something is not clear or doesn't work, please refer to [me](mailto:magdieltercero@hotmail.com)

### Prerequisites

Following libraries needed

```
import argparse
import os
import sys
import numpy
import keras
import pickle
import pyperclip
import json
import pandas
import spacy
```

### Installing

Pls download (clone) this repo to a desired Folder

With bash or cmd navigate to the folder where the repo is found on your Computer

```
(base) C:\Users\magdi\OneDrive\Documents\VSCode\Python\Talentbait\CodeChallenge---TextClassification>
```

Install the packages

```sh
install.sh
```

You should be good to go

## Running the tests

To call the library just perform: 

```
(base) C:\Users\magdi\OneDrive\Documents\VSCode\Python\Talentbait\CodeChallenge---TextClassification>textClassifierCLI
```

For help just ge with ```textClassifierCLI -h``` and you'll get:

```
usage: textClassifierCLI [-h] [-s] [-j] [-p]

---------------------------------------------------------------
Description:
Classify a single str entry or a JSON formated data with labels
    1. soft: for soft skills like 'team player'
    2. tech: for tech skills like'python experience'
    3. none: for all other type of sentences
---------------------------------------------------------------

optional arguments:
  -h, --help    show this help message and exit
  -s , --str    may be used to classify a single Text in prompt
  -j, --json    state that a JSON file will be classified
  -p , --path   path to JSON file (may contain labes)
                default='tech_soft_none.json'
```

There are *3* main functionalities:
1. label a string in prompt
```
C:\Users\>textClassifierCLI -s <string_to_classify>
```
2. label the DataSet in a generated output file (useless but for development)
```
C:\Users\>textClassifierCLI -j
```
3. label a json in a generated output file
```
C:\Users\>textClassifierCLI -j -p <absolute_path>
```
It works as well with a relative path if within folder

## Authors

* **Magdiel Martinez** - *Initial work* - [Magdiel3](https://github.com/Magdiel3)

## License

This project is licensed under the MIT License - see the [LICENSE.md](LICENSE.md) file for details

## Acknowledgments

* Thanks stackoverflow, python docs, keras docs, and all other documentations
* Purpose: Apply for an awesome job 
* Motivation: Learn (I really did) even tough I may not get the Job unu
