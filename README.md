# opensubtitles-dataloader
[![PyPI version](https://badge.fury.io/py/opensubtitles-dataloader.svg)](https://badge.fury.io/py/opensubtitles-dataloader)

Download, preprocess and use sentences from the [OpenSubtitles v2018 dataset](http://opus.nlpl.eu/OpenSubtitles-v2018.php) without ever needing to load all of it into memory.

## Download
See possible languages [here](http://opus.nlpl.eu/OpenSubtitles-v2018.php).
````bash
opensubtitles-download en
````
Load tokenized version.
````bash
opensubtitles-download en --token
````

## Use in Python
### Load
````python
opensubtites_dataset = OpenSubtitlesDataset('en')
````
Load only the first 1 million lines.
````python
opensubtites_dataset = OpenSubtitlesDataset('en', first_n_lines=1_000_000)
````
Group sentences into groups of 5.
````python
opensubtites_dataset = OpenSubtitlesDataset('en', 5)
````
Group sentences into groups ranging from 2 to 5.
````python
opensubtites_dataset = OpenSubtitlesDataset('en', (2,5))
````
Split sentences using "\n".
````python
opensubtites_dataset = OpenSubtitlesDataset('en', delimiter="\n")
````
Do preprocessing.
````python
opensubtites_dataset = OpenSubtitlesDataset('en', preprocess_function=my_preprocessing_function)
````
### Split for Training
````python
train, valid, test = opensubtites_dataset.split()
````
Set the fractions of the original dataset.
````python
train, valid, test = opensubtites_dataset.split([0.7, 0.15, 0.15])
````
Use a seed.
````python
train, valid, test = opensubtites_dataset.split(seed=42)
````
### Access
index.
````python
train, valid, text = OpenSubtitlesDataset('en').splits()
train[20_000]
````
pytorch.
````python
from torch.utils.data import DataLoader
train, valid, text = OpenSubtitlesDataset('en').splits()
train_loader = DataLoader(train, batch_size=16)
````
