# opensubtitles-dataloader

Download, preprocess and use sentences from the [OpenSubtitles v2018 dataset](http://opus.nlpl.eu/OpenSubtitles-v2018.php) without ever needing to load all of it into memory.

## Download
See possible languages [here](http://opus.nlpl.eu/OpenSubtitles-v2018.php).
````bash
opensubtitles-download en
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
opensubtites_dataset = OpenSubtitlesDataset('en', n_sents=5)
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
Per default, the entries in splits are sorted by length to make batching easier, this can be turned off.
````python
train, valid, test = opensubtites_dataset.split(sort_by_len=False)
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
