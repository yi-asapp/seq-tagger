# ASAPP-SeqTagger: a library for neural sequence labeling

This repository forms a light-weight library for neural sequence labeling. Given a list of tokens, a sequence tagging model will generate a list of tags, each tag corresponds an input token, by leveraging the dependencies between the tags.

A common use case for sequence labeling is the problem of named entity recognition (NER). Assuming the input sentence is "ASAPP is a startup company in New York City", the NER model needs to recognize that 'ASAPP' is an ORGANIZATION and 'New York City' is a LOCATION. The problem can be formalized as a sequence tagging problem by encoding each token into one of the following types of tags - _B-ENTITY, I-ENTITY, and O_. The tags correpsonding to the example above are _B-ORG_ for 'ASAPP', _B-LOC_ for 'New', and _I-LOC_ for 'York' and 'City'.

## Model

| Main file | Dependencies | Description |
| ------------ | ------------ | ------------ |
| tagger | None | A word-level BiLSTM model. Ignore tag-tag dependencies. |
| char_tagger | None | A character-level LSTM and word-level BiLSTM model. Ignore tag-tag dependencies. |
| crf_tagger | crf | A word-level BiLSTM model. A global CRF model is used to model tag-tag dependencies. |
| crf_char_tagger | crf | A character-level LSTM and word-level BiLSTM model. A global CRF model is used to model tag-tag dependencies. |
| ssvm_char_tagger | ssvm | A character-level LSTM and word-level BiLSTM model. A global SSVM model is used to model tag-tag dependencies. |

## Data

### CoNLL-2003 data

| TYPE | DESCRIPTION |
| ----- | ----- |
| PER | Named person or family. |
| LOC | Name of politically or geographically defined location (cities, provinces, countries, international regions, bodies of water, mountains). |
| ORG | Named corporate, governmental, or other organizational entity. |
| MISC | Miscellaneous entities, e.g. events, nationalities, products or works of art. |

### OntoNotes 5 data

| TYPE | DESCRIPTION |
| ----- | ----- |
| PERSON | People, including fictional. |
| NORP | Nationalities or religious or political groups. |
| FAC | Buildings, airports, highways, bridges, etc. |
| ORG | Companies, agencies, institutions, etc. |
| GPE | Countries, cities, states. |
| LOC | Non-GPE locations, mountain ranges, bodies of water. |
| PRODUCT | Objects, vehicles, foods, etc. (Not services.) |
| EVENT | Named hurricanes, battles, wars, sports events, etc. |
| WORK_OF_ART | Titles of books, songs, etc. |
| LAW | Named documents made into laws. |
| LANGUAGE | Any named language. |
| DATE | Absolute or relative dates or periods. |
| TIME | Times smaller than a day. |
| PERCENT | Percentage, including "%". |
| MONEY | Monetary values, including unit. |
| QUANTITY | Measurements, as of weight or distance. |
| ORDINAL | "first", "second", etc. |
| CARDINAL | Numerals that do not fall under another type. |
