# KGCV_Strawberry datasets
## Image data
The image data can be downloaded from https://doi.org/10.5281/zenodo.10957909
### Object and phenology detection
The folder "strawberry_img_random.zip" contains images and the corresponding JSON labels for object and phenological stages detection.

### Fruit size and decimal phenological stage
The folder "strawberry_img_tagged.zip" contains images and the corresponding JSON labels for fruit size and decimal phenological stages detection.

For example,
"label": "small g, 8.84, 7.62, 0.4",
This label means the fruit has an 8.84mm diameter and 7.62mm length, 
with the main stage being small green and the decimal stage being DS-4
Merge and split Data
A Python script, "datasetProcessing.py", can be used to merge and split the image data into training and testing set.

## Measurements
### Plant traits measurements
The folder "measurement.zip" includes treatment-level and fruit-level ground truth data. 

### Treatment-level
```
data_dryMatter_2022.csv
data_dryMatter_2023.csv
data_freshMatter_2022.csv
data_freshMatter_2023.csv
data_fruitNumber_2022.csv
data_fruitNumber_2023.csv
data_plantBiomass_2022.csv
data_plantBiomass_2023.csv
```
### Fruit-level
Fruit conditions with five classes, 1-5 represent Normal, Wizened, Malformed, Wizened & Malformed, and Overripe, respectively.
```
data_size_freshWeight_condition_2022_0N.csv
data_size_freshWeight_condition_2022_50N.csv
data_size_freshWeight_condition_2022_100N.csv
data_size_freshWeight_condition_2022_150N.csv
```
Fruit size for tagged fruits
```
data_taggedFruit_diameter_2022.csv
data_taggedFruit_diameter_2023.csv
data_taggedFruit_length_2022.csv
data_taggedFruit_length_2023.csv
```
Fresh yield and lifespan for tagged fruits (only available in experiment 2023)
```
data_taggedFruit_freshMatter_2023.csv
data_taggedFruit_lifespan_2023.csv
```
Weather data
```
weather_daily_2022.csv
weather_daily_2023.csv
```
