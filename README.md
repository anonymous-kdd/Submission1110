## This is the implementation of SEENet for the submission 1110 in KDD23.

### Dependencies

- python >= 3.8
- torch >= 1.6.0
- dgl == 0.6.1
- numpy >= 1.21.5
- pandas >= 1.4.2


### Datasets
The raw data of different cities can be downloaded as follows:

[Business-based urban data](https://sites.google.com/site/yangdingqi/home/foursquare-dataset)

[Mobility-based urban data](https://ride.divvybikes.com/system-data)

The downloaded data should be preprocessed to generate the business-based relationship dataset (Business-RD) and the mobility-based relationship dataset (Mobi-RD) for locations, respectively:

Then run the preprocessing code for SEENet model:
```
python preprocess/run_Business_RD.py.py --input_data BUSINESS_RAW_DATA_PATH --output_path YOUR_OUTPUT_PATH
python preprocess/run_Mobi_RD.py.py --input_data MOBI_RAW_DATA_PATH --output_path YOUR_OUTPUT_PATH
```

### How to run
The model is first pre-trained with our self-supervised learning:
```
python train_ssl.py --dataset DATASET --local-weight LOCAL_W --global-weight GLOBAL_W --global-batch-size BATCH_SIZE
```

Then the model loads the pre-trained file and performs the inference:
```
python train_trial.py --dataset DATASET --pretrain-path MODEL_PATH
```

We also provide an example script on Tokyo dataset to show the overall running process for reproducibility:
```
bash run.sh
```
