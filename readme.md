
## Deployment

To run this project locally, you'll first need to download the following datasets:

For the [model](https://www.kaggle.com/datasets/fareselmenshawii/large-license-plate-dataset)
and for the [testing](https://www.kaggle.com/datasets/andrewmvd/car-plate-detection), though you can use any dataset for testing.

### Clone this repo
```bash
  git clone https://github.com/Downmoto/A3.git
  cd A3
```


### Train your own model
```bash
  # once inside the A3 directory
  cd plate_detector
  # create datasets directory
  # place images and labels directories inside datasets directory (the model dataset)
  python script.py
  python main.py # this will take a significant amount of time
```

### Use the model you've trained, or the one provided (GroupSevenTrained.pt)
```bash
  # go back to A3 directory if you are not there
  # create a datasets directory
  # place images directory inside datasets directory (the testing dataset)
  # do not place annotation/label directory in datasets
  python main.py # this will take a bit of time
```

### Results
You will find an output folder containing images with the CV read text labeled above the license plates. You will also find a results.csv file.

