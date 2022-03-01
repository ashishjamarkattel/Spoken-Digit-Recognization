## Spoken Digit recognization

This repo consist of the analysis and building the model which is able to 
classify Spoken Digit recognization. The dataset was taken from kaggle. It includes 
about 4000 wave file which have 0-9 Spoken Digits Where each class 0-9 have around 200
sounds.

This repo consist how to featurize the audio data for the machine learning model, Here when i try to 
give the raw data directly to the model without featurizing it the accuray was around 10% but
after featurizing ie using Spectogram of the audio data and fitting model 
accuracy increased to 95-96 percent.

### About 
There are 2 model for model using Spectogram Feature that i build here where one includes using the mean after the LSTM layers and other
using the Flatten layers each bringing the dimension to 1D excluding the batch.
However, using mean approch has given quite good result but Flatten seem's to Outperform it.

### Accuracy

#### Raw Data
- 0.09 percent 
#### Spectogram Data
- Model1  :   86 percent (using mean layer)
- Model2  :   96 percent  (using Flatten layer)

### 
## Run Locally

Clone the project

```bash
git clone "this repo"
```

Go to the project directory

```bash
  cd Spoken-Digit-Recognization
```

Run for your Data

```bash
    python recognization_test.py -f "yourfilename.wav"
```

### 
### Dependency
- Tensorflow
- Keras
- Librosa
- Numpy 
- Matplotlib

### 
### Conclusion

#### Problem
Even getting the high accuracy of 96 percent the model seem's to not perform well when i tested for my sound
which is why we need highly distributed data. Since data contains only 1-2 person speaking.So model only recognize 
their sound.

#### Solution 

- Get high distributed data.
    -  Accent
    -  region
    -  Sound of various person.

- Increase the dataset

- Use Other featurize technique.
  - Fourier Transform.
  - Spectogram etc.

  
