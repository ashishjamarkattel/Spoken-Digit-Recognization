import argparse
from keras.models import load_model
import librosa
from keras.preprocessing.sequence import pad_sequences
import numpy as np



model = load_model("bestmodel3.h5")
max_length = int(0.8 * 22050)
sample_rate = 22050


def convert_to_spectrogram(raw_data):

    spectrum = librosa.feature.melspectrogram(y=raw_data, sr=sample_rate, n_mels=64)
    logmel_spectrum = librosa.power_to_db(S=spectrum, ref=np.max)
    return logmel_spectrum

def predict(file_name):

    x, s = librosa.load(file_name,sr=sample_rate)
    x = pad_sequences([x],maxlen=max_length,dtype="float32",padding="pre")
    x = convert_to_spectrogram(x[0])
    x = np.expand_dims(x,axis=0)
    pred = model.predict(x)
    pred = np.argmax(pred)
    
    return pred


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Predict the audio ")
    parser.add_argument("-f","--file",required=True,help="audio file that you want to predict supported : Wave")
    parser.add_argument("-q","--quite",action="store_true",help="Quietly print the value")
    parser.add_argument("-v","--verbose",action="store_true",help="Give the details ")
    args = parser.parse_args()
    file_name = args.file
    if file_name.split('.')[-1]=="wav":

        try:
            pred_val = predict(args.file)
            if args.verbose:
                print("The predicted sound value for given audio {} is {}".format(args.file,pred_val))
            else:
                print(pred_val)
            
        except:
            print("Something went wrong ......")
    else:
        print("File format not match, please choose other file supported format .wav")