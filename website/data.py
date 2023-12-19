import librosa
import pandas as pd
import numpy as np
import os
import soundfile as sf



def get_features(file_path, file_name):
    #returns features extracted using librosa lib.
    y, sr = librosa.load(file_path)
    iter_ = int(y.shape[0]/sr)
    features = []
    for i in range(iter_):
        y_seg = y[i*sr:(sr*i+sr)]
        chroma_stft = np.mean(librosa.feature.chroma_stft(y=y_seg, sr=sr))
        rms = np.mean(librosa.feature.rms(y=y_seg))
        spec_cent = np.mean(librosa.feature.spectral_centroid(y=y_seg, sr=sr))
        spec_bw = np.mean(librosa.feature.spectral_bandwidth(y=y_seg, sr=sr))
        rolloff = np.mean(librosa.feature.spectral_rolloff(y=y_seg, sr=sr))
        zcr = np.mean(librosa.feature.zero_crossing_rate(y_seg))
        mfcc = np.mean(librosa.feature.mfcc(y=y_seg, sr=sr, n_mfcc=20),axis=1)
        features.append([chroma_stft,rms,spec_cent,spec_bw,rolloff,zcr,*mfcc]) # starred expressions for mfcc to get 20 columns instead of 20 row

    return features

def make_features_df(audio_dir):
    #get train features from original Database
    labels = ['REAL','FAKE']  # add 'FAKE' later # audio are in separate folders: FAKE and REAL
    feature_list = []
    for label in labels:
        files = os.listdir(os.path.join(audio_dir, label))
        for file in files:
            file_features=get_features(os.path.join(audio_dir,label,file),file)
            for segment_features in file_features:
                    feature_list.append(segment_features + [label])

    columns = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20', 'LABEL']

    features_df = pd.DataFrame(feature_list,columns = columns)
    return features_df

def make_test_features_df(audio_dir,filename):
    #get test feature from a giver file.
    file_features=get_features(os.path.join(audio_dir,filename),filename)
    feature_list = []
    for segment_features in file_features:
            feature_list.append(segment_features)

    columns = ['chroma_stft', 'rms', 'spectral_centroid', 'spectral_bandwidth',
       'rolloff', 'zero_crossing_rate', 'mfcc1', 'mfcc2', 'mfcc3', 'mfcc4',
       'mfcc5', 'mfcc6', 'mfcc7', 'mfcc8', 'mfcc9', 'mfcc10', 'mfcc11',
       'mfcc12', 'mfcc13', 'mfcc14', 'mfcc15', 'mfcc16', 'mfcc17', 'mfcc18',
       'mfcc19', 'mfcc20']

    features_df = pd.DataFrame(feature_list,columns = columns)
    return features_df
