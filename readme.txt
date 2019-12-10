Sam Richmond
SR4742

First, the UrbanSound8K dataset must be downloaded at https://urbansounddataset.weebly.com/urbansound8k.html
This will result in a folder called UrbanSound8K, which should be moved from ~/Downloads (or wherever)
to the root of this project. Then, the file fix.py should be run with python3, which will save two
Numpy files, X.npy and y.npy, which are the librosa extracted and cropped mel spectrograms of each
sound file. Then, main.py should be run with python3, which will try to train the model on the computer's GPU
and will print the progress of each epoch.

The actual dataset is too big to be committed on git, so getting it from the above link seems the most efficient
way to share this info (vs google drive or something like that).

