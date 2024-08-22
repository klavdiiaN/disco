# Data bias identification with MyTH

This folder contains the implementation of the `MyTH` approach as a default task in the DISCO framework. `MyTH`, which stands for `MyThisYourThat` ([Naumova et al.](https://github.com/EPFLiGHT/MyTH)), is a novel extension of `ProtoPNet` ([Chen et al.](https://arxiv.org/abs/1806.10574)) for interpretable and privacy-preserving identification of data bias in federated learning for images. It allows data owners to diagnose each others data without looking at it: i.e compare one client's _This_ with other's _That_.

The original implementation of `MyTH` in Python can be found [here](https://github.com/EPFLiGHT/MyTH).
The materials necessary to implement the baseline original `ProtoPNet` were adopted from [this repository](https://github.com/cfchen-duke/ProtoPNet).

[This document](./MyTH.md) describes how `MyTH` works.

So far, this code can be run using the `cli` functionality of DISCO in `Node.js`. 

To do so, first, please download or clone this repository and install `Node.js` and the dependecies as described in the [Developer guide](../../../../DEV.md). Then from the root repository, build the project by running:
````
npm -ws run build
````
After that, please, go to the `cli` folder:
````
cd cli
````
You are almost ready to run the script. Before, let's ensure that the dataset is ready. The dataset should be placed in the parent to DISCO directory and have the following structure:
- 📁 dataset_directory
  - 📁 client_0
    - 📁 train
      - 📁 class_0
        - 📄 imageFile.jpg
        - ...
      - 📁 class_1 
      - ...
    - 📁 validation (optional)
    - 📁 push (optional)
  - 📁 client_1
  - 📁 client_2
  - 📁 client_3

[This repository](https://github.com/EPFLiGHT/MyTH/blob/main/utils.py) contains scripts that may be helpful for distributing data across client folders. You can experiment with different number of clients. Validation set is optional (it must have the same organization as the `train` folder). If it's not provided, 10% of the training data will be saved for validation. The `push` dataset is (a part of) training data and necessary to visualize the prototypes. You may need to provide it separately if the training data has been augmented with image modifications such as flipping, rotation, etc. The `push` folder must also be organized as `train`.
Now, you're ready to start.

### 💡 Training a global model in a federated setting

To train a global model, run the script `dist/bias_detection.js` using `node` command and providing the following arguments:

- `-t` task name which is `ppnet` in this case
- `-u` number of users (clients)
- `-e` number of training epochs
- `-b` batch size
- `-c` number of classes in the dataset
- `-d` path to the dataset folder, `../../FOLDER_NAME/`
- `-p` optional flag to indicate if a separate push dataset was provided
- `-v` optional flag to indicate if a separate validation dataset was provided

For example, to train a model with 4 clients for 11 training epochs with 10 classes dataset, 80 batch size and a separate validation set, run:
````
node dist/bias_detection.js -t ppnet -u 4 -e 11 -b 80 -c 10 -d ../../training_data/ -v
`````
For each client N, the script saves the trained model and visualized prototypes after every 10 training epochs into folders named `models-clientN` and `prots-clientN`, respectively. It also writes the validation loss, balanced accuracy, sensitivity, and specificity values after each epoch into a JSON file named `results.json` in the folder `models-clientN`.

### 💡 Training a local model

To train a local model on one client's data only, run `dist/bias_detection.js` with the arguments described above setting `-u` to 1. Keep in mind, however, that the dataset folder in this case should contain only the folder of the client to train. Otherwise, the model will be trained on the data of the first client in the dataset directory. 

In this case, the script saves the trained model and visualized prototypes after every 10 training epochs into folders named `models-clientLocal` and `prots-clientLocal`, respectively.

### 🔍 Bias identification in a test image

To find a patch in a test image mostly activated by the prototypes learned by different models (local and global), we adapted a script from the [ProtoPNet's repository](https://github.com/cfchen-duke/ProtoPNet). To do the analysis, run [`local_analysis.ts`](local_analysis.ts) specifying the following parameters:

- `-i` name of a test image
- `-l` label of a test image as a number (an index of a class to which the image belongs. Note, that the counting starts from 0)
- `-n` an index of a client whose model is going to be analyzed (put Local for local models)
- `-e` number of epochs for which the model was trained
- `-c` number of classes in the dataset

The data needed is two folders named `models-clientN` and `prots-clientN` saved during training, where N is a client index or 'Local' for local models, and an image to analyse in `.jpg` or `.png` format. This data should be placed in the root DISCO directory.

For example, if I am the fourth client and I want to find regions in a `test.jpg` image (class 0) most activated by a global model, I run from the root DISCO directory:
`````
node discojs/dist/models/ppnet/local_analysis.js -i test.jpg -l 0 -n 3 -e 10 -c 10
`````
If I want to analyse my local model, I run:
`````
node discojs/dist/models/ppnet/local_analysis.js -i test.jpg -l 0 -n Local -e 10 -c 10
`````
The script will output the predicted class and top-10 most activated prototypes with the similarity scores and class-connection values (how strongly prototypes are connected to the predicted class). It will also create a folder with the most activated patches in the test image and the corresponding nearest prototypes from the training set. The folder is named `most_activated_prototypes`. Consider saving it in a different directory or renaming it before running the script again.