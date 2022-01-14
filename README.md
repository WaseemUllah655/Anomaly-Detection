# Anomaly-Detection
Anomaly detection using CNN-LSTM
Abstract 
In current technological era, surveillance systems generate an enormous volume of video data on a daily basis, making its analysis a difficult task for computer vision experts.
Manually searching for unusual events in these massive video streams is a challenging task, since they occur inconsistently and with low probability in real-world surveillance.
In contrast, deep learning-based anomaly detection reduces human labour and its decision making ability is comparatively reliable, thereby ensuring public safety. 
In this paper, we present an efficient deep features-based intelligent anomaly detection framework that can operate in surveillance networks with reduced time complexity. 
In the proposed framework, we first extract spatiotemporal features from a series of frames by passing each one to a pre-trained Convolutional Neural Network (CNN) model. 
The features extracted from the sequence of frames are valuable in capturing anomalous events. We then pass the extracted deep features to multilayer Bi-directional long short-term memory (BD-LSTM) model, which can accurately classify ongoing anomalous/normal events in complex surveillance scenes of smart cities. We performed extensive experiments on various anomaly detection benchmark datasets to validate the functionality of the proposed framework within complex surveillance scenarios. We reported a 3.41% and 8.09% increase in accuracy on UCF-Crime and UCFCrime2Local datasets compared to state-of-the-art methods.

# Features Extraction 

ResNet-50 has a total of 50 weighted number of layers, with 23.5 million trainable parameters. We extracted features from the last fully connected layer of a 15-frames sequence, which are then fed into the BD-LSTM for further processing. 
We performed experiments on the features of various CNNs by integrating our approach with different sequential models. The VGG-19 with multi-layer BD-LSTM achieved 82% accuracy on UCF-Crime and 87.5% on UCFCrime2Local, while the inception V3 with multi-layer BD-LSTM reached 80% accuracy on UCF-Crime and 88% on UCFCrime2Local. The ResNet-50 with multi-layer BD-LSTM achieved greater success, with an accuracy of 85.53% on UCF-Crime and 89.05% on UCFCrime2Local.


# Multi-layer BD-LSTM

The features extracted by the ResNet-50 are used to decide either the normal or anomalous events that are fed to the multi-layer BD-LSTM in the form of chunks for an anomaly detection decision. The first chunk of the 1000 features from the initial frame of the video forms the input to the multi-layer BD-LSTM at time “t”, while the next feature’s chunk at t+1 forms the second input to the multi-layer BD-LSTM, and so on. The overall structural design of the proposed multi-layer BD-LSTM is illustrated in Fig. 5. Fig. 5(a) shows the training stage, in which the training data are passed to the model. The hidden state combines the forward and backward passes in the output layer, while backpropagation is used to adjust the bias and weights. A sample of 20% of the total data is used for validation purposes, while cross-entropy is employed for error rate assessment along with a learning rate as default of 0.001.

Citation Policy
If you would like to cite this repository, please use the following DOI:

DOI https://doi.org/10.1007/s11042-020-09406-3

or use this BibTex

@article{ullah2021cnn,
  title={CNN features with bi-directional LSTM for real-time anomaly detection in surveillance networks},
  author={Ullah, Waseem and Ullah, Amin and Haq, Ijaz Ul and Muhammad, Khan and Sajjad, Muhammad and Baik, Sung Wook},
  journal={Multimedia Tools and Applications},
  volume={80},
  number={11},
  pages={16979--16995},
  year={2021},
  publisher={Springer}
}
