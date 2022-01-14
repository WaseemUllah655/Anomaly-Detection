# Anomaly-Detection
Anomaly detection using CNN-LSTM
Abstract 
In current technological era, surveillance systems generate an enormous volume of video data on a daily basis, making its analysis a difficult task for computer vision experts.
Manually searching for unusual events in these massive video streams is a challenging task, since they occur inconsistently and with low probability in real-world surveillance.
In contrast, deep learning-based anomaly detection reduces human labour and its decision making ability is comparatively reliable, thereby ensuring public safety. 
In this paper, we present an efficient deep features-based intelligent anomaly detection framework that can operate in surveillance networks with reduced time complexity. 
In the proposed framework, we first extract spatiotemporal features from a series of frames by passing each one to a pre-trained Convolutional Neural Network (CNN) model. 
The features extracted from the sequence of frames are valuable in capturing anomalous events. We then pass the extracted deep features to multilayer Bi-directional long short-term memory (BD-LSTM) model, which can accurately classify ongoing anomalous/normal events in complex surveillance scenes of smart cities. We performed extensive experiments on various anomaly detection benchmark datasets to validate the functionality of the proposed framework within complex surveillance scenarios. We reported a 3.41% and 8.09% increase in accuracy on UCF-Crime and UCFCrime2Local datasets compared to state-of-the-art methods.
