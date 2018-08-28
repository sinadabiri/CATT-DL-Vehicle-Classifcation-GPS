# A deep learning approach for vehicle classification using large scale GPS data

## Summary 
This is the project that I have done as an intern over the summer 2018 in [Center for Advanced Transportation Technologu](http://www.catt.umd.edu/).

I have conducted two primary tasks in this project: 
**Labeling a large-scale GPS trajectory dataset based on the FHWA classification scheme.** An efficient programmatic approach is developed to label a large scale GPS data (~20 million GPS trajectory) by means of vehicle class information obtained from Virtual Weight Station vehicle records. Information from [open source routing machine API](http://project-osrm.org/docs/v5.5.1/api/#general-options) is also exploited to improve the accuracy of our labeling approach by extracting the map-based distance rather than the direct distance between coordinates.   
**Developing a deep Convolutional Neural Network for Vehicle-Classification (CNN-VC).** For the first time in this domain, a CNN-based deep-learning model is developed to identify vehicle's class from the proposed GPS representation. The CNN-VC comprises a stack of CNN layers for extracting abstract features from the GPS representation, a pooling operation for encapsulating the most important information, and a softmax layer for performing the classification task.

The abstract regading this project is as follows:
**Abstract**
Vehicle classification is an essential step in a wide range of transportation domains, ranging from toll systems and parking space optimization to traffic management and urban planning. Fixed-point traffic flow sensors (e.g., weigh-in-motion systems and video image processors) have been extensively used for classifying vehicles. In spite of popularity, such sensors are subjected to major issues such as high installation and maintenance cost, disruptive to traffic flow, and low spatial coverage. Well-established positioning tools such as GPS is a viable alternative sensor that can address these issues by recording vehicles' spatiotemporal information while they are moving in a traffic network. In this paper, first, an efficient programmatic strategy is designed to label large-scale GPS trajectories according to the fine-grained Federal Highway Administration (FHWA) vehicle classes by means of Virtual Weigh Station vehicle records. Using the large-scale labeled GPS data, a deep convolutional neural network is proposed for identifying the vehicles' classes from their trajectories. Since a raw GPS trajectory does not convey meaningful information, this paper proposes a novel representation of GPS trajectories, which is not only compatible with deep-learning models, but also comprising vehicle-motion characteristics and roadway features. To this end, an open source navigation system is also exploited to obtain more accurate information on travel time and distance between GPS coordinates. An extensive set of experiments is conducted on several datasets with various vehicle categorizations to evaluate our proposed model in comparison to state-of-the-art deep-learning and classical machine-learning techniques. The experimental results reveal that our model consistently outperforms others in terms of robust performance metrics.

## Code Repository
**1.Labeling Procedure:** 'Filtering-Initial.py', 'Labeling_Spark_OSRM.py', 'Retrieve_all_trips.py' are all the files associated with the programmetic labeling strategy. 'Labeling_Spark_OSRM.py' comprises the core part of the strategy, in which **Spark** is utilized for speeding up by processing the data in a parallel fahsion.
**2.GPS Represntation:** 'CATT_Instance_Creation.py' and 'CATT_DL_DATA_Creation.py' create the GPS trajectories as samples and then convert them into image-like tensors by extracting motion-related and roadway features out of that. 
**3.Deep Learning Models:** Several deep learning models based on CNN, RNN, and attention mechansim have been developed to classify vehicles from their corresponding GPS trajectory. 

## Paper
I will provide the draft of paper associated with this project in the future soon. 

## Contact

Please contact me at sina@vt.edu if you have any questions. 

