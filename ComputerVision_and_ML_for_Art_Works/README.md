# Computer Vision and ML for Art Works

> A complete pipeline to detect, classify and segment art works and people from live videos.

- This repository aims to expose the choices and technical aspects concerning the final project of Vision And Cognitive Systems course A.A 2019/2020.
	Given an input video recorded inside “Galleria Estense” interiors, the mandatory tasks for the project are:
	* Painting detection and rectification
	* Painting Retrieval
	* People detection
	* People localization

The approach chose to deal with the previous complex points was to divide into two different branches the technical aspects of implementations. 
The first regarding the paintings and the second regarding people detection. 
The first case was chosen to detect paintings without using neural networks but wholly rely on classical image processing techniques; on the other hand concerning human detection, it was employed a Mask RCNN, pre-trained on the COCO dataset, able to correctly retrieve the subject in the context. 
To validate detections on both parts and in the attempt to eliminate false positive detections, two different linear SVM classifiers trained on HOG features have been implemented. This expedient in case of paintings laid to an improvement of results with the desired outcome to remove outliers, instead of in case of a human subject made it possible to cut out the false positives detection on statues since in the COCO dataset isn’t present that type of class. 
About paintings, once detection is validated by SMV, a SIFT classification classifies canvases against a pre-computed DB of descriptors.

The entire chain of operations on video images can be summarized as follow:
* Image processing on input frame video
* Painting detection from the context
* Painting outliers removals through linear SVM classifier trained on HOG features
* Painting classification from DB
* People and statue detection with Mask R-CNN
* People detection outliers through linear SVM classifier trained on HOG features
* A user interface for final visualization collecting all the information retrieved


[!Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Architecture.JPG)
