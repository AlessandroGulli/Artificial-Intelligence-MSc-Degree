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


![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Architecture.JPG)

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/User.jpg)

To extract just rectangular shape, the painting’s subject has been little blurred to smooth unwanted potential contours which might be detected in further steps of processing. First of all, it was set a filter to get structuring elements of rectangular shapes choosing with a kernel size of 5x5, afterward, it is applied a morphological transformation exploiting the same kernel to exalt edges and finally proceeding with another morphological step to close small holes inside foreground objects.
From this first pass, the image is switched from RGB to Lab space isolating just L component, in turn a threshold is applied to adjust the result. Other space colors were tried (HSV, YCbCr, HLS), but Lab showed the best performances in general.
On the other hand, a further objective was taking the original image to isolate darker colors components. The process is achieved by matching pixels below a precise threshold, inasmuch to creating a particular mask. At this stage it is used the mask to select pixels from the original image, filtering the result with a Gaussian kernel and finally applying a threshold to change the image into a binary one. This way it is possible to generate a black-white picture to get a darker region of starting image. Once the two binary images have been generated with the steps above, afterward are merged. 
From this union the contours are extracted, drawing, and finally filled to get a uniform surface

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Mask.JPG)

Therefore, once obtained the mask of a given painting, it needs to isolate it from the entire context. At this stage the image from the previous passes is analyzed in the attempt to assign for every mask found an identity. All the masks are considered as centroid, a spot for a candidate painting. Computing the areas of centroids, finding the center of each one, and knowing the distance from each other can lead to identifying the position of a single region on the foreground.

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/centroids.JPG)

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/countorus.JPG)

Further analyzing the points of masks contours it is possible to extract top-left, top-right, bottom -left, and bottom-right corners of the shapes. Knowing precisely the four coordinates is crucial to rectify the perspective.

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/rectifiy.JPG)

Before correcting the perspective and validating the detection, a couple of further steps are implemented. The first one is based on geometric features, like sides ratio of the rectangular boxes found to inscribed the given shapes, discarding extreme ratio results, which would shed light on a possible fault classification. Second, it is to testing if a given shape is a painting or not for real. Particularly, this latter task is done by an SVM classifier, whose aspects will be overviewed in the next chapters. As we know, the Faster R-CNN/Mask R-CNN. architectures leverage a Region Proposal Network (RPN) to generate regions of an image that potentially contain an object. So the method implemented somehow it recalls these concepts, classifying each ROI using the extracted features with a Support Vector Machine. Clearly in the context of the project, the other advantage is represented by removing out-of-interest subjects. Therefore, positive classification outcomes make the perspective of the paintings corrected.

Regarding people detection, it has been implemented a pre-trained Mask R-CNN on the COCO dataset. Once loaded weights and configuration, the NN is up and running. Since the nature of technical choice to pick something already functional, the implementation of this part regarded only how to manage at best the potential of the CNN. Therefore, a single video frame is passed as an input, once it is processed, the algorithm gives back to the users the coordinates of boxes and masks of the detected object. Iterating through these boxes it is possible to print out this information on an image. Since the detections interested is about people, other ID classes have been filtered out. In our case people ID is equal to 1 as specified in the COCO dataset. Besides, if some people are detected inside canvas they are rejected, because they are an outlier. A noteworthy aspect is to exploit also the false positive detection on statues to identify with correctness this latter as well. In the end, once a person has been detected both on a real human subject or on a statue, this outcome is in turn evaluated by an SVM classifier that has been first trained on a dataset of people and statues. The expedient allows us to discriminate the subject and correctly segment it.

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/MaskRCNN.JPG)

To get features from images, as mentioned in the introduction, have been computed the Histograms of Oriented Gradients. The HOG descriptor returns a real-valued feature vector. The dimensionality of this feature vector is dependent on the parameters chosen for the orientations, pixels_per_cell, and cells_per_block.
Both the datasets have been collected by hand from the web and free datasets. Among paintings pictures are not present examples from “Galleria Estense”. Nevertheless the same operational function, the classifiers are distinct in terms of characteristics. Paintings classifier works on Lab images with a dimension of 128x128 pixels, on the other hand, people classifiers work on HLS images with 100x100 pixels. The choice of these parameters came out after several attempts with the aim to achieve the best and homogenous result possible to work fine on a good part of the videos provided. Features for paintings – not painting images have been retrieved setting the following parameters: 
* 9 orientations 
* 8x8 pixels per cell 
* 2x2 cell per block
![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/HOG.JPG)

Features for people – statues images have been retrieved setting the following parameters:
* 44 orientations 
* 8x8 pixels per cell 
* 1x1 cell per block

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/HOG2.JPG)

The feature vectors dataset fed a linear SVM whose accuracy on the test set of painting classifier is 0.946, whilst on the test set of people classifier is 0.939. Although parallel approaches were developed, as developing from scratch a BOVW classifier or simply classify the histogram in different colorspaces, HOG gave the best result possible. As shown in figure 15 the action of the SVM classifier allows discard subjects that are erroneously labeled as paintings. The figures below clearly show as an aisle, a barrier, a bunch of plates and even a door if weren’t present this filtering mechanism, would have been displayed into the final view.

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Removals.JPG)

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Removals1.JPG)

![Image](https://github.com/AlessandroGulli/AI_MS_Degree/blob/main/ComputerVision_and_ML_for_Art_Works/images/Removals2.JPG)

