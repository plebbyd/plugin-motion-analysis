# Science
Segments of the area of where moving objects are or an object that is continuously moving with a certain pattern from videos taken from a street view camera. The algorithm is first extracting changes of brightness of the surface of the images in temporal dimension which is shown as a sinusoidal signal and second using the FFT algorithm to analyze the signal to create a temporal descriptor. In addition to the temporal information, the algorithm utilized a spatial descriptor extracted by Local Binary Pattern (LBP) histogram [1-2] to understand the scene in spatial perspective [3]. The local temporal and spatial descriptors are utilized to determine local probabilistic classification using a Random Decision Forest [4] and final detection maps were computed by means of regularization using a binary Markov Random Field [5].
 
# AI@Edge:
The model accepts an array of shapes [N videos, 60 frames, 800 pixels, 600 pixels, 3 color channels] where N is the number of separate videos to process. Note that the 60 frames must be spaced out in time according to the model's framerate. This means that if you are inferencing using the 5fps model (the model I recommend- it is not too resource intensive to buffer while still retaining accuracy), then make sure to feed it 60 frames spaced out according to 5fps timing, that is 0.2s between each frame capture. This means that the model would receive 60 frames over the course of 12s and would output one water mask from that motion data. The video is then passed through the consecutive image processing. Each pixel of the video frames processed and classified how much it ccan be segmented as a pixel of an area showing a flow.

# Using the code
Output: recorded video, inference image  
Input: 5 second video (12 fps, total 60 frames)  
Image resolution: 800x600  
Inference time:  
Model loading time:  

# Arguments
   '-stream': ID or name of a stream, e.g. top-camera  
   '-duration': Time duration for input video (default = 10)  
   '-resampling': Resampling the sample to -resample-fps option (default = 12)  
   '-resampling-fps': Frames per second for input video (default = 12)  
   '-skip-second': Seconds to skip before recording (default = 3)  
   '-sampling-interval': Inferencing interval for sampling results (default = -1, no interval)  

# Reference
[1] Timo Ojala, Matti Pietikainen, and Topi Maenpaa. "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." IEEE Transactions on pattern analysis and machine intelligence 24, no. 7 (2002): 971-987.  
[2] Qian, Xueming, Xian-Sheng Hua, Ping Chen, and Liangjun Ke. "PLBP: An effective local binary patterns texture descriptor with pyramid representation." Pattern Recognition 44, no. 10-11 (2011): 2502-2515.  
[3] Pascal Mettes, Robby T. Tan, and Remco C. Veltkamp. "Water detection through spatio-temporal invariant descriptors." Computer Vision and Image Understanding 154 (2017): 182-191.  
[4] Antonio Criminisi, Jamie Shotton, and Ender Konukoglu. "Decision forests: A unified framework for classification, regression, density estimation, manifold learning and semi-supervised learning." Foundations and trends® in computer graphics and vision 7, no. 2–3 (2012): 81-227.  
[5] Yuri Boykov, and Vladimir Kolmogorov. "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." IEEE transactions on pattern analysis and machine intelligence 26, no. 9 (2004): 1124-1137.
