## Background
Segments of the water flooded area from video taken from a street view camera. The algorithm is first extracting changes of brightness of the surface of the images in temporal dimension which is shown as a sinusoidal signal and second using the FFT algorithm to analyze the signal to create a temporal descriptor. In addition to the temporal information, the algorithm utilized a spatial descriptor extracted by Local Binary Pattern (LBP) histogram [1-2] to understand the scene in spatial perspective [3]. The local temporal and spatial descriptors are utilized to determine local probabilistic classification using a Random Decision Forest [4] and final detection maps were computed by means of regularization using a binary Markov Random Field [5].
 
## AI at Edge:
The application first records 5fps video for 12 seconds or 1 fps for 60 seconds or 50 fps for 1.2 seconds (60 frames total) to analyze the area where water is. The video is then passed through the consecutive image processing. Each pixel of the video frames processed and classified how much it can be segmented as water pixel.


### Reference
[1] Timo Ojala, Matti Pietikainen, and Topi Maenpaa. "Multiresolution gray-scale and rotation invariant texture classification with local binary patterns." IEEE Transactions on pattern analysis and machine intelligence 24, no. 7 (2002): 971-987.

[2] Qian, Xueming, Xian-Sheng Hua, Ping Chen, and Liangjun Ke. "PLBP: An effective local binary patterns texture descriptor with pyramid representation." Pattern Recognition 44, no. 10-11 (2011): 2502-2515.

[3] Pascal Mettes, Robby T. Tan, and Remco C. Veltkamp. "Water detection through spatio-temporal invariant descriptors." Computer Vision and Image Understanding 154 (2017): 182-191.

[4] Antonio Criminisi, Jamie Shotton, and Ender Konukoglu. "Decision forests: A unified framework for classification, regression, density estimation, manifold learning and semi-supervised learning." Foundations and trends® in computer graphics and vision 7, no. 2–3 (2012): 81-227.

[5] Yuri Boykov, and Vladimir Kolmogorov. "An experimental comparison of min-cut/max-flow algorithms for energy minimization in vision." IEEE transactions on pattern analysis and machine intelligence 26, no. 9 (2004): 1124-1137.
