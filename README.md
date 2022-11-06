# Traffic_Sign_Classification
Traffic sign classification using CNN.

## About the project

### Dataset
[German Traffic Sign Recognition Benchmark](https://benchmark.ini.rub.de/). Used subdataset which consists of 12 classes of traffic signs (and includes 16470 images).

### Neural network
Consists of 3 convolutional layers (which have both MaxPooling and Dropout layers between them), as well as 2 fully connected layers.

### Preprocessing
Preprocessing includes using CLAHE (contrast limited adaptive histogram equatization) method, which increases contrast of the image as well as making edges more clear.

### Results
- Accuracy of 99.5% with using CLAHE, and 96.3% without using CLAHE.
- Network is also tested on another dataset named [rMASTIF](http://www.zemris.fer.hr/~kalfa/Datasets/rMASTIF/), on which it achieves 91% accuracy (using CLAHE preprocessing).

### Requirements:
- numpy
- matplotlib
- keras
- tensorflow
- opencv - cv2
- skimage
- pandas
- PIL
