# Vehicle-Detector
Computes the number of wheels present in a vehicle by identifying it using a simple primitive CNN coded in tensorflow.

USAGE : 
python3 test.py <filename>
  
TRAINING :
To train it on your own dataset create a training folder consisting of seperate folders for each class
Change the name of the dataset folder in train.py.
Type python3 train.py to run the model.

Example of Dataset: <br />

. <br />
train.py <br />
test.py <br />
Dataset <br />
- Two wheelers <br />
  - one.jpg <br />
  - two.jpg <br />
  - ... <br />
- Three wheelers <br />
  - one.jpg <br />
  - two.jpg <br />
  - ... <br />
- Four Whelers <br />
  - one.jpg <br />
  - two.jpg <br />
  - ... <br />
  - ... <br />
vehicles.data-00000-of-00001 <br />
vehicles.index <br />
vehicles.meta
