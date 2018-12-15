# Vehicle-Detector
## Computes the number of wheels present in a vehicle by identifying it using a simple primitive CNN coded in tensorflow.

**USAGE :**

```python3 test.py <filename>```
  
**TRAINING :**

To train it on your own dataset create a training folder consisting of seperate folders for each class
Change the name of the dataset folder in train.py.
Type ```python3 train.py``` to run the model.

**Example of Dataset:**

```
.
- train.py
- test.py
- Dataset
  - Two wheelers
    - one.jpg
    - two.jpg
    - ...
  - Three wheelers
    - one.jpg
    - two.jpg
    - ...
  - Four Whelers
    - one.jpg
    - two.jpg
    - ...
  - ...
- vehicles.data-00000-of-00001
- vehicles.index
- vehicles.meta
```

**TODO : **

- [ ] Train the model on more vaehicle classes
- [ ] Make the CNN model more complex

