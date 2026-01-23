# Tip Recognition
This program uses a lightweight CNN to classify if the region in the tip rack below the pipette is a tip or hole, after move the pipette vertically above the exact position of the tip/hole.

The accuracy is almost 100%.

# Dataset Direction
The dataset should be organized as follows: 

```
|--tip_recognition    
  |--image_classification_CNN  
  |  |--train.py  
  |  |--test.py   
  |  |--dataset   
  |  |  |--train   
  |  |  |  |--hole   
  |  |  |  |  |--iamge_hole_id.jpg   
  |  |  |  |  |--   
  |  |  |  |--tip   
  |  |  |  |  |--image_tip_id.jpg   
  |  |  |--val     
  |  |  |  |--hole   
  |  |  |  |--tip   
  |  |  |--test     
  |  |  |  |--hole   
  |  |  |  |--tip   
```

# Train

```
cd image_classification_CNN
python train.py
```

It will plot figures of metrics and output metrics as csv.

# Test
Please change the dir of images to be tested in `test.py` according to your real dir.

```
cd image_classification_CNN
python test.py
```
