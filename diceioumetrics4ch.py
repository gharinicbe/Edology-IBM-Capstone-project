#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
def dice_coefficient(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)  # Cast to float32
    y_pred = tf.cast(y_pred, tf.float32)  # Cast to float32
    smooth= 1e-15
    y_true = tf.keras.layers.Flatten()(y_true)
    y_pred = tf.keras.layers.Flatten()(y_pred)

    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return (2.0 * intersection + smooth) / (union + smooth)

def dice_loss(y_true, y_pred):
    # Calculate Dice coefficient for each class and take the mean
    num_classes = 4  # Change this to the actual number of classes
    dice_coefficients = []
    for i in range(num_classes):
        class_true = y_true[..., i]
        class_pred = y_pred[..., i]
        dice_coefficients.append(dice_coefficient(class_true, class_pred))
    return 1.0 - tf.reduce_mean(dice_coefficients)

class IoUClassMetrics(tf.keras.metrics.Metric):
    def __init__(self, num_classes=4, name="iou_class", **kwargs):
        super(IoUClassMetrics, self).__init__(name=name, **kwargs)
        self.num_classes = num_classes
        self.intersection = [self.add_weight(name=f"intersection_{i}", initializer="zeros") for i in range(num_classes)]
        self.union = [self.add_weight(name=f"union_{i}", initializer="zeros") for i in range(num_classes)]

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        for class_id in range(self.num_classes):
            intersection = tf.reduce_sum(y_true[..., class_id] * y_pred[..., class_id])
            union = tf.reduce_sum(y_true[..., class_id] + y_pred[..., class_id]) - intersection

            # Use class_id as an index to update intersection and union values
            self.intersection[class_id].assign_add(intersection)
            self.union[class_id].assign_add(union)

    def result(self):
        iou_per_class = [intersection / (union + 1e-10) for intersection, union in zip(self.intersection, self.union)]
        return iou_per_class

