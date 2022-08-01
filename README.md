# EVA Assignments

```
```

# S8
```

Training of CFAR10 with Resnet18 Architecture.
Target is to reach 85%.
Full setup is modular and this Assignment shows what is the effect of Big Receptive Field based model on small size dataset.

Results-
We saw there is huge overfitting as Training Accuracy is 100%. Means model has generated huge RF and learned the training dataset completely.
```

# S9
```

Training of CFAR10 with Resnet18 Architecture with different transformation.
Target is to reach 87%.
Full setup is modular and this Assignment shows how to use different transformations like Albumentations. 
Implemention of GradCam as module to show how the output layers are looking at the image.

Results-
We saw there is less overfitting as Training Accuracy 99.86%. Accuracy of Testset increased as due to different Augmantation Techniques. Also we saw the Gradcam functionaity in image for each epoch. As the model accuracy increased, Gradcam showed the confidence of model last layer's output.
```
