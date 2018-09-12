# HBPNet in TensorFlow-Slim

This is implementation of the HBPNet in tensorflow-slim. The origin paper is "[Hierarchical Bilinear Pooling for Fine-Grained Visual Recognition](https://arxiv.org/abs/1807.09915)" . The author gave a model based on caffe and you can find it [here](https://github.com/ChaojianYu/Hierarchical-Bilinear-Pooling).

## Network structure

![HBPNet](https://github.com/Jowekk/HBPNet/blob/master/HBPNet.png)

##The datasets

This model requires data in TF-record format. You can download the aircraft datasets, birds datasets .



##Pre-trained Models

The fine tuned model is available in google drive: aircraft, birds.



## Train and eval

Trian aircraft/birds:

~~~shell
sh train_aircraft.sh
~~~

~~~shell
sh train_birds.sh
~~~

Eval aircraft/birds:

~~~shell
sh eval_aircraft.sh
~~~

~~~shell
sh eval_birds.sh
~~~

## View the feature maps

Using the jupyter notebook to run

~~~
jupyter notebook show_result.ipynb
~~~

For example, the bilinear_p2 feature maps

![041](https://github.com/Jowekk/HBPNet/blob/master/images/041.png)

![041](https://github.com/Jowekk/HBPNet/blob/master/images/042.png)

![041](https://github.com/Jowekk/HBPNet/blob/master/images/043.png)




