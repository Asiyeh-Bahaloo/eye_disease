How to run scripts?
==============
There are several arguments which need to be set when running scripts:


Arguments in evaluation scripts
---------------
#. weights_path
    Path to the model's weights file

#. data_folder
    "Path to the model's input data folder

#. result_path
    Path to the folder you want to save your results

#. loss
    type of loss function with which you want to compile your model. example: binary_crossentropy


Arguments in training scripts
---------------
#. weights_path
    Path to the model's weights file

#. data_folder
    "Path to the model's input data folder
    
#. result_path
    Path to the folder you want to save your results

#. loss
    type of loss function with which you want to compile your model. 

#. batch_size
    size of the batch you want to use
#. epochs
    number of epochs you want to train your model
#. patience
    number of patience you want to use for early stopping
#. imagenet_weights
    Whether to use imagenet weights or not?
#. train_label
    Path to the train input labels file (.csv)

#. val_label
    Path to the val input labels file (.csv)
#. learning_rate
    setting learning rate of optimizer
#. decay_rate
    setting decay rate of optimizer
#. momentum_rate
    setting momentum rate of optimizer
#. nesterov_flag
    setting nesterov term of  optimizer: True or False


**Now Let's see some examples of each script**


* eval_inception_v3.py

.. code-block:: bash
   
    python eye/scripts/eval_inception_v3.py --weights=./Data/model_weights_inception_v3.h5 --data=./Data --result=./Data

* eval_resnet_v2.py

.. code-block:: bash
   
    python eye/scripts/eval_resnet_v2.py --weights=./Data/model_weights_resnet_v2.h5 --data=./Data --result=./Data

* eval_vgg16.py

.. code-block:: bash
   
    python eye/scripts/eval_vgg16.py --weights=./Data/model_weights_vgg16.h5 --data=./Data --result=./Data

* eval_vgg19.py

.. code-block:: bash
   
    python eye/scripts/eval_vgg19.py --weights=./Data/model_weights_vgg19.h5 --data=./Data --result=./Data

* eval_xception.py

.. code-block:: bash

    python eye/scripts/eval_xception.py --weights=./Data/model_weights_xception.h5 --data=./Data --result=./Data



* train_inception_v3.py

.. code-block:: bash

    python eye/scripts/train_inception_v3.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data


* train_resnet_v2.py

.. code-block:: bash

    python eye/scripts/train_resnet_v2.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data


* train_vgg16.py

.. code-block:: bash

    python scripts/train_vgg16.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data

* train_vgg19.py

.. code-block:: bash

    python eye/scripts/train_vgg19.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data

* train_xception.py

.. code-block:: bash

    python scripts/train_xception.py --batch=2 --epoch=1 --patience=5 --loss=binary_crossentropy --data=./Data --result=./Data




