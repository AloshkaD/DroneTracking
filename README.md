# Deep Reinforcement Learning Drone


This repository integrates AirSim with openAI gym and keras-rl for autonomous vehicles through deep reinforcement learning. AirSim allows you to easly create your own environment in Unreal Editor and keras-rl let gives your the RL tools to solve the task.

Requirements:

[AirSim](https://github.com/Microsoft/AirSim)

[stable_baseline](https://github.com/hill-a/stable-baselines)

[openAI gym](https://github.com/openai/gym)


Some modifications are required to the stable-baseline models, namely the cnn used for processing the states. 

If the state returns a grayscale depth image of the shape (cols, row, channels), and the image has a h or w smaller than 80, follow these instructions to make the repo work.
go to ```stable-baselines\stable_baselines\common\policies.py```
 and add 
```
def multi_cnn(scaled_images, **kwargs):
    """
    CNN for dealing with multi input environment that include 1D and 2D data. 

    :param scaled_images: (TensorFlow Tensor) Image input placeholder
    :param kwargs: (dict) Extra keywords parameters for the convolutional layers of the CNN
    :return: (TensorFlow Tensor) The CNN output layer
    """
    ##d_img = Input(tensor=scaled_images)
    #d_img=scaled_images
    #d_img=tf.reshape(d_img, (20, 100, 1))
    #print('d_img shape',scaled_images.shape)
    activ = tf.nn.relu
    layer_1 = activ(conv(scaled_images, 'c1', n_filters=32, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    #print('######layer_1######', layer_1.shape)
    layer_2 = activ(conv(layer_1, 'c2', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = activ(conv(layer_2, 'c3', n_filters=64, filter_size=3, stride=1, init_scale=np.sqrt(2), **kwargs))
    layer_3 = conv_to_fc(layer_3)
    return activ(linear(layer_3, 'fc1', n_hidden=512, init_scale=np.sqrt(2)))
```
```
class MultiCnnPolicy(FeedForwardPolicy):
    """
    Policy object that implements actor critic, using a CNN (the nature CNN)

    :param sess: (TensorFlow session) The current TensorFlow session
    :param ob_space: (Gym Space) The observation space of the environment
    :param ac_space: (Gym Space) The action space of the environment
    :param n_env: (int) The number of environments to run
    :param n_steps: (int) The number of steps to run for each environment
    :param n_batch: (int) The number of batch to run (n_envs * n_steps)
    :param reuse: (bool) If the policy is reusable or not
    :param _kwargs: (dict) Extra keyword arguments for the nature CNN feature extraction
    """

    def __init__(self, sess, ob_space, ac_space, n_env, n_steps, n_batch, reuse=False, **_kwargs):
        super(MultiCnnPolicy, self).__init__( sess,ob_space, ac_space, n_env, n_steps, n_batch, reuse,
                                        cnn_extractor=multi_cnn, feature_extraction="cnn", **_kwargs)
```

and make sure to register it 

```
_policy_registry = {
    ActorCriticPolicy: {
        "CnnPolicy": CnnPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "CnnLnLstmPolicy": CnnLnLstmPolicy,
        "MlpPolicy": MlpPolicy,
        "MlpLstmPolicy": MlpLstmPolicy,
        "MlpLnLstmPolicy": MlpLnLstmPolicy,
        "MyCnnPolicy": MyCnnPolicy,
        "MultiCnnPolicy":MultiCnnPolicy,
    }
}
```

#Runnng the simulation with Keras-rl

Install keras-rl from https://github.com/keras-rl/keras-rl

The Prioritized Experience Replay isn't available by default by keras-rl. In order to add the PER support, you need to replace the following files in the installed keras-rl folder 

polocy.py
memory.py
utils.py

Special thanks to @jakegrigsby for his help with the PER. I've included the files with PER in keras-rl folder. 

Another modification is required to add multiobservation support to keras-rl. In dqn.py, the method below needs to be edited in the following mannaer

```import numpy as np
    def process_state_batch(self, batch):
        batch_samples = []
        batch = np.array(batch)
 
        if self.processor is None:
            return batch
        batch=self.processor.process_state_batch(batch)
        for i in batch:
            #print('before',i.shape) 
            i=np.squeeze(i,axis=1)
            
            batch_samples.append(np.array(i))
            #print('after',i.shape)
        batch =  batch_samples 
        return batch```

dqn.py is also included in keras-rl folder for convenience. 

