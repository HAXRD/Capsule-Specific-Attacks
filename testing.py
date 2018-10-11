
import tensorflow as tf 
from input_data.cifar10 import cifar10_input
from models import cnn_model

models = {
    'cnn': cnn_model.CNNModel,
}

def default_hparams():
  """Builds an HParam object with default hyperparameters."""
  return tf.contrib.training.HParams(
      decay_rate=0.96,
      decay_steps=2000,
      leaky=False,
      learning_rate=0.001,
      loss_type='margin',
      num_prime_capsules=32,
      padding='VALID',
      remake=True,
      routing=3,
      verbose=False,
  )

if __name__ == '__main__':
    with tf.Graph().as_default():
        batched_dataset = cifar10_input.inputs('test', '/Users/xu/Downloads/cifar-10-batches-bin', 2)
        batched_dataset_iterator = batched_dataset.make_one_shot_iterator()

        # build the model 
        model = models['cnn'](default_hparams())
        model.build_model_on_multi_gpus(2)
    pass
