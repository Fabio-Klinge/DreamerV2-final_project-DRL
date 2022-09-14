import tensorflow as tf
import tensorflow_probability as tfp


class OneHotDist(tfp.distributions.OneHotCategorical):
    """
    Implements Straight-Through Gradients when sampling. Copied from original DreamerV2 author repository. Still outputs a LOT of incorrect warnings. Those are not important, but not easily fixed.
    """
    def __init__(self, logits: tf.Tensor = None, probs: tf.Tensor = None, dtype=None):
        self._sample_dtype = dtype or tf.float32
        super(OneHotDist, self).__init__(logits=logits, probs=probs)

    def mode(self):
        return tf.cast(super().mode(), self._sample_dtype)

    def sample(self, sample_shape=(), seed=None):
        # Straight through biased gradient estimator.
        sample = tf.cast(super().sample(sample_shape, seed), self._sample_dtype)
        probs = self._pad(super().probs_parameter(), sample.shape)
        sample += tf.cast(probs - tf.stop_gradient(probs), self._sample_dtype)
        return sample

    def _pad(self, tensor, shape):
        tensor = super().probs_parameter()
        while len(tensor.shape) < len(shape):
            tensor = tensor[None]
        return tensor
