import tensorflow as tf

class FracOptimizer(tf.keras.optimizers.Optimizer):
    def __init__(self, learning_rate=0.01, alpha=0.5, name="FracOptimizer", **kwargs):
        super().__init__(learning_rate=learning_rate, name=name, **kwargs)
        self.alpha = alpha

    def update_step(self, grad, var, learning_rate=None):
        lr = self._learning_rate if learning_rate is None else learning_rate
        update = tf.sign(grad) * tf.pow(tf.abs(grad) + 1e-8, self.alpha)
        var.assign_sub(lr * update)


    def get_config(self):
        config = super().get_config()
        config.update({
            "learning_rate": float(self._learning_rate),  # <- aqui!
            "alpha": self.alpha
        })
        return config


