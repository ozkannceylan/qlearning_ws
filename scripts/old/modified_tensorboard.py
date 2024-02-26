import tensorflow as tf
from  keras.callbacks import TensorBoard
import os

# Own version of tensorboard
class ModifiedTensorBoard(TensorBoard):

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.step = 1
		self.writer = tf.summary.create_file_writer(self.log_dir)
		self._log_write_dir = self.log_dir

	def set_model(self, model):
		self.model = model

		self._train_dir = os.path.join(self._log_write_dir, 'train')
		self._train_step = self.model._train_counter

		self._val_dir = os.path.join(self._log_write_dir, 'validation')
		self._val_step = self.model._test_counter

		self._should_write_train_graph = False

	# Save logs with our step number and not from 0th step every time
	def on_epoch_end(self, epoch, logs=None):
		self.update_stats(**logs)

	# Train for one batch only, no need to save anything at epoch end
	def on_batch_end(self, batch, logs=None):
		pass

	# Won't close writer
	def on_train_end(self, _):
		pass

	def update_stats(self, **stats):
		self._write_logs(stats, self.step)

	def update_stats(self, **stats):
		with self.writer.as_default():
			for key, value in stats.items():
				tf.summary.scalar(key, value, step=self.step)