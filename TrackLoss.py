
class TrackLoss():
	def __init__(self, interval=15):
     """
     Initialize a new interval.

     Args:
         self: (todo): write your description
         interval: (int): write your description
     """
		self.best_epoch = -1
		self.best_score = 1000
		self.last_warm = -1
                self.interval = interval 

	def update(self, loss, epoch):
     """
     Updates loss.

     Args:
         self: (todo): write your description
         loss: (todo): write your description
         epoch: (int): write your description
     """
		if loss < self.best_score:
			self.best_score = loss
			self.best_epoch = epoch
			self.last_warm = epoch

	def drop_learning_rate(self, epoch):
     """
     Removes the learning rate.

     Args:
         self: (todo): write your description
         epoch: (int): write your description
     """
		if (epoch - self.best_epoch) > self.interval and (epoch - self.last_warm) > self.interval:
			self.last_warm = epoch
			return True

		else:
			return False
