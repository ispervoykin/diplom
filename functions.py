import math

def poisson(k, lambd):
	if k > 0 and k == int(k):
		return lambd ** k * math.exp(-lambd) / math.factorial(k)
	return 0

def exponential(k, lambd):
	if k > 0 and k == int(k):
		return lambd * math.exp(-lambd*k)
	return 0

class Queue():
	q = []

	def __init__(self, size) -> None:
		self.size = size

	def append(self, elem):
		if len(self.q) < self.size:
			self.q.append(elem)
			return True
		
		return False
	
	def pop_front(self):
		return self.q.pop(0)
	
	def get(self):
		return self.q
	