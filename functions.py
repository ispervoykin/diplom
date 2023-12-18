import math

def generatePoisson(lambd: int, r: int) -> list[list[int], list[int]]:
	"""
	Parameters
	----------
	lambd : int
			(>= 0) - expected rate of occurences
	r : int
			(> 0) - amount of resources in the model

	Returns
	-------
	list[list[int], list[int]]
			two lists, containing n amounts of busy resources from 0 to r in the first, and their CDF probabilities in the second
	"""
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		probability = math.pow(lambd, k) * math.exp(-lambd) / math.factorial(k)
		spentResourcesProbs.append(accumulation + probability)
		accumulation += probability
	return numOfSpentResources, spentResourcesProbs

def generateGeometric(p: float, r: int) -> list[list[int], list[int]]:
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		probability = math.pow(1-p, k-1) * p
		spentResourcesProbs.append(accumulation + probability)
		accumulation += probability
	return numOfSpentResources, spentResourcesProbs

def generateBinomial(p: float, n: int, r: int) -> list[list[int], list[int]]:
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		if n >= k:
			probability = (math.factorial(n) / (math.factorial(k) * math.factorial(n-k))) * math.pow(1-p, n-k) * math.pow(p, k)
			spentResourcesProbs.append(accumulation + probability)
			accumulation += probability
		else:
			spentResourcesProbs.append(1)

	return numOfSpentResources, spentResourcesProbs

def getR(r: int, distribution: str):
	"""

	Parameters
	----------
	r : int
			(>= 0) - amount of resources in the model
	distribution: str
			either of the three: "poisson", "geometric", or "binomial"

	Returns
	-------
	function
			function-generator that returns the number of spent resources
	"""
	match distribution:
		case "poisson":
			numOfSpentResources, spentResourcesProbs = generatePoisson(5, r)
		case "geometric":
			numOfSpentResources, spentResourcesProbs = generateGeometric(1/r, r)
		case "binomial":
			numOfSpentResources, spentResourcesProbs = generateBinomial(0.6, 5, r)
		case _:
			print("no matching distribution found")
			exit(-1)
	
	l = len(spentResourcesProbs)
	def inner_func(p: int) -> int:
		print(numOfSpentResources, spentResourcesProbs)
		if p < spentResourcesProbs[0]:
			return numOfSpentResources[0]
		lastProb = spentResourcesProbs[-1]
		for i in range(1, l):
			if p < spentResourcesProbs[i]:
				return numOfSpentResources[i-1]
		return numOfSpentResources[l-1]
	
	return inner_func

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
	