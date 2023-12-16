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
	lambd += 1
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		probability = math.pow(lambd, k) * math.exp(-lambd) / math.factorial(k)
		spentResourcesProbs.append(accumulation + probability)
		accumulation += probability
	return numOfSpentResources, spentResourcesProbs

def generateGeometric(lambd: int, r: int) -> list[list[int], list[int]]:
	p = 1 / (lambd + 1)
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		probability = math.pow(1-p, k-1) * p
		spentResourcesProbs.append(accumulation + probability)
		accumulation += probability
	return numOfSpentResources, spentResourcesProbs

def generateBinomial(lambd: int, r: int) -> list[list[int], list[int]]:
	p = 1 / (lambd + 1)
	accumulation = 0
	numOfSpentResources = []
	spentResourcesProbs = []
	for k in range(1, r+1):
		numOfSpentResources.append(k)
		probability = (math.factorial(r) / (math.factorial(k) * math.factorial(r-k))) * math.pow(1-p, r-k) * math.pow(p, k)
		spentResourcesProbs.append(accumulation + probability)
		accumulation += probability
	return numOfSpentResources, spentResourcesProbs

def getR(lambd: int, r: int, distribution: str):
	"""

	Parameters
	----------
	lambd : int
			(>= 0) - expected rate of occurences
	r : int
			(>= 0) - amount of resources in the model
	distribution: str
			either of three: "poisson", "geometric", or "binomial"

	Returns
	-------
	function
			function-generator that returns the number of spent resources
	"""
	match distribution:
		case "poisson":
			numOfSpentResources, spentResourcesProbs = generatePoisson(lambd, r)
		case "geometric":
			numOfSpentResources, spentResourcesProbs = generateGeometric(lambd, r)
		case "binomial":
			numOfSpentResources, spentResourcesProbs = generateBinomial(lambd, r)
		case _:
			print("no matching distribution found")
			exit(-1)
	
	l = len(spentResourcesProbs)
	def inner_func(p: int) -> int:
		for i in range(-1, l):
			if i < l-1 and p <= spentResourcesProbs[i+1]:
				return numOfSpentResources[i+1]
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
	