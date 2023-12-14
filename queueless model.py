import random
import numpy as np
import time
import matplotlib.pyplot as plt
import heapq
import numpy as np

time_start = time.perf_counter()

def imitation(mu, N, R, totalPackets, distribution, args):
	numOfBlockages = 0

	arrivalTimes = []  # time, resources spent at the time
	exitTimes = []  # time, resources spent at the time
	heapq.heapify(arrivalTimes)
	heapq.heapify(exitTimes)

	weightedRs = []

	currentTime = 0
	currentSpentResources = 0
	prevResources = 0
	previousTime = 0
	nextPacketArrivalTime = distribution(*args)
	for i in range(1, totalPackets+1):
		currentTime += nextPacketArrivalTime
		# process the previous packets
		while (bool(exitTimes) and exitTimes[0][0] <= currentTime):
			if bool(arrivalTimes) and arrivalTimes[0][0] < exitTimes[0][0]:
				currTime = arrivalTimes[0][0]
				weightedRs.append(prevResources * (currTime - previousTime))
				previousTime = currTime
				prevResources += heapq.heappop(arrivalTimes)[1]
			else:
				currTime = exitTimes[0][0]
				weightedRs.append(prevResources * (currTime - previousTime))
				previousTime = currTime
				currentSpentResources -= exitTimes[0][1]
				prevResources -= heapq.heappop(exitTimes)[1]

		# process the current packet
		currentPacketResourceNeeds = random.randint(1, R)
		if (currentPacketResourceNeeds + currentSpentResources <= R and len(exitTimes) + 1 <= N):
			currentPacketServicingTime = np.random.exponential(1/mu)
			heapq.heappush(exitTimes, [currentTime + currentPacketServicingTime, currentPacketResourceNeeds])
			heapq.heappush(arrivalTimes, [currentTime, currentPacketResourceNeeds])
			currentSpentResources += currentPacketResourceNeeds
		else:
			numOfBlockages += 1
		nextPacketArrivalTime = distribution(*args)

	while (bool(exitTimes)):
		if bool(arrivalTimes) and arrivalTimes[0][0] < exitTimes[0][0]:
			currentTime = arrivalTimes[0][0]
			weightedRs.append(prevResources * (currentTime - previousTime))
			previousTime = currentTime
			prevResources += heapq.heappop(arrivalTimes)[1]
		else:
			currentTime = exitTimes[0][0]
			weightedRs.append(prevResources * (currentTime - previousTime))
			previousTime = currentTime
			prevResources -= heapq.heappop(exitTimes)[1]

	return numOfBlockages / totalPackets, sum(weightedRs) / currentTime

def thousandRounds(mu, N, R, totalPackets, j, distribution, args):
	blockageRates = []
	avgResourcesSpentPerStep = []
	for i in range(1000):
		currRound = imitation(mu, N, R, totalPackets, distribution, args)
		blockageRates.append(currRound[0])
		avgResourcesSpentPerStep.append(currRound[1])
		print(i + j*1000)

	return np.average(blockageRates), np.average(avgResourcesSpentPerStep) 

def runImitation(mu, lambdas, N, R, totalPackets, distribution):
	# runs the imitation 1000 times and returns blockage rates and average spent resources for a given distribution

	results = []

	match distribution:
		case np.random.poisson:
				for j, lambd in enumerate(lambdas):
					results.append(thousandRounds(mu, N, R, totalPackets, j, np.random.poisson, [1/(lambd+1)]))
		case np.random.geometric:
						for j, lambd in enumerate(lambdas):
							results.append(thousandRounds(mu, N, R, totalPackets, j, np.random.geometric, [1/(lambd+1)]))
		case np.random.binomial:
				for j, lambd in enumerate(lambdas):
					results.append(thousandRounds(mu, N, R, totalPackets, j, np.random.binomial, [1, 1/(lambd+1)]))
	
	return [result[0] for result in results], [result[1] for result in results]

N, R = 3, 3
totalPackets = 1000
mu = 1
lambdas = np.linspace(0, 20, 1000)
ro = lambdas / mu

poissonResults = runImitation(mu, lambdas, N, R, totalPackets, np.random.poisson)
geometricResults = runImitation(mu, lambdas, N, R, totalPackets, np.random.geometric)
binomialResults = runImitation(mu, lambdas, N, R, totalPackets, np.random.binomial)

x_ticks = np.arange(0, 21, 1)
y_ticks = np.arange(0, 1.05, 0.05)
plt.plot(ro, poissonResults[0], 'r', label="Пуассоновское распределение")
plt.plot(ro, geometricResults[0], 'b', label="Геометрическое распределение")
plt.plot(ro, binomialResults[0], 'g', label="Биномиальное распределение")
plt.xlabel("Нагрузка ρ", fontsize=16)
plt.ylabel("Вероятность отказа π", fontsize=16)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
plt.title("График зависимости вероятности отказа π от нагрузки ρ", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.show()

y_ticks = np.arange(0, R + R / 20, R / 20)
plt.plot(ro, poissonResults[1], 'r', label="Пуассоновское распределение")
plt.plot(ro, geometricResults[1], 'b', label="Геометрическое распределение")
plt.plot(ro, binomialResults[1], 'g', label="Биномиальное распределение")
plt.xlabel("Нагрузка ρ", fontsize=16)
plt.ylabel("Средний объём занятого ресурса R", fontsize=16)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
plt.title("График зависимости среднего объёма занятого ресурса R от нагрузки ρ", fontsize=16)
plt.grid(True)
plt.legend(fontsize=16)
plt.show()

time_end = time.perf_counter()
print(time_end - time_start)