import random
import numpy as np
import time
import matplotlib.pyplot as plt
import heapq
import functions
import numpy as np

time_start = time.perf_counter()

def imitation(lambd, mu, N, R, totalPackets, get_r):
	numOfBlockages = 0

	arrivalTimes = []  # [arrival time, resources spent]
	exitTimes = []  # [exit time, resources spent]
	heapq.heapify(arrivalTimes)
	heapq.heapify(exitTimes)

	weightedRs = []

	currentTime = 0
	currentSpentResources = 0
	prevResources = 0
	previousTime = 0
	nextPacketArrivalTime = np.random.exponential(1/lambd)
	for _ in range(0, totalPackets):
		currentTime += nextPacketArrivalTime
		# process the previous packets
		while len(exitTimes) > 0 and exitTimes[0][0] <= currentTime:
			if len(arrivalTimes) > 0 and arrivalTimes[0][0] < exitTimes[0][0]:
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
		currentPacketResourceNeeds = get_r(random.random())
		if currentPacketResourceNeeds + currentSpentResources <= R and len(exitTimes) + 1 <= N:
			currentPacketServicingTime = np.random.exponential(1/mu)
			heapq.heappush(exitTimes, [currentTime + currentPacketServicingTime, currentPacketResourceNeeds])
			heapq.heappush(arrivalTimes, [currentTime, currentPacketResourceNeeds])
			currentSpentResources += currentPacketResourceNeeds
		else:
			numOfBlockages += 1
		nextPacketArrivalTime = np.random.exponential(1/lambd)

	while len(exitTimes) > 0:
		if len(arrivalTimes) > 0 and arrivalTimes[0][0] < exitTimes[0][0]:
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

def thousandRounds(lambd, mu, N, R, totalPackets, j, get_r):
	blockageRates = []
	avgResourcesSpentPerStep = []
	for i in range(100):
		currRound = imitation(lambd, mu, N, R, totalPackets, get_r)
		blockageRates.append(currRound[0])
		avgResourcesSpentPerStep.append(currRound[1])
		print(i + j*1000)

	return np.average(blockageRates), np.average(avgResourcesSpentPerStep) 

def runImitation(mu, lambdas, N, R, totalPackets, distribution):
	# runs the imitation 1000 times and returns blockage rates and average spent resources for a given distribution

	results = []

	for j, lambd in enumerate(lambdas):
		results.append(thousandRounds(lambd, mu, N, R, totalPackets, j, functions.getR(R, distribution)))
	
	return [result[0] for result in results], [result[1] for result in results]

N, R = 3, 3
totalPackets = 100
mu = 1
lambdas = np.linspace(0.001, 20, 1000)
ro = lambdas / mu

poissonResults = runImitation(mu, lambdas, N, R, totalPackets, "poisson")
geometricResults = runImitation(mu, lambdas, N, R, totalPackets, "geometric")
binomialResults = runImitation(mu, lambdas, N, R, totalPackets, "binomial")

x_ticks = np.arange(0, 21, 1)
y_ticks = np.arange(0, 1.05, 0.05)
plt.plot(ro, poissonResults[0], 'purple', label="Пуассоновское распределение")
plt.plot(ro, geometricResults[0], 'green', label="Геометрическое распределение")
plt.plot(ro, binomialResults[0], 'blue', label="Биномиальное распределение")
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
plt.plot(ro, poissonResults[1], 'purple', label="Пуассоновское распределение")
plt.plot(ro, geometricResults[1], 'green', label="Геометрическое распределение")
plt.plot(ro, binomialResults[1], 'blue', label="Биномиальное распределение")
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