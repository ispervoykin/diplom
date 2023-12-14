import random
import numpy as np
import time
import matplotlib.pyplot as plt
import functions
import heapq
import numpy as np

time_start = time.perf_counter()

totalPackets = 100

def imitation(lambd, mu, N, R):
	numOfBlockages = 0

	arrivalTimes = []  # time, resources spent at the time
	exitTimes = []  # time, resources spent at the time
	heapq.heapify(arrivalTimes)
	heapq.heapify(exitTimes)

	weightedRs = []

	currentTime = 0
	currentSpentResources = 0
	ResourcesOtrezki = 0
	previousTime = 0
	nextPacketArrivalTime = np.random.exponential(1/lambd)
	for i in range(1, totalPackets+1):
		currentTime += nextPacketArrivalTime
		# process the previous packets
		while (bool(exitTimes) and exitTimes[0][0] <= currentTime):
			if bool(arrivalTimes) and arrivalTimes[0][0] < exitTimes[0][0]:
				currTime = arrivalTimes[0][0]
				weightedRs.append(ResourcesOtrezki * (currTime - previousTime))
				previousTime = currTime
				ResourcesOtrezki += heapq.heappop(arrivalTimes)[1]
			else:
				currTime = exitTimes[0][0]
				weightedRs.append(ResourcesOtrezki * (currTime - previousTime))
				previousTime = currTime
				currentSpentResources -= exitTimes[0][1]
				ResourcesOtrezki -= heapq.heappop(exitTimes)[1]

		# process the current packet
		currentPacketResourceNeeds = random.randint(1, R)
		if (currentPacketResourceNeeds + currentSpentResources <= R and len(exitTimes) + 1 <= N):
			currentPacketServicingTime = np.random.exponential(1/mu)
			heapq.heappush(exitTimes, [currentTime + currentPacketServicingTime, currentPacketResourceNeeds])
			heapq.heappush(arrivalTimes, [currentTime, currentPacketResourceNeeds])
			currentSpentResources += currentPacketResourceNeeds
		else:
			numOfBlockages += 1
		nextPacketArrivalTime = np.random.exponential(1/lambd)

	while (bool(exitTimes)):
		if bool(arrivalTimes) and arrivalTimes[0][0] < exitTimes[0][0]:
			currentTime = arrivalTimes[0][0]
			weightedRs.append(ResourcesOtrezki * (currentTime - previousTime))
			previousTime = currentTime
			ResourcesOtrezki += heapq.heappop(arrivalTimes)[1]
		else:
			currentTime = exitTimes[0][0]
			weightedRs.append(ResourcesOtrezki * (currentTime - previousTime))
			previousTime = currentTime
			ResourcesOtrezki -= heapq.heappop(exitTimes)[1]

	return numOfBlockages / totalPackets, sum(weightedRs) / currentTime

def thousandRounds(lambd, mu, N, R, j):
	blockageRates = []
	avgResourcesSpentPerStep = []
	for i in range(100):
		currRound = imitation(lambd, mu, N, R)
		blockageRates.append(currRound[0])
		avgResourcesSpentPerStep.append(currRound[1])
		print(i + j*1000)

	return np.average(blockageRates), np.average(avgResourcesSpentPerStep) 


mu = 1
lambdas = np.linspace(0, 20, 1000)
ro = lambdas / mu

N, R = 3, 3

results = []
for j, lambd in enumerate(lambdas):
	results.append(thousandRounds(lambd, mu, N, R, j))

blockageRates = [result[0] for result in results]
avgSpentResources = [result[1] for result in results]

x_ticks = np.arange(0, 21, 1)
y_ticks = np.arange(0, 3.25, 0.25)
plt.plot(ro, blockageRates)
plt.xlabel("Нагрузка ρ", fontsize=16)
plt.ylabel("Вероятность отказа π", fontsize=16)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
plt.title("График зависимости вероятности отказа π от нагрузки ρ", fontsize=16)
plt.grid(True)
plt.show()

plt.plot(ro, avgSpentResources)
plt.xlabel("Нагрузка ρ", fontsize=16)
plt.ylabel("Средний объём занятого ресурса R", fontsize=16)
plt.xticks(x_ticks)
plt.yticks(y_ticks)
plt.tick_params(axis="x", labelsize=14)
plt.tick_params(axis="y", labelsize=14)
plt.title("График зависимости среднего объёма занятого ресурса R от нагрузки ρ", fontsize=16)
plt.grid(True)
plt.show()

time_end = time.perf_counter()
print(time_end - time_start)