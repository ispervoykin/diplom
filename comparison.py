import random
import numpy as np
import time
import matplotlib.pyplot as plt
import heapq
import functions
import numpy as np

import numpy as np
import matplotlib.pyplot as plt
import math
from datetime import datetime as dt


def create_distribution(r: int, tetta: float) -> list[float]:
    distribution = [np.exp(-tetta)*tetta**i/math.factorial(i) if i > 0 else 0 for i in range(r + 1)]
    distribution /= sum(distribution)
    return distribution


all_convs = {}


def conv(r: int, k: int) -> float:
    if k <= 0:
        return 0
    if k == 1:
        return p[r]
    if a := all_convs.get(r): # для хотя бы какого то ускорения программы
        if a.get(k):
            return all_convs[r][k]
    result = 0
    for i in range(r + 1):
        result += p[i] * conv(r - i, k - 1)
    all_convs[r] = {k: result}
    return result


def solve_sur(N: int, R: int, ld: int, mu: int) -> list:
    right_part = [1] # в начале правой части единица
    right_part2 = [0] * (N * R * (R+1)) # добавляем в правую часть нули
    right_part += right_part2 # создаем вектор правых частей размера N*R*(R+1) так как r2 от 0 до 3
    #right_part.append(1) # добавляем в конец вектора единицу, там будет нормировка, так как на предыдущем шаге не учитывается состояние 0,0,0
    right_part = np.array(right_part) # итого вектор правых частей размера N*R*(R+1)+1

    matrix = []
    first_col = [1] * (N * R * (R+1) + 1) #первое заменяем на единицы
    # first_col[0] = -ld # состояние 0,0,0
    # for j in range(1, R + 1):
    #     first_col[1+(R+1)*(j-1)] = mu #см тетрадь преобразование трех чисел в нумерацию

    matrix.append(first_col)
    for i in range(N*R*(R+1)): #делаем матрицу, далее заполняем по пордяку см тетрадь
        matrix.append(0)
    for k in range(1, N):
        for r in range(1, R + 1):
            cur_col = [0] * (N * R * (R+1) + 1)
            cur_col[1+(R+1)*(r-1)+(k-1)*R*(R+1)] = -(ld * sum(p[:(R + 1)]) + k * mu)
            if r >= k: # прямое ограничение на состояния где заявки занимают меньшее количество ресурсов, чем количество заявок
                for j in range(0, r + 1):
                    if k - 1 == 0 and j == 0:
                        cur_col[0] = ld * p[r - j]
                    elif k - 1 > 0 and j > 0:
                        cur_col[1+(R+1)*(j-1)+(k-2)*R*(R+1)] = ld * p[r - j]

                for i in range(1, R + 1):
                    for s in range(R - i + 1, R+1):
                        if r >= s:
                            # if (con := conv(i, k)) == 0:
                            #     cur_col[1+s+(R+1)*(i-1)+(k-1)*R*(R+1)] = 0
                            # else:
                            #     cur_col[1+s+(R+1)*(i-1)+(k-1)*R*(R+1)] = k * mu * p[i+s-r] * conv(r-s, k-1) / con
                            if (con := convs[k-1][i]) == 0 or k-2 < 0:
                                cur_col[1 + s + (R + 1) * (i - 1) + (k - 1) * R * (R + 1)] = 0
                            else:
                                cur_col[1 + s + (R + 1) * (i - 1) + (k - 1) * R * (R + 1)] = k * mu * p[i + s - r] * convs[k-2][r-s] / con

                for j in range(0, R - r + 1):
                    if r + j > 0:
                        # if (s := conv(r + j, k + 1)) == 0:
                        #     cur_col[1+(R+1)*(r+j-1)+k*R*(R+1)] = 0
                        # else:
                        #     cur_col[1+(R+1)*(r+j-1)+k*R*(R+1)] = (k + 1) * mu * p[j] * conv(r, k) / s
                        if (s := convs[k][r+j]) == 0 or k-1 < 0:
                            cur_col[1+(R+1)*(r+j-1)+k*R*(R+1)] = 0
                        else:
                            cur_col[1+(R+1)*(r+j-1)+k*R*(R+1)] = (k + 1) * mu * p[j] * convs[k-1][r] / s

            matrix[1+(R+1)*(r-1)+(k-1)*R*(R+1)] = cur_col
            #matrix.append(cur_col)
    #print(f'Первое уравн {dt.time(dt.now())}')

    for k in range(1, N):
        for r in range(1, R+1):
            for s in range(1, R + 1): # попробую так, нужно изменить на условие которое там, но чтобы осталось нужное количество столбцов
                cur_col = [0] * (N * R * (R + 1) + 1)
                cur_col[1+s+(R+1)*(r-1)+(k-1)*R*(R+1)] = -k * mu
                if R-r+1 <= s <= R and r >= k: #ограничение из правой части уравнения плюс прямое ограничение на состояния где заявки занимают меньшее количество ресурсов, чем количество заявок
                    for j in range(1, R - r + 1):
                        # if (con := conv(r+j, k+1)) == 0:
                        #     cur_col[1+s+(R+1)*(r+j-1)+k*R*(R+1)] = 0
                        # else:
                        #     cur_col[1 + s + (R + 1) * (r + j - 1) + k * R * (R + 1)] = (k + 1) * mu * p[j] * conv(r, k) / con
                        if (con := convs[k][r+j]) == 0 or k-1<0:
                            cur_col[1+s+(R+1)*(r+j-1)+k*R*(R+1)] = 0
                        else:
                            cur_col[1 + s + (R + 1) * (r + j - 1) + k * R * (R + 1)] = (k + 1) * mu * p[j] * convs[k-1][r] / con

                    cur_col[1+(R+1)*(r-1)+(k-1)*R*(R+1)] = ld * p[s]

                matrix[1+s+(R+1)*(r-1)+(k-1)*R*(R+1)] = cur_col
                #matrix.append(cur_col)
    #print(f'Второе {dt.time(dt.now())}')
    length = len(matrix)

    for r in range(1, R + 1):
        cur_col = [0] * (N * R * (R + 1) + 1)
        cur_col[1 + (R + 1) * (r-1) + (N-1)*R*(R+1)] = -(ld + N*mu)
        if r >= N: #прямое ограничение на состояния где заявки занимают меньшее количество ресурсов, чем количество заявок
            for j in range(0, r + 1):
                cur_col[1+(R+1)*(j-1)+(N-2)*R*(R+1)] = ld * p[r - j]

            for i in range(1, R + 1):
                for s in range(R - i + 1, R + 1):
                    if r >= s:
                        # if (con := conv(i, N)) == 0:
                        #     cur_col[1 + s + (R + 1) * (i - 1) + (N - 1) * R * (R + 1)] = 0
                        # else:
                        #     cur_col[1 + s + (R + 1) * (i - 1) + (N - 1) * R * (R + 1)] = N * mu * p[i + s - r] * conv(r - s,N - 1) / con
                        if (con := convs[N][i]) == 0 or N-2 < 0:
                            cur_col[1 + s + (R + 1) * (i - 1) + (N - 1) * R * (R + 1)] = 0
                        else:
                            cur_col[1 + s + (R + 1) * (i - 1) + (N - 1) * R * (R + 1)] = N * mu * p[i + s - r] * convs[N-2][r-s] / con

        matrix[1 + (R + 1) * (r-1) + (N-1)*R*(R+1)] = cur_col
    #print(f'Третье уравн {dt.time(dt.now())}')
        #matrix.append(cur_col)

    for r in range(1, R + 1):
        for s in range(1, R+1):
            cur_col = [0] * (N * R * (R + 1) + 1)
            cur_col[1 + s +(R+1)*(r-1) + (N-1)*R*(R+1)] = -N * mu
            if r >= N: # прямое ограничение на состояния где заявки занимают меньшее количество ресурсов, чем количество заявок
                cur_col[1 + (R+1)*(N-1) + (N-1)*R*(R+1)] = ld*sum(p[:r+1])

            matrix[1 + s +(R+1)*(r-1) + (N-1)*R*(R+1)] = cur_col
            #matrix.append(cur_col)
    #print(f'Четвертое уравн {dt.time(dt.now())}')

    A = np.matrix(matrix).T
    answer = right_part.dot(np.linalg.inv(A))
    answer = answer.tolist()[0]
    return answer
    

time_start = time.perf_counter()

def imitation(lambd, mu, N, R, totalPackets, get_r, Q):
	numOfBlockages = 0

	arrivalTimes = []  # [arrival time, resources spent]
	exitTimes = []  # [exit time, resources spent]
	heapq.heapify(arrivalTimes)
	heapq.heapify(exitTimes)

	weightedRsSum = 0
	queue = functions.Queue(Q)  # resources spent

	currentTime = 0
	currentSpentResources = 0
	prevResources = 0
	previousTime = 0
	nextPacketArrivalTime = np.random.exponential(1/lambd)
	for _ in range(0, totalPackets):
		currentTime += nextPacketArrivalTime
		# process the packets in service
		while len(exitTimes) > 0 and exitTimes[0][0] <= currentTime:
			if len(arrivalTimes) > 0 and arrivalTimes[0][0] < exitTimes[0][0]:
				currTime = arrivalTimes[0][0]
				weightedRsSum += prevResources * (currTime - previousTime)
				previousTime = currTime
				prevResources += heapq.heappop(arrivalTimes)[1]
			else:
				currTime = exitTimes[0][0]
				weightedRsSum += prevResources * (currTime - previousTime)
				previousTime = currTime
				currentSpentResources -= exitTimes[0][1]
				prevResources -= heapq.heappop(exitTimes)[1]

				# After servicing a packet, pass another from the queue
				if len(queue) > 0 and queue[0] + currentSpentResources <= R:
					currentPacketServicingTime = np.random.exponential(1/mu)
					top_element = queue.pop_front()
					heapq.heappush(arrivalTimes, [currTime, top_element])
					heapq.heappush(exitTimes, [currTime + currentPacketServicingTime, top_element])
					currentSpentResources += top_element

		# add the current packet to the queue
		currentPacketResourceNeeds = get_r(random.random())
		if len(queue) < Q:
			queue.append(currentPacketResourceNeeds)
		else:
			numOfBlockages += 1

		# pass elements from the queue to service
		while len(queue) > 0 and queue[0] + currentSpentResources <= R and len(exitTimes) < N:
			currentPacketServicingTime = np.random.exponential(1/mu)
			top_element = queue.pop_front()
			heapq.heappush(arrivalTimes, [currentTime, top_element])
			heapq.heappush(exitTimes, [currentTime + currentPacketServicingTime, top_element])
			currentSpentResources += top_element

		nextPacketArrivalTime = np.random.exponential(1/lambd)

	while len(exitTimes) > 0:
		if len(arrivalTimes) > 0 and arrivalTimes[0][0] < exitTimes[0][0]:
			currentTime = arrivalTimes[0][0]
			weightedRsSum += prevResources * (currentTime - previousTime)
			previousTime = currentTime
			prevResources += heapq.heappop(arrivalTimes)[1]
		else:
			currentTime = exitTimes[0][0]
			weightedRsSum += prevResources * (currentTime - previousTime)
			previousTime = currentTime
			currentSpentResources -= exitTimes[0][1]
			prevResources -= heapq.heappop(exitTimes)[1]
			if len(queue) > 0 and queue[0] + currentSpentResources <= R:
				currentPacketServicingTime = np.random.exponential(1/mu)
				top_element = queue.pop_front()
				heapq.heappush(arrivalTimes, [currentTime, top_element])
				heapq.heappush(exitTimes, [currentTime + currentPacketServicingTime, top_element])
				currentSpentResources += top_element

	return numOfBlockages / totalPackets, weightedRsSum / currentTime

def thousandRounds(lambd, mu, N, R, totalPackets, j, get_r, Q):
	blockageRates = []
	avgResourcesSpentPerStep = []
	for i in range(1000):
		currRound = imitation(lambd, mu, N, R, totalPackets, get_r, Q)
		blockageRates.append(currRound[0])
		avgResourcesSpentPerStep.append(currRound[1])
		print(i + j*1000)

	return np.average(blockageRates), np.average(avgResourcesSpentPerStep) 

def runImitation(mu, lambdas, N, R, totalPackets, distribution, Q):
	# runs the imitation 1000 times and returns blockage rates and average spent resources for a given distribution

	results = []

	for j, lambd in enumerate(lambdas):
		results.append(thousandRounds(lambd, mu, N, R, totalPackets, j, functions.getR(R, distribution), Q))
	
	return [result[0] for result in results], [result[1] for result in results]

'--------------------------------------------------------------------------------------------------'

# для преобразования qk(r) в номера столбцов


N = 3
R = 3
ld = 1
mu = 1
tetta = 5
#p = [0, 0.3, 0.5, 0.2]
p = create_distribution(R, tetta)

convs = [[0 for i in range(R + 1)] for j in range(N+1)]
for r in range(0, R+1):
    convs[0][r]= p[r]

for k in range(1, N+1):
    for r in range(0, R+1):
        summ = 0
        for j in range(0, r + 1):
            if k-1 >= 0:
                summ += convs[k-1][j]*p[r-j]
        convs[k][r] = summ

answer = solve_sur(N, R, ld, mu)
# for i, val in enumerate(p):
#     print(f"p{i}={val}")
# for k in range(0, N):
#     for r in range(0, R+1):
#         print(f'p{r}({k+1}) = {convs[k][r]}')
# print(sum(answer))
# print(f"({0}, {0}, {0}) -> {answer[0]}")
# for i in range(1, N + 1):
#     for j in range(1, R + 1):
#         for l in range(0, R + 1):
#             print(f"({i}, {j}, {l}) -> {answer[1 + l + (R + 1) * (j - 1) + (i - 1) * R * (R + 1)]}")

pb = answer[0] * 1 # сразу присуммируем q0(0) умноженное на 1(т.к складываются все pj)
n_mean = 0
r_mean = 0
for n in range(1, N + 1):
    for r in range(1, R + 1):
        for s in range(0, R + 1):
            if s == 0:
                n_mean += answer[1 + s + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)] * n
            else:
                n_mean += answer[1 + s + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)] * (n+1)
            r_mean += answer[1 + s + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)] * r

for n in range(1, N + 1):
    for r in range(1, R + 1):
        pb = pb + answer[1 + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)]

print(f'Вероятности: {answer}')
print(f'Вероятность потери: {1 - pb}')
print(f'Среднее число заявок в системе: {n_mean}')
print(f'Среднее число занятого ресурса: {r_mean}')

'--------------------------------------------------------------------------------------------------'




N, R, Q = 3, 3, 3
totalPackets = 1000
mu = 1
lambdas = np.linspace(0.001, 20, 1000)
ro = lambdas / mu

poissonResults = runImitation(mu, lambdas, N, R, totalPackets, "poisson", Q)

X = np.linspace(0.001, 20, 1000)
Y = list()
for x in X:
    ans = solve_sur(N, R, x, mu)
    pb = ans[0] * 1
    for n in range(1, N + 1):
        for r in range(1, R + 1):
            pb += ans[1 + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)]
    Y.append(1-pb)

x_ticks = np.arange(0, 21, 1)
y_ticks = np.arange(0, 1.05, 0.05)
plt.plot(ro, poissonResults[0], 'blue', label="Имитация")
plt.plot(X, Y, 'red', label='Математическая модель')
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

Y = list()
for x in X:
    ans = solve_sur(N, R, x, mu)
    n_mean = 0
    for n in range(1, N + 1):
        for r in range(1, R + 1):
            for s in range(0, R + 1):
                if s == 0:
                    n_mean += ans[1 + s + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)] * n
                else:
                    n_mean += ans[1 + s + (R + 1) * (r - 1) + (n - 1) * R * (R + 1)] * (n + 1)
    Y.append(n_mean)
plt.plot(X, Y)

y_ticks = np.arange(0, R + R / 20, R / 20)
plt.plot(ro, poissonResults[1], 'blue', label="Имитация")
plt.plot(X, Y, 'red', label='Математическая модель')
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