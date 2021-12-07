import zipfile
import math

file = zipfile.ZipFile('/Users/adriangotca/Downloads/TSP-D-Instances-v1.2.zip')


def solve(input_file, tsp_file, dp_file):
    truck_cost, drone_cost, N = read_input(input_file)
    tsp = read_tsp(tsp_file)
    print(tsp)

    T = [[[0 for _ in range(N+2)] for _ in range(N+2)] for _ in range(N+2)]
    T2 = [[0 for _ in range(N+2)] for _ in range(N+2)]
    M = [[0 for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        T[i][i+1][0] = truck_cost[tsp[i]][tsp[i+1]]
        T2[i][i+1] = T[i][i+1][0]
    partial_sums = []
    partial_sums.append(0)
    for i in range(1, N+1):
        partial_sums.append(truck_cost[tsp[i]][tsp[i+1]]+partial_sums[-1])
    for i in range(1, N+1):
        for k in range(i+1, N+1):
            for j in range(k+1, N+1):
                T[i][j][k] = max(drone_cost[tsp[i]][tsp[k]]+drone_cost[tsp[k]][tsp[j]], partial_sums[j] -
                                 partial_sums[i-1]-truck_cost[tsp[k-1]][tsp[k]]-truck_cost[tsp[k]][tsp[k+1]]+truck_cost[tsp[k-1]][tsp[k+1]])
    for i in range(N+1):
        for j in range(i+1, N+1):
            min_value = T[i][j][i+1]
            kmin = i+1
            for k in range(i+1, j):
                kmin = k if T[i][j][k] < min_value else kmin
                min_value = min(T[i][j][k], min_value)
            T2[i][j] = min_value
            M[tsp[i]][tsp[j]] = kmin
    V=[]
    P=[]
    V.append(0)
    V.append(0)
    P.append(0)
    P.append(0)
    for i in range(2,N+1):
        min_value = V[i-1]+T2[i-1][i]
        min_iter = i-1
        for j in range(1, i-1):
            min_iter = j if V[j]+T2[j][i]<min_value else min_iter
            min_value = min(min_value, V[j]+T2[j][i])
        V.append(min_value)
        P.append(min_iter)
    print(M)
    print(P)


    for idx, line in enumerate(file.read(dp_file).decode().split('\n')):
        if idx in [0, 2, 3]:
            continue
        if line.endswith('*/'):
            line = line.strip('/* Totalcost:')
        print(line)


def read_tsp(tsp_file):
    tsp = [0]
    for idx, line in enumerate(file.read(tsp_file).decode().split('\n')):
        if idx in [0, 1, 2, 3]:
            continue
        if line == '':
            break
        tsp.append(int(line.split('\t')[0])+1)
    tsp.append(tsp[1])
    return tsp


def get_distance(point1, point2):
    difx = point1[0]-point2[0]
    dify = point1[1]-point2[1]
    return math.sqrt(difx**2+dify**2)


def read_input(input_file):
    X = []
    Y = []
    for idx, line in enumerate(file.read(input_file).decode().split('\n')):
        if idx in [0, 2, 4, 6, 8]:
            continue
        if line == '':
            break
        if idx == 1:
            truck_speed = float(line)
        elif idx == 3:
            drone_speed = float(line)
        elif idx == 5:
            N = int(line)
        else:
            X.append(float(line.split(' ')[0]))
            Y.append(float(line.split(' ')[1]))
    truck_cost = [[0 for _ in range(N+1)] for _ in range(N+1)]
    drone_cost = [[0 for _ in range(N+1)] for _ in range(N+1)]
    for i in range(1, N+1):
        for j in range(i, N+1):
            dist = get_distance((X[i-1], Y[i-1]), (X[j-1], Y[j-1]))
            truck_cost[i][j] = truck_cost[j][i] = dist/truck_speed
            drone_cost[i][j] = drone_cost[j][i] = dist/drone_speed
    return truck_cost, drone_cost, N


if __name__ == '__main__':
    files = file.namelist()

    txt_files = list(filter(lambda name: name.endswith('.txt'), files))
    no_ext_files = list(map(lambda name: name.rstrip('.txt'), txt_files))
    filenames = list(map(lambda name: name.split('/')[-1], no_ext_files))
    tsp_files = list(filter(lambda name: name.endswith('-tsp'), filenames))
    dp_files = list(filter(lambda name: name.endswith('-DP'), filenames))

    test_files = list(filter(lambda name: name +
                             '-tsp' in tsp_files and name+'-DP' in dp_files, filenames))
    for f in test_files:
        input_file_to_open = list(
            filter(lambda filename: filename.find(f+'.txt') != -1, files))[0]
        tsp_file_to_open = list(
            filter(lambda filename: filename.find(f+'-tsp.txt') != -1, files))[0]
        dp_file_to_open = list(
            filter(lambda filename: filename.find(f+'-DP.txt') != -1, files))[0]
        solve(input_file_to_open, tsp_file_to_open, dp_file_to_open)
        break
