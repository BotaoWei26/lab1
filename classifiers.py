import numpy as np
import heapq

def get_MED(means):
    def MED(x):
        dist = []
        for z in means:
            g = np.dot(-z, x) + (1/2) * np.dot(z, z)
            dist.append(g)
        return np.argmin(dist)
    return MED


def get_GED(means, vars):
    Ss = [np.linalg.inv(S) for S in vars]

    def GED(x):
        dist = []
        for z, S in zip(means, Ss):
            d = np.sqrt((x-z).T @ S @ (x-z))
            dist.append(d)
        return np.argmin(dist)
    return GED


def get_normal_dist(mean, var):
    n = len(mean)
    S = np.linalg.inv(var)
    dem = (2 * np.pi) ** (n/2) * np.linalg.det(var) ** (1/2)

    def normal_dist(x):
        num = np.exp((-1/2 * (x-mean).T @ S @ (x-mean)))
        return num / dem

    return normal_dist


def get_MAP(Ns, means, vars):
    Ps = [s / sum(Ns) for s in Ns]
    normal_dists = [get_normal_dist(mean, var) for mean, var in zip(means, vars)]
    def MAP(x):
        probs = []
        for P, normal_dist in zip(Ps, normal_dists):
            p = P * normal_dist(x)
            probs.append(p)
        return np.argmax(probs)
    return MAP


def get_kNN(points, k):
    def kNN(x):
        cl_means = []
        for cl_points in points:
            cl_points = cl_points.T
            q = []
            for point in cl_points:
                dist = np.linalg.norm(point - x)
                q.append((dist, point))

            heapq.heapify(q)
            mean = np.zeros(cl_points.shape[1])
            for _ in range(k):
                q_item = heapq.heappop(q)
                mean += q_item[1]
            mean /= k
            cl_means.append(mean)

        q = []
        for cl, mean in enumerate(cl_means):
            dist = np.linalg.norm(mean - x)
            q.append((dist, cl))
        heapq.heapify(q)
        best_cl = heapq.heappop(q)[1]
        return best_cl
    return kNN

