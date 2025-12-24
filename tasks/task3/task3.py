import json
import numpy as np


# ----------------- СЛУЖЕБНЫЕ ФУНКЦИИ -----------------

def build_relation_matrix(ranking, n):
    pos = [0] * n
    cur = 0
    for block in ranking:
        if not isinstance(block, list):
            block = [block]
        for x in block:
            pos[x - 1] = cur
        cur += 1

    Y = np.zeros((n, n), dtype=int)
    for i in range(n):
        for j in range(n):
            if pos[i] >= pos[j]:
                Y[i, j] = 1
    return Y


def warshall(E):
    n = len(E)
    W = E.copy()
    for k in range(n):
        for i in range(n):
            for j in range(n):
                W[i, j] = W[i, j] or (W[i, k] and W[k, j])
    return W


def extract_clusters(W):
    n = len(W)
    used = [False] * n
    clusters = []

    for i in range(n):
        if not used[i]:
            c = []
            for j in range(n):
                if W[i, j] and W[j, i]:
                    c.append(j + 1)
                    used[j] = True
            clusters.append(sorted(c))
    return clusters


# ----------------- ОСНОВНАЯ ФУНКЦИЯ -----------------

def main(json1: str, json2: str) -> str:
    R1 = json.loads(json1)
    R2 = json.loads(json2)

    all_objs = set()
    for R in (R1, R2):
        for b in R:
            if not isinstance(b, list):
                b = [b]
            all_objs |= set(b)
    n = max(all_objs)

    Y1 = build_relation_matrix(R1, n)
    Y2 = build_relation_matrix(R2, n)

    kernel = []
    for i in range(n):
        for j in range(i + 1, n):
            if (Y1[i, j] * Y2[i, j] == 0) and (Y1[j, i] * Y2[j, i] == 0):
                kernel.append([i + 1, j + 1])

    C = Y1 * Y2
    for i, j in kernel:
        C[i - 1, j - 1] = 1
        C[j - 1, i - 1] = 1

    E = C * C.T
    E_star = warshall(E)
    clusters = extract_clusters(E_star)

    k = len(clusters)
    G = np.zeros((k, k), dtype=int)

    for i in range(k):
        for j in range(k):
            if i != j:
                a = clusters[i][0] - 1
                b = clusters[j][0] - 1
                if C[a, b]:
                    G[i, j] = 1

    used = [False] * k
    order = []

    def dfs(v):
        used[v] = True
        for u in range(k):
            if G[v, u] and not used[u]:
                dfs(u)
        order.append(v)

    for i in range(k):
        if not used[i]:
            dfs(i)
    order.reverse()

    result = []
    for i in order:
        c = clusters[i]
        result.append(c[0] if len(c) == 1 else c)

    return json.dumps({
        "kernel": kernel,
        "consistent_ranking": result
    }, ensure_ascii=False)


# ----------------- ЗАПУСК -----------------

if __name__ == "__main__":
    print("=== Согласование кластерных ранжировок ===")

    with open("a_range.json", encoding="utf-8") as f:
        a = f.read()
    with open("b_range.json", encoding="utf-8") as f:
        b = f.read()
    with open("c_range.json", encoding="utf-8") as f:
        c = f.read()

    print("\nA vs B")
    print(main(a, b))

    print("\nA vs C")
    print(main(a, c))

    print("\nB vs C")
    print(main(b, c))

