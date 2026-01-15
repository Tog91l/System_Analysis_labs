import numpy as np

with open("tasks/task2.csv", "r", encoding="utf-8") as file:
    raw_data = file.read().split("\n")


def process_graph(data_lines: list[str]) -> None:
    connections = []
    nodes = set()

    for entry in data_lines:
        entry = entry.strip()
        if entry:
            src, dst = entry.split(',')
            u, v = int(src), int(dst)
            connections.append((u, v))
            nodes.update([u, v])

    nodes = sorted(nodes)
    size = len(nodes)
    index_map = {node: i for i, node in enumerate(nodes)}

    # Матрица прямых связей
    direct_rel = np.zeros((size, size), dtype=int)
    for u, v in connections:
        i, j = index_map[u], index_map[v]
        direct_rel[i, j] = 1

    # Матрица обратных связей
    reverse_rel = direct_rel.T

    # Матрица косвенных связей
    transitive_rel = direct_rel.copy()
    for k in range(size):
        for i in range(size):
            if transitive_rel[i, k]:
                transitive_rel[i] = np.logical_or(transitive_rel[i], transitive_rel[k]).astype(int)

    indirect_rel = transitive_rel - direct_rel
    np.fill_diagonal(indirect_rel, 0)
    indirect_rel = np.clip(indirect_rel, 0, 1)

    # Матрица обратных косвенных связей
    reverse_indirect_rel = indirect_rel.T

    # Матрица соподчинения
    peer_rel = np.zeros((size, size), dtype=int)
    for i in range(size):
        for j in range(size):
            if i != j and np.any(direct_rel[:, i] & direct_rel[:, j]):
                peer_rel[i, j] = 1

    def show_matrix(matrix: np.ndarray, caption: str) -> None:
        print(f"\n{caption}")
        print("   ", " ".join(f"{n:>2}" for n in nodes))
        for i, row in enumerate(matrix):
            print(f"{nodes[i]:>2}:", " ".join(f"{x:>2}" for x in row))

    print("Вершины графа:", nodes)
    show_matrix(direct_rel, "1) Прямое управление")
    show_matrix(reverse_rel, "2) Прямое подчинение")
    show_matrix(indirect_rel, "3) Опосредованное управление")
    show_matrix(reverse_indirect_rel, "4) Опосредованное подчинение")
    show_matrix(peer_rel, "5) Соподчинение")


process_graph(raw_data)
