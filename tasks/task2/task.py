import os
import numpy as np
import math
from typing import Tuple, List

def calculate_graph_entropy(matrices: List[np.ndarray]) -> Tuple[float, float]:
   
    n = matrices[0].shape[0]
    m_count = len(matrices)
    entropy_sum = 0.0

    for matrix in matrices:
        for i in range(n):
            for j in range(n):
                if i != j:
                    p_ij = matrix[i, j] / (n - 1)
                    if p_ij > 0:
                        entropy_sum += p_ij * math.log2(p_ij)

    total_entropy = -entropy_sum
    max_entropy = (1 / math.e) * n * m_count
    normalized_entropy = total_entropy / max_entropy if max_entropy > 0 else 0

    return total_entropy, normalized_entropy


def build_edge_permutations(edges: List[Tuple[str, str]], vertices: List[str]) -> List[List[Tuple[str, str]]]:

    n = len(vertices)
    all_possible_edges = [
        (vertices[i], vertices[j])
        for i in range(n)
        for j in range(n)
        if i != j
    ]

    existing = set(edges)
    candidate_edges = [edge for edge in all_possible_edges if edge not in existing]

    all_permutations = []
    for idx_to_replace in range(len(edges)):
        for new_edge in candidate_edges:
            new_edges = edges.copy()
            new_edges[idx_to_replace] = new_edge
            all_permutations.append(new_edges)

    return all_permutations


def analyze_graph_structure(input_text: str, root_vertex: str) -> Tuple[float, float]:

    lines = input_text.strip().split('\n')
    edges = []
    vertices = set()

    for line in lines:
        if line.strip():
            v1, v2 = map(str.strip, line.split(','))
            vertices.update([v1, v2])
            edges.append((v1, v2))

    ordered_vertices = [root_vertex] + sorted(v for v in vertices if v != root_vertex)
    n = len(ordered_vertices)
    vertex_index = {v: i for i, v in enumerate(ordered_vertices)}

    edge_variants = build_edge_permutations(edges, ordered_vertices)

    best_entropy = -float('inf')
    best_normalized = 0.0
    best_edges = None

    for perm_edges in edge_variants:
        adjacency = np.zeros((n, n), dtype=bool)
        for v1, v2 in perm_edges:
            adjacency[vertex_index[v1], vertex_index[v2]] = True

        m_direct = adjacency.astype(int)
        m_reverse = m_direct.T

        transitive = adjacency.copy()
        for _ in range(1, n):
            transitive |= (transitive @ adjacency)

        m_trans_only = (transitive & ~adjacency).astype(int)
        m_trans_rev = m_trans_only.T
        m_reverse_bool = m_reverse.astype(bool)
        m_common = np.zeros((n, n), dtype=int)
        for i in range(n):
            for j in range(i + 1, n):
                if np.any(m_reverse_bool[i] & m_reverse_bool[j]):
                    m_common[i, j] = m_common[j, i] = 1

        matrices = [m_direct, m_reverse, m_trans_only, m_trans_rev, m_common]

        entropy_val, norm_val = calculate_graph_entropy(matrices)

        if entropy_val > best_entropy:
            best_entropy = entropy_val
            best_normalized = norm_val
            best_edges = perm_edges.copy()

    # --- Результат ---
    if best_edges:
        print("\n Наилучшая перестановка рёбер:")
        print(f"Исходные рёбра: {edges}")
        print(f"Оптимальные рёбра: {best_edges}")

    return best_entropy, best_normalized


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(current_dir, "task2.csv")

    with open(csv_path, "r") as file:
        graph_data = file.read()

    root = input("Введите значение корневой вершины: ").strip()
    H, h = analyze_graph_structure(graph_data, root)

    print("\n Результаты анализа:")
    print(f"Полная энтропия = {H:.4f}")
    print(f"Нормированная энтропия = {h:.4f}")
