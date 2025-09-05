#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import sys
import argparse
import time
import heapq
from collections import deque, defaultdict

# ---------------------------
# 入力パース
# ---------------------------

def parse_edges(stdin):
    """
    標準入力から 'u, v, w' を読む。空行やフォーマット不正はスキップ。
    自己ループは無視。戻り値: [(u,v,w), ...], skipped_lines
    """
    edges = []
    skipped = 0
    for raw in stdin:
        line = raw.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            skipped += 1
            continue
        try:
            u = int(parts[0]); v = int(parts[1]); w = float(parts[2])
        except ValueError:
            skipped += 1
            continue
        if u == v:
            # 自己ループは単純路に使えないのでスキップ
            continue
        edges.append((u, v, w))
    return edges, skipped

# ---------------------------
# グラフ構築
# ---------------------------

def build_graph(edges, directed=False):
    """
    隣接辞書 graph[u][v] = w を作成。
    平行辺は重い方を採用。directed=False なら無向として対称に追加。
    """
    g = defaultdict(dict)
    for u, v, w in edges:
        if w > g[u].get(v, float("-inf")):
            g[u][v] = w
        if not directed:
            if w > g[v].get(u, float("-inf")):
                g[v][u] = w
    return g

def vertices_of(graph):
    vs = set(graph.keys())
    for u, nbrs in graph.items():
        vs |= set(nbrs.keys())
    return vs

# ---------------------------
# 連結成分（弱連結）
# ---------------------------

def weakly_connected_components(graph):
    undirected = defaultdict(set)
    for u, nbrs in graph.items():
        for v in nbrs:
            undirected[u].add(v)
            undirected[v].add(u)

    seen = set()
    comps = []
    for s in vertices_of(graph):
        if s in seen:
            continue
        stack = [s]
        seen.add(s)
        comp = set()
        while stack:
            x = stack.pop()
            comp.add(x)
            for y in undirected[x]:
                if y not in seen:
                    seen.add(y)
                    stack.append(y)
        comps.append(comp)
    return comps

# ---------------------------
# 無向木の判定
# ---------------------------

def is_tree_component(graph, comp):
    """
    comp が無向の木か判定（辺数 = 頂点数 - 1）。
    """
    edge_count = 0
    for u in comp:
        for v in graph.get(u, {}):
            if v in comp and u < v:
                edge_count += 1
    return edge_count == len(comp) - 1

# ---------------------------
# Dijkstra（正の重み）
# ---------------------------

def dijkstra(graph, start, allowed):
    dist = {v: float("inf") for v in allowed}
    parent = {v: None for v in allowed}
    dist[start] = 0.0
    pq = [(0.0, start)]
    while pq:
        d, u = heapq.heappop(pq)
        if d != dist[u]:
            continue
        for v, w in graph.get(u, {}).items():
            if v not in allowed:
                continue
            nd = d + w
            if nd < dist[v]:
                dist[v] = nd
                parent[v] = u
                heapq.heappush(pq, (nd, v))
    return dist, parent

# ---------------------------
# 無向木の直径（重み付き）
# ---------------------------

def tree_diameter(graph, comp):
    # 任意のAから全頂点への距離 → 最遠B
    A = next(iter(comp))
    distA, _ = dijkstra(graph, A, comp)
    B = max(distA, key=distA.get)
    # Bから全頂点への距離 → 最遠C が直径
    distB, parent = dijkstra(graph, B, comp)
    C = max(distB, key=distB.get)
    path = restore_path(parent, C)
    best_len = distB[C]
    return path, best_len

# ---------------------------
# DAG 判定 & 最長路
# ---------------------------

def is_dag_component(graph, comp):
    indeg = {v: 0 for v in comp}
    for u in comp:
        for v in graph.get(u, {}):
            if v in comp:
                indeg[v] += 1
    q = deque([v for v in comp if indeg[v] == 0])
    seen = 0
    while q:
        u = q.popleft()
        seen += 1
        for v in graph.get(u, {}):
            if v in comp:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)
    return seen == len(comp)

def dag_longest_path(graph, comp):
    # トポロジカル順（Kahn）
    indeg = {v: 0 for v in comp}
    for u in comp:
        for v in graph.get(u, {}):
            if v in comp:
                indeg[v] += 1
    q = deque([v for v in comp if indeg[v] == 0])
    topo = []
    while q:
        u = q.popleft()
        topo.append(u)
        for v in graph.get(u, {}):
            if v in comp:
                indeg[v] -= 1
                if indeg[v] == 0:
                    q.append(v)

    dp_len = {v: 0.0 for v in comp}
    parent = {v: None for v in comp}
    for u in topo:
        for v, w in graph.get(u, {}).items():
            if v not in comp:
                continue
            cand = dp_len[u] + w
            if cand > dp_len[v]:
                dp_len[v] = cand
                parent[v] = u
    end = max(comp, key=lambda x: dp_len[x])
    path = restore_path(parent, end)
    return path, dp_len[end]

# ---------------------------
# 部分集合DP（厳密） n <= limit
# ---------------------------

def subset_dp_longest_path(graph, comp):
    idx = {v:i for i, v in enumerate(comp)}
    rev = {i:v for v, i in idx.items()}
    n = len(comp)
    size = 1 << n

    dp = [ [float("-inf")] * n for _ in range(size) ]
    parent = [ [None] * n for _ in range(size) ]

    for v in comp:
        i = idx[v]
        dp[1<<i][i] = 0.0
        parent[1<<i][i] = None

    for mask in range(size):
        for i in range(n):
            if dp[mask][i] == float("-inf"):
                continue
            u = rev[i]
            for v, w in graph.get(u, {}).items():
                if v not in idx:
                    continue
                j = idx[v]
                if mask & (1<<j):
                    continue
                nmask = mask | (1<<j)
                cand = dp[mask][i] + w
                if cand > dp[nmask][j]:
                    dp[nmask][j] = cand
                    parent[nmask][j] = i

    best_len = float("-inf")
    best_mask = 0
    best_j = None
    for mask in range(size):
        for j in range(n):
            if dp[mask][j] > best_len:
                best_len = dp[mask][j]
                best_mask = mask
                best_j = j

    path_idx = []
    cur_mask, cur_j = best_mask, best_j
    while cur_j is not None:
        path_idx.append(cur_j)
        pj = parent[cur_mask][cur_j]
        if pj is None:
            break
        cur_mask ^= (1<<cur_j)
        cur_j = pj
    path_idx.reverse()
    path = [rev[i] for i in path_idx]
    return path, (best_len if best_len != float("-inf") else 0.0)

# ---------------------------
# 分枝限定 DFS（大規模用）
# ---------------------------

def branch_and_bound_longest_simple_path(graph, comp, time_limit=None):
    start_time = time.perf_counter()
    nodes = list(comp)

    sorted_neighbors = {
        u: sorted(((v, w) for v, w in graph.get(u, {}).items() if v in comp),
                  key=lambda x: -x[1])
        for u in comp
    }
    max_incident = {u: (max((w for _, w in sorted_neighbors[u]), default=0.0)) for u in comp}

    best_path = []
    best_len = 0.0

    def time_up():
        return time_limit is not None and (time.perf_counter() - start_time) >= time_limit

    def upper_bound(current_len, visited_count):
        remaining = len(comp) - visited_count
        max_w = max(max_incident.values()) if max_incident else 0.0
        return current_len + remaining * max_w

    def dfs(u, visited, path, cur_len):
        nonlocal best_path, best_len
        if cur_len > best_len:
            best_len = cur_len
            best_path = path[:]
        if time_up():
            return
        if upper_bound(cur_len, len(visited)) <= best_len:
            return
        for v, w in sorted_neighbors.get(u, []):
            if v in visited:
                continue
            visited.add(v)
            path.append(v)
            dfs(v, visited, path, cur_len + w)
            path.pop()
            visited.remove(v)

    start_order = sorted(nodes, key=lambda x: -max_incident.get(x, 0.0))
    for s in start_order:
        if time_up():
            break
        dfs(s, {s}, [s], 0.0)

    return best_path, best_len

# ---------------------------
# ユーティリティ
# ---------------------------

def restore_path(parent, end):
    path = []
    cur = end
    while cur is not None:
        path.append(cur)
        cur = parent[cur]
    path.reverse()
    return path

def solve_component(graph, comp, directed, exact_limit, time_limit):
    n = len(comp)
    if n == 1:
        v = next(iter(comp))
        return [v], 0.0

    if not directed:
        if is_tree_component(graph, comp):
            return tree_diameter(graph, comp)

    if directed and is_dag_component(graph, comp):
        return dag_longest_path(graph, comp)

    if n <= exact_limit:
        return subset_dp_longest_path(graph, comp)
    else:
        return branch_and_bound_longest_simple_path(graph, comp, time_limit=time_limit)

# ---------------------------
# メイン
# ---------------------------

def main():
    ap = argparse.ArgumentParser(description="最長片道きっぷ：スケール対応版（順序安定化つき）")
    ap.add_argument("--directed", action="store_true", help="入力を有向グラフとして扱う（既定: 無向）")
    ap.add_argument("--exact-limit", type=int, default=20, help="部分集合DPに切替える最大頂点数（既定: 20）")
    ap.add_argument("--time-limit", type=float, default=None, help="大規模用の分枝限定DFSの全体打ち切り秒（指定なし=最後まで探索）")
    args = ap.parse_args()

    edges, skipped = parse_edges(sys.stdin)
    if not edges:
        return

    graph = build_graph(edges, directed=args.directed)
    if not graph:
        return

    comps = weakly_connected_components(graph)

    start_all = time.perf_counter()
    best_path = []
    best_len = float("-inf")

    for comp in comps:
        remaining = None
        if args.time_limit is not None:
            elapsed = time.perf_counter() - start_all
            remaining = max(0.0, args.time_limit - elapsed)
        path, length = solve_component(graph, comp, args.directed, args.exact_limit, remaining)
        if length > best_len:
            best_len = length
            best_path = path
        if args.time_limit is not None and (time.perf_counter() - start_all) >= args.time_limit:
            break

    # ★ 無向グラフでは逆順と比べて辞書順が小さい方を採用（順序安定化）
    if not args.directed and best_path and tuple(best_path) > tuple(reversed(best_path)):
        best_path = list(reversed(best_path))

    for v in best_path:
        sys.stdout.write(str(v) + "\r\n")
    sys.stdout.flush()

    if skipped:
        # 採点を邪魔しないように stderr に情報表示
        sys.stderr.write(f"[info] skipped malformed lines: {skipped}\n")
        sys.stderr.flush()

if __name__ == "__main__":
    main()
