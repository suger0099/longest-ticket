#!/usr/bin/env python3
import sys

# 無向グラフ（隣接辞書）: graph[u][v] = 重み
graph = {}

# 入力を読む（空行や前後空白OK、自己ループは無視、平行辺は重い方）
for line in sys.stdin:
    line = line.strip()
    if not line:
        continue
    parts = [p.strip() for p in line.split(",")]
    if len(parts) != 3:
        continue
    try:
        u = int(parts[0]); v = int(parts[1]); w = float(parts[2])
    except ValueError:
        continue
    if u == v:
        continue
    graph.setdefault(u, {})
    graph.setdefault(v, {})
    # 平行辺は重い方を採用
    if w > graph[u].get(v, float("-inf")):
        graph[u][v] = w
        graph[v][u] = w

best_path = []
best_dist = 0.0

def dfs(node, visited, path, dist):
    global best_path, best_dist
    if dist > best_dist:
        best_dist = dist
        best_path = path[:]
    # 重い辺優先で展開（少し速くなることがある）
    for nxt, w in sorted(graph.get(node, {}).items(), key=lambda x: -x[1]):
        if nxt not in visited:
            visited.add(nxt)
            path.append(nxt)
            dfs(nxt, visited, path, dist + w)
            path.pop()
            visited.remove(nxt)

if graph:
    for start in graph:
        dfs(start, {start}, [start], 0.0)

# 仕様に合わせて CRLF で出力
for v in best_path:
    sys.stdout.write(str(v) + "\r\n")
sys.stdout.flush()
