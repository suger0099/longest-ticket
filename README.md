# 最長片道きっぷの旅（Python）

このプログラムは「最長片道きっぷの旅」問題を解きます。  
入力（駅間の路線）から、最も長い片道切符で行ける経路を求めて出力します。  
小規模は正確に、大規模は近似で結果を返すよう工夫しています。

---

## 実行方法

| OS      | コマンド例 |
|---------|------------|
| Windows | `py -3 src\longest_ticket.py < tests\sample.in` |
| macOS   | `python3 src/longest_ticket.py < tests/sample.in` |
| Linux   | `python3 src/longest_ticket.py < tests/sample.in` |

---

## 用意したテストケース

- **sample.in** : 問題文の例  
- **tree.in** : 無向木（直径アルゴリズム）  
- **dag.in** : 有向DAG（トポロジカルDP）  
- **cycle.in** : 閉路ありグラフ（DFS/分枝限定）  
- **isolated.in** : 孤立頂点を含むケース  

---

## 入力仕様

- 入力: `始点ID, 終点ID, 距離` を1行ずつ（前後空白OK／CRLF可）  
- 出力: 最長経路の頂点IDを1行ずつ（**CRLF区切り**）

### 特徴
- 既定は **無向グラフ**（鉄道想定）
- **アルゴリズム自動切替**
  - 無向・木（閉路なし） → 直径（Dijkstra×2）で厳密
  - 有向DAG → トポロジカル順の最長路DPで厳密
  - 一般グラフ
    - 小規模（≤20頂点 既定） → 部分集合DPで厳密
    - 大規模 → 分枝限定DFS（重い辺優先＋上界）＋ `--time-limit` で実用
- 依存: なし（標準ライブラリのみ）
- 入力の揺れに強い: 前後空白・空行OK、壊れた行はスキップ、自己ループは無視、平行辺は**重い方**を採用

---

## 出力例
```
入力ファイル `tests/sample.in`:

1, 2, 8.54
2, 3, 3.11
3, 1, 2.19
3, 4, 4
4, 1, 1.4

実行コマンド (Windows):

```powershell
py -3 src\longest_ticket.py < tests\sample.in

出力:

1
2
3
4

