# 最長片道きっぷの旅（Python・まず動く版）

入力: `始点ID, 終点ID, 距離` を1行ずつ（前後空白OK／CRLF可）  
出力: 最長経路の頂点IDを1行ずつ（**CRLF区切り**）

- 既定は **無向グラフ**（鉄道想定）
- アルゴリズム: DFSで単純路の最長経路を探索（小〜中規模向け）
- 依存: なし（標準ライブラリのみ）

## 実行例
```bash
python3 src/longest_ticket.py < tests/sample.in > out.txt
