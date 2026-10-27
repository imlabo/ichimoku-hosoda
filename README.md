# スクリーナーの出力

このブランチは `daily-screener` ワークフローが毎営業日 17:30 JST に更新します。
コードは `main` にあります。データを main に置くと Render の再ビルドが毎日走るため分けています。

- `screener/watchlist.json` — 監視リスト（アプリの /screener がこれを読みます）
- `screener/watchlist.md` — 同じ内容を人が読む用
- `screener/daily/` — 日別の検出結果
- `screener/shards/` — 市場ごとの中間結果
