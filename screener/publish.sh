#!/usr/bin/env bash
# スクリーナーの出力を screener-data ブランチへコミットする。
# main には置かない: main へのコミットは Render の再ビルドを起こし、
# 毎営業日のデータ更新でビルドを使い切ってしまうため。
#
# 使い方: screener/publish.sh "<コミットメッセージ>" <コミットするパス...>
set -euo pipefail

MSG="$1"; shift
BRANCH="${DATA_BRANCH:-screener-data}"
WT="$(mktemp -d)/data"

git config user.name "screener-bot"
git config user.email "bot@users.noreply.github.com"

# 出力をいったん退避（worktree 追加で作業ツリーが切り替わっても残るように）
STAGE="$(mktemp -d)"
for p in "$@"; do
  if [ -e "$p" ]; then
    mkdir -p "$STAGE/$(dirname "$p")"
    cp -r "$p" "$STAGE/$(dirname "$p")/"
  fi
done

git fetch origin "$BRANCH" --depth 1 2>/dev/null || true
if git rev-parse --verify "origin/$BRANCH" >/dev/null 2>&1; then
  git worktree add "$WT" "origin/$BRANCH" >/dev/null
  git -C "$WT" checkout -B "$BRANCH" "origin/$BRANCH" >/dev/null
else
  echo "$BRANCH が無いので作成します"
  git worktree add --detach "$WT" >/dev/null
  git -C "$WT" checkout --orphan "$BRANCH" >/dev/null
  git -C "$WT" rm -rf --cached . >/dev/null 2>&1 || true
  find "$WT" -mindepth 1 -maxdepth 1 ! -name '.git' -exec rm -rf {} +
fi

for p in "$@"; do
  if [ -e "$STAGE/$p" ]; then
    mkdir -p "$WT/$(dirname "$p")"
    cp -r "$STAGE/$p" "$WT/$(dirname "$p")/"
  fi
done

cd "$WT"
git add -A -f .
if git diff --cached --quiet; then
  echo "変更なし"
  exit 0
fi
git commit -q -m "$MSG"
for i in 1 2 3 4 5; do
  if git pull --rebase --autostash origin "$BRANCH" 2>/dev/null || true; then :; fi
  if git push -q origin "HEAD:$BRANCH"; then
    echo "$BRANCH へ push しました"
    exit 0
  fi
  echo "push に失敗。15秒待って再試行 ($i/5)"
  sleep 15
done
exit 1
