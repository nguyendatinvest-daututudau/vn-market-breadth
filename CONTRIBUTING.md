# Contributing — Quy trình để không bao giờ revert mất code

## Vì sao từng bị revert
- 1 commit đụng 19 file + `git pull --rebase --autostash` auto-merge đã xóa nhầm `buildUdVolumePanel` và mang lại `nav-cta` duplicate.
- Không có test frontend, CI vẫn pass dù UI trắng.
- `docs/index.html` 2600 dòng, mọi feature chen vào cùng chỗ `render()` nên rebase nào cũng conflict.

## Quy tắc mới

1. **Một commit = một ý.** Không `git add data/ docs/` bulk. `update.yml` chỉ add data breadth, `evaluate.yml` chỉ add evaluation/calibration.
2. **Không rebase trên `main` chia sẻ.** Dùng `git config pull.rebase false` (đã set local). Khi `push` báo `fetch first`, chạy `git fetch && git merge origin/main` rồi giải conflict thủ công, không `rm -fr .git/rebase-merge`.
3. **Mọi thay đổi UI phải pass `check_frontend.py`.** Script kiểm tra UD Volume, chart `pts`, `nav-cta`, `data-sym`, Zweig. Chạy local trước commit, CI cũng chạy.
4. **Workflow chặn revert:** cả 2 workflow đều `python scripts/check_frontend.py` trước `pytest`. Nếu revert, CI fail ngay.
5. **Branch & PR:** Với feature mới, tạo branch riêng, push, mở PR, đợi CI xanh mới merge. Không push thẳng `main` với bulk change.

## Chạy local trước mỗi commit

```bash
python scripts/check_frontend.py
python -m pytest -q
python -m compileall -q scripts
```

Nếu thiếu, cài hook:

```bash
cp .githooks/pre-commit .git/hooks/pre-commit
```

## Khi gặp conflict

```bash
git fetch
git merge origin/main
# giải conflict từng file, nhớ kiểm tra UD Volume và nav
python scripts/check_frontend.py
git add <file-cu-the>
git commit
```
