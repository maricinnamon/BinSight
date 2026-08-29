# Release notes

## Template — "What's New" (4000 char limit, both languages)

Keep it to what changed for the person using the app. "Refactored the detection
decoder" is not a release note.

### English

```
<one line on the main change>

• <change>
• <change>

Fixes
• <fix>
```

### Українська

```
<один рядок про головну зміну>

• <зміна>
• <зміна>

Виправлення
• <виправлення>
```

---

## 1.0 — first release (draft)

### English

```
The first release of BinSight.

Point your camera at paper, plastic or metal and BinSight recognises it on the
spot — no account, no internet, nothing uploaded.

• Live recognition of paper, plastic and metal
• A box around what it finds, with its confidence shown as a percentage
• Brief handling advice for each material
• Works entirely offline; camera frames are never stored or sent
• English and Ukrainian, switchable in the app
```

### Українська

```
Перший випуск BinSight.

Наведіть камеру на папір, пластик або метал — BinSight розпізнає матеріал
одразу. Без акаунта, без інтернету, без завантажень.

• Живе розпізнавання паперу, пластику й металу
• Рамка навколо знайденого та впевненість у відсотках
• Коротка порада щодо кожного матеріалу
• Працює повністю офлайн; кадри не зберігаються й не надсилаються
• Англійська та українська з перемиканням у застосунку
```

---

## What to record internally for every release

Not for the App Store — for the repository, so a regression can be traced.

| Field | Where it comes from |
|---|---|
| Marketing version / build | `MARKETING_VERSION` / `CURRENT_PROJECT_VERSION` |
| Model SHA-256 | `models/pytorch/BinSightYOLO26n.pt` |
| CoreML package hash | `models/coreml/export_summary.json` → `tree_sha256` |
| Held-out test metrics | `reports/yolo26n_training_report.md` |
| Parity result | `models/coreml/parity_report.json` |

Current release model: `f6ca723341610b4d5f19856fcecbb30528193e2f7bed47c5d21deda539276c96`
— held-out test mAP50 0.6740, mAP50-95 0.4976, precision 0.7533, recall 0.5744.
