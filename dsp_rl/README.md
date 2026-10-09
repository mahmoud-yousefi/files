# بازتولید Chen et al. (2024) — DSP با MPNN + DDQN

## ترتیب اجرا
```bash
pip install -r requirements.txt
python sanity_check.py                                   # ۱) بدون torch؛ باید OK بدهد
python train.py --n 20 --episodes 2000 --eval_every 200 --out runs/smoke   # ۲) smoke test، چند دقیقه
python train.py --n 20 --episodes 200000 --seed 0 --out runs/er20_s0       # ۳) آموزش اصلی (paper: n=20,40 → 200k اپیزود؛ n=80,100 → 800k)
python evaluate.py grid_opt --model runs/er20_s0/model.pt --out results/table3
python evaluate.py er_ba    --model runs/er20_s0/model.pt --out results/table1
python evaluate.py lattice  --model runs/er20_s0/model.pt --out results/table2
```
GPU لازم نیست؛ گراف‌ها کوچکند. `--device cuda` اگر خواستی.
مقاله بدون seed چندگانه گزارش کرده؛ برای پایان‌نامه حداقل ۵ seed (`--seed 0..4`) اجرا کن.

## فصل ۵ — TDSP (مجموعه‌ی احاطه‌کننده‌ی تام)
```bash
python experiments/tdsp/exp1_construction.py --model runs/er20_s0/model.pt \
     --nmax 20,40,100,200 --reps 5 --ilp_time 60 --out results/tdsp/exp1
```
اثباتِ سازنده در `tdsp.py`؛ `|S|≤2|D|` با `two_stage_tdsp` assert می‌شود. نتایج و منابع (opt/module/greedy) در `results/tdsp/exp1/construction.{md,csv}`. جزئیات در `DECISIONS.md` (جدول T). اجرای آزمایشی کامل روی 2026-10-10.

## چطور نتیجه را برای من بفرست
۱. اگر خطا گرفتی: متن کامل traceback را paste کن (اولین کار ما رفع باگ احتمالی است، چون بخش torch را من نتوانستم اجرا کنم).
۲. بعد از هر آزمایش این‌ها را به چت پیوست کن: `runs/<نام>/config.json`، `runs/<نام>/log.csv`، و `results/*/*.md` (یا `.csv`).
۳. هر روز در `results.md` خودت یک خط بنویس: تاریخ، دستور اجرا، مشاهده. نتایج را در Word ننویس؛ Word در انتها با هم ساخته می‌شود.
