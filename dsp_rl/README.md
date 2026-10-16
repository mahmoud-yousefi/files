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

```bash
python experiments/tdsp/exp2_action_space.py --model runs/er20_s0/model.pt --out results/tdsp/exp2
```
آزمایش ۲: آیا ماسک مقاله برای TDSP کار می‌کند؟ نتیجه: **۱۷۷/۱۷۷ مستقل، ۰/۱۷۷ احاطه‌کننده‌ی تام** ⇒ ماسکِ مقاله ذاتاً با TDSP ناسازگار است (T4). خروجی در `results/tdsp/exp2/action_space.{md,csv}`.

```bash
python experiments/tdsp/exp3_env.py --out results/tdsp/exp3
```
آزمایش ۳: فضای عمل بازطراحی‌شده در `tdsp_env.py` (`TDSPEnv`) — با همان رابط `reset/candidates/step`. انتخاب `v` فقط `N(v)` را راضی می‌کند نه خودِ `v`؛ پایان وقتی همه همسایه‌ی داخلِ `S` دارند. تأیید: ۱۱۸ مکانیک پاس، همه‌ی اجراها احاطه‌کننده‌ی تام، و `greedy_env` = `greedy_tds` در ۵۹/۵۹. خروجی در `results/tdsp/exp3/env.{md,csv}`.

```bash
python experiments/tdsp/interface_check.py       # تطبیق عامل/محیط: باید «ALL INTERFACE CHECKS PASSED» بدهد
python train.py --env tdsp --n 20 --episodes 200000 --seed 0 --out runs/tdsp_er20_s0   # آموزش RL روی TDSP
```
آزمایش ۳ب (تطبیق): `TDSPEnv` حالت **دوکاناله** `[vf, ins]` می‌دهد؛ عامل نامزدها را از `env.candidates()` می‌گیرد (نه `vf==0`)؛ و پاداش `size` (`r=-1`) بازده را `-|S|` می‌کند. برای TDSP در آموزش گراف‌های دارای رأس منزوی حذف می‌شوند. جزئیات در `DECISIONS.md` (T6).

```bash
python train.py --env tdsp --reward size  --n 20 --episodes 200000 --seed 0 --out runs/tdsp_er20_s0_size
python train.py --env tdsp --reward paper --n 20 --episodes 200000 --seed 0 --out runs/tdsp_er20_s0_paper
python experiments/tdsp/exp4_eval.py \
     --model size=runs/tdsp_er20_s0_size/model.pt --model paper=runs/tdsp_er20_s0_paper/model.pt \
     --out results/tdsp/exp4
```
آزمایش ۴ (RL مستقیم روی TDSP): روی لتیس‌ها مدل از greedy بهتر است (میانگین `|S|/greedy` ≈ ۰.۹۵، مثلاً HEX 196 → ۸۳ در برابر ۹۸)؛ ولی روی گراف‌های تصادفیِ بزرگ `n≥80` بدتر است (تعمیم‌نیافته، چون فقط روی `n=20` آموزش دیده). خروجی در `results/tdsp/exp4/eval.{md,csv}`.

## چطور نتیجه را برای من بفرست
۱. اگر خطا گرفتی: متن کامل traceback را paste کن (اولین کار ما رفع باگ احتمالی است، چون بخش torch را من نتوانستم اجرا کنم).
۲. بعد از هر آزمایش این‌ها را به چت پیوست کن: `runs/<نام>/config.json`، `runs/<نام>/log.csv`، و `results/*/*.md` (یا `.csv`).
۳. هر روز در `results.md` خودت یک خط بنویس: تاریخ، دستور اجرا، مشاهده. نتایج را در Word ننویس؛ Word در انتها با هم ساخته می‌شود.

```bash
python experiments/tdsp/exp6_baselines.py --model size=runs/tdsp_er20_s0_size_v2/model.pt --out results/tdsp/exp6
```
آزمایش ۶ (خط مبنای قوی): TDSP = مسئله‌ی set-cover، پس علاوه بر greedy، `remove_redundant` و **GRASP** (در `heuristics.py`) اضافه شدند. نتیجه‌ی صادقانه: مدل در برابر GRASP **بدتر** است (۱۳ بدتر، ۶ مساوی، ۱ بهتر از ۲۰)؛ مزیتِ مدل **سرعت** است (~۵ برابر سریع‌تر از GRASP). خروجی در `results/tdsp/exp6/baselines.{md,csv}`.

```bash
python experiments/tdsp/exp7_reduction.py --out results/tdsp/exp7
```
آزمایش ۷ (قواعد کاهش): سه قاعده (منزوی، مؤلفه‌ها، برگ) در `reductions.py`. نتیجه‌ی **منفی و صادقانه**: قاعده‌ی برگ برای TDSP **نامعتبر** است (مثال نقض `P3`)؛ قواعد ۱ و ۲ درست‌اند (۱۵۶ مقایسه با ILP، صفر خطا) ولی روی گراف‌های ما **هیچ‌کدام فعال نمی‌شوند** (همه تک‌مؤلفه‌ای، بدون رأس منزوی، درجه‌ی کمینه ≥۲). خروجی در `results/tdsp/exp7/reduction.{md,csv}`.

```bash
python experiments/tdsp/exp8_forcing.py phase0 --model size_v2=runs/tdsp_er20_s0_size_v2/model.pt --out results/tdsp/exp8
```
آزمایش ۸ (اجبارِ دانش ریاضی): توزیع آزمون عوض شد — خانواده‌های اسپارسِ پربرگی `tree` و `ba1` در `graphs.py`، جایی که مجموعه‌ی اجباری F (پشتیبان‌های برگ؛ در *هر* TDS عضو هستند) در ۱۰۰٪ نمونه‌ها غیرتهی است (~۳۰٪ رأس‌ها). **فاز ۰ (بدون آموزش):** مدل exp5 به‌صورت zero-shot روی اسپارس هم‌سطح بهینه/GRASP است؛ اجبارِ F کیفیت را حفظ می‌کند ولی تعداد تصمیم‌های RL را ۶۶–۸۸٪ کم می‌کند. خروجی: `results/tdsp/exp8/phase0_*.{md,csv}`. **فاز ۱ (دو بازوی آموزشی؛ A روی Colab/T4 و B روی مک/CPU — T9):** هر دو به `val|S|=9.50` رسیدند ولی B در **~۵۵هزار** اپیزود و A در **~۱۴۰هزار** (همگرایی ~۲.۵×)؛ روی TREE، `armA`/`armA+F`/`armB+F` همه **بهینه** (|S|/γ_t=1.00 تا n=100؛ n=200 برابر GRASP) و تصمیم‌های RL ~۶۴٪ کمتر (مثلاً 94.3→34.1). **هشدار:** سیاستِ خالیِ بازوی B بدون seed مختل است (تا 1.45× بدتر) — همیشه `armB+F` گزارش شود. آموزش اسپارس روی متراکم هزینه دارد (armA روی ER n=100 تا 1.48×γ_t) ولی armB روی لتیس‌ها عالی است (HEX 196 → 77 در برابر GRASP 83). خروجی: `results/tdsp/exp8/phase1_*.{md,csv,png}`؛ لاگ/کانفیگ بازوها در `runs/armA_sparse_base/` و `runs/armB_sparse_forced/`؛ نوت‌بوک: `colab_tdsp_exp8.ipynb`. جزئیات در `DECISIONS.md` (T8/T9).
