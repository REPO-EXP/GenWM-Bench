
import csv, sys, os
from collections import defaultdict

if len(sys.argv) < 2:
    csv_path = 'outputs/attack_wam/all_attacks.csv'
else:
    csv_path = sys.argv[1]

if not os.path.exists(csv_path):
    print("No CSV found at", csv_path)
    sys.exit(0)

rows = []
methods = set()
with open(csv_path) as f:
    for r in csv.DictReader(f):
        rows.append(r)
        methods.add(r['Method'])

by = defaultdict(lambda: defaultdict(list))
all_ba = defaultdict(list)
all_psnr = defaultdict(list)

seen = set()
for r in rows:
    key = (r['Method'], r['Attack'])
    if key in seen: continue  
    seen.add(key)
    try:
        ba = float(r['bit_acc'])
        by[r['Attack']][r['Method']].append(ba)
        all_ba[r['Method']].append(ba)
    except:
        pass
    try:
        all_psnr[r['Method']].append(float(r['PSNR_atk']))
    except:
        pass

ml = sorted(methods)
W = 12

lines = []
hdr = f"{'Attack':<26}"
for m in ml:
    hdr += f" {m:>{W}}"
lines.append(hdr)
lines.append("=" * len(hdr))

for a in sorted(by.keys()):
    line = f" {a:<25}"
    for m in ml:
        v = by[a].get(m, [])
        if v:
            line += f" {sum(v) / len(v):>{W}.4f}"
        else:
            line += f" {'N/A':>{W}}"
    lines.append(line)

lines.append("-" * len(hdr))
avg_line = f" {'bit_acc (avg)':<25}"
for m in ml:
    v = all_ba[m]
    if v:
        avg_line += f" {sum(v) / len(v):>{W}.4f}"
    else:
        avg_line += f" {'N/A':>{W}}"
lines.append(avg_line)

psnr_line = f" {'PSNR_atk (avg)':<25}"
for m in ml:
    v = all_psnr[m]
    if v:
        psnr_line += f" {sum(v) / len(v):>{W}.4f}"
    else:
        psnr_line += f" {'N/A':>{W}}"
lines.append(psnr_line)

for l in lines:
    print(l)

txt_path = csv_path.replace('.csv', '.txt')
with open(txt_path, 'w') as f:
    for l in lines:
        f.write(l + '\n')
print(f"\nSaved: {txt_path}")
