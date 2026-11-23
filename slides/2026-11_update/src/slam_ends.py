import json, re, sys
src = open(sys.argv[1]).read()
slam = json.loads(re.search(r'const SLAM = (\{.*?\});\n', src).group(1))
print('speeds', slam['speeds'], 'T end', slam['T'][-1], 'n', len(slam['T']))
for sa, d_sa in slam['series'].items():
    for step, d_st in d_sa.items():
        for sp, d in d_st.items():
            row = []
            for f in ['EKF', 'FEJ-EKF', 'EqF']:
                if f in d:
                    m = d[f]
                    row.append(f"{f} {m['nees_native'][-1]:.2f} anc{m['anchor_rmse'][-1]:.1f}cm yaw{m['nees_yaw'][-1]:.2f}")
            print(f"sa={sa} S{step} v={slam['speeds'][int(sp)]}: " + ' | '.join(row))
