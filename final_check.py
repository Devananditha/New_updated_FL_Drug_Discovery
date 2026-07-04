content = open('decentralized_paper.tex', encoding='utf-8').read()
checks = [
    ('0.708', 'Local-Only F1 real number'),
    ('0.667', 'DCA-FL F1 real number'),
    ('0.684', 'Fault-tolerant F1 real number'),
    ('0.768', 'AUC real number'),
    ('102.53', 'Comm overhead MB'),
    ('153.80', 'Centralized comm MB'),
    ('3{,}932', 'Global vocab size'),
    ('6,230', 'Edges per partition'),
    ('li2020fedprox', 'FedProx reference'),
    ('tab:comm', 'Comm overhead table'),
    ('tab:fault', 'Fault ablation table'),
    ('client drift', 'Client drift explanation'),
    ('33\\%', '33% communication saving'),
    ('0.000\\pm0.000', 'Centralized crash result'),
]
singular_checks = [
    ('Byzantine and Sybil', 1, 'Byzantine limitation (once)'),
    ('Clock Skew', 1, 'Clock skew (once)'),
]
print(f'Lines: {content.count(chr(10))} | Bibitems: {content.count(chr(92)+"bibitem{")}')
print()
all_ok = True
for term, label in checks:
    ok = term in content
    status = 'PASS' if ok else 'FAIL (missing!)'
    if not ok:
        all_ok = False
    print(f'  [{"OK" if ok else "!!"}] {status}: {label}')
for term, expected, label in singular_checks:
    count = content.count(term)
    ok = count == expected
    status = f'PASS (count={count})' if ok else f'FAIL: count={count}, expected {expected}'
    if not ok:
        all_ok = False
    print(f'  [{"OK" if ok else "!!"}] {status}: {label}')
print()
print('ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED')
