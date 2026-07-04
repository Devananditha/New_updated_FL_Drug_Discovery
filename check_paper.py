"""Quick verification of the final paper."""
import re

with open("decentralized_paper.tex", "r", encoding="utf-8") as f:
    content = f.read()

checks = [
    # Things that SHOULD be present
    ("Devananditha V", False, "Real author 1"),
    ("Shiven Patro", False, "Real author 2"),
    ("VIT-AP University", False, "Real institution"),
    ("vitap.ac.in", False, "Real emails"),
    ("MAR-FL", False, "MAR-FL related work"),
    ("Totoro", False, "Totoro related work"),
    ("P3P-Fed", False, "P3P-Fed related work"),
    ("MELLODDY", False, "MELLODDY biomedical FL"),
    ("FeatureCloud", False, "FeatureCloud biomedical FL"),
    ("marfl2025", False, "MAR-FL bibitem"),
    ("totoro2024", False, "Totoro bibitem"),
    ("p3pfed2025", False, "P3P-Fed bibitem"),
    ("heyndrickx2023", False, "MELLODDY bibitem"),
    ("matschinske2023", False, "FeatureCloud bibitem"),
    ("Moshpit All-Reduce", False, "DCA-FL vs MAR-FL comparison"),
    ("content-addressed", False, "Content-addressed routing"),
    ("without blockchain consensus overhead", False, "Fixed blockchain claim"),
    ("ultrametric", False, "XOR ultrametric correct"),
    ("at-most-once training execution", False, "CRDT idempotency correct"),
    ("NTP", False, "Clock skew addressed"),
    ("tab:results", False, "Results table skeleton"),
    # Things that SHOULD NOT be present (bad = True)
    ("sub-second consensus propagation", True, "Unsupported sub-second claim"),
    ("zero blockchain overhead", True, "Marketing phrasing"),
    ("1st Author Name", True, "Placeholder author"),
    ("Institution/University Name", True, "Placeholder institution"),
]

total_lines = content.count('\n')
total_refs = content.count('\\bibitem{')
print(f"Total lines: {total_lines}")
print(f"Total references: {total_refs}")
print()

all_pass = True
for term, is_bad, label in checks:
    present = term in content
    if is_bad:
        ok = not present
        status = "PASS (removed)" if ok else "FAIL (still present!)"
    else:
        ok = present
        status = "PASS" if ok else "FAIL (missing!)"
    if not ok:
        all_pass = False
    mark = "OK" if ok else "!!"
    print(f"  [{mark}] {status}: {label}")

print()
print("ALL CHECKS PASSED" if all_pass else "SOME CHECKS FAILED - review above")
