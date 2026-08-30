Dataset validation summary:
  Path data: ./data
  Expected features: 210
  Target per class: 300

Per-class counts:
   A: 875 (OK)
   B: 682 (OK)
   C: 1435 (OK)
   D: 655 (OK)
   E: 634 (OK)
   F: 775 (OK)
   G: 366 (OK)
   H: 45 (KURANG)
   I: 67 (KURANG)
   J: 21 (KURANG)
   K: 85 (KURANG)
   L: 146 (KURANG)
   M: 91 (KURANG)
   N: 28 (KURANG)
   O: 30 (KURANG)
   P: 38 (KURANG)
   Q: 36 (KURANG)
   R: 41 (KURANG)
   S: 31 (KURANG)
   T: 35 (KURANG)
   U: 43 (KURANG)
   V: 49 (KURANG)
   W: 43 (KURANG)
   X: 28 (KURANG)
   Y: 26 (KURANG)
   Z: 24 (KURANG)

Kelas di bawah TARGET_PER_CLASS: H, I, J, K, L, M, N, O, P, Q, R, S, T, U, V, W, X, Y, Z

Dataset quality:
  Target per class: 300
  Min/Max samples: 21 / 1435
  Imbalance ratio: 68.33x

Collection priority:
   J: tambah 279 sample (sekarang 21)
   Z: tambah 276 sample (sekarang 24)
   Y: tambah 274 sample (sekarang 26)
   X: tambah 272 sample (sekarang 28)
   N: tambah 272 sample (sekarang 28)
   O: tambah 270 sample (sekarang 30)
   S: tambah 269 sample (sekarang 31)
   T: tambah 265 sample (sekarang 35)
   Q: tambah 264 sample (sekarang 36)
   P: tambah 262 sample (sekarang 38)
   R: tambah 259 sample (sekarang 41)
   W: tambah 257 sample (sekarang 43)
   U: tambah 257 sample (sekarang 43)
   H: tambah 255 sample (sekarang 45)
   V: tambah 251 sample (sekarang 49)
   I: tambah 233 sample (sekarang 67)
   K: tambah 215 sample (sekarang 85)
   M: tambah 209 sample (sekarang 91)
   L: tambah 154 sample (sekarang 146)

================================================================================
(2) Holdout evaluation: refit cloned model (test_size=0.2, seed=42)
================================================================================
Accuracy: 0.9803
Macro avg  (P/R/F1): 0.9884 / 0.9853 / 0.9860
Weighted   (P/R/F1): 0.9810 / 0.9803 / 0.9803

Per-class (Precision / Recall / F1 / Support):
   A: 0.9943 / 1.0000 / 0.9972 / 175
   B: 0.8958 / 0.9485 / 0.9214 / 136
   C: 0.9749 / 0.9477 / 0.9611 / 287
   D: 1.0000 / 1.0000 / 1.0000 / 131
   E: 1.0000 / 1.0000 / 1.0000 / 127
   F: 1.0000 / 1.0000 / 1.0000 / 155
   G: 1.0000 / 1.0000 / 1.0000 / 73
   H: 1.0000 / 1.0000 / 1.0000 / 9
   I: 1.0000 / 1.0000 / 1.0000 / 13
   J: 1.0000 / 1.0000 / 1.0000 / 4
   K: 1.0000 / 1.0000 / 1.0000 / 17
   L: 1.0000 / 1.0000 / 1.0000 / 29
   M: 1.0000 / 0.9444 / 0.9714 / 18
   N: 1.0000 / 1.0000 / 1.0000 / 6
   O: 1.0000 / 1.0000 / 1.0000 / 6
   P: 1.0000 / 1.0000 / 1.0000 / 8
   Q: 1.0000 / 1.0000 / 1.0000 / 7
   R: 1.0000 / 1.0000 / 1.0000 / 8
   S: 1.0000 / 1.0000 / 1.0000 / 6
   T: 1.0000 / 1.0000 / 1.0000 / 7
   U: 1.0000 / 1.0000 / 1.0000 / 9
   V: 0.8333 / 1.0000 / 0.9091 / 10
   W: 1.0000 / 0.7778 / 0.8750 / 9
   X: 1.0000 / 1.0000 / 1.0000 / 6
   Y: 1.0000 / 1.0000 / 1.0000 / 5
   Z: 1.0000 / 1.0000 / 1.0000 / 5
Top confusion pairs (true -> predicted):
   C -> B : 15
   B -> C : 7
   W -> V : 2
   M -> A : 1

================================================================================
(3) Cross-validation (cv=5, seed=42) via cross_val_predict
================================================================================
Accuracy: 0.9782
Macro avg  (P/R/F1): 0.9816 / 0.9766 / 0.9787
Weighted   (P/R/F1): 0.9786 / 0.9782 / 0.9783

Per-class (Precision / Recall / F1 / Support):
   A: 0.9920 / 0.9977 / 0.9949 / 875
   B: 0.9004 / 0.9413 / 0.9204 / 682
   C: 0.9708 / 0.9484 / 0.9595 / 1435
   D: 1.0000 / 1.0000 / 1.0000 / 655
   E: 1.0000 / 0.9984 / 0.9992 / 634
   F: 1.0000 / 1.0000 / 1.0000 / 775
   G: 0.9918 / 0.9945 / 0.9932 / 366
   H: 1.0000 / 1.0000 / 1.0000 / 45
   I: 0.9853 / 1.0000 / 0.9926 / 67
   J: 1.0000 / 0.9524 / 0.9756 / 21
   K: 1.0000 / 1.0000 / 1.0000 / 85
   L: 1.0000 / 0.9863 / 0.9931 / 146
   M: 0.9674 / 0.9780 / 0.9727 / 91
   N: 1.0000 / 0.9643 / 0.9818 / 28
   O: 1.0000 / 1.0000 / 1.0000 / 30
   P: 1.0000 / 0.9474 / 0.9730 / 38
   Q: 0.9211 / 0.9722 / 0.9459 / 36
   R: 0.9744 / 0.9268 / 0.9500 / 41
   S: 1.0000 / 1.0000 / 1.0000 / 31
   T: 0.9722 / 1.0000 / 0.9859 / 35
   U: 0.9767 / 0.9767 / 0.9767 / 43
   V: 0.9074 / 1.0000 / 0.9515 / 49
   W: 1.0000 / 0.9535 / 0.9762 / 43
   X: 1.0000 / 0.8929 / 0.9434 / 28
   Y: 0.9615 / 0.9615 / 0.9615 / 26
   Z: 1.0000 / 1.0000 / 1.0000 / 24
Top confusion pairs (true -> predicted):
   C -> B : 71
   B -> C : 40
   W -> V : 2
   R -> Q : 2
   L -> V : 2
   G -> A : 2
   C -> A : 2
   A -> M : 2
   Y -> I : 1
   X -> U : 1
   X -> G : 1
   X -> C : 1
