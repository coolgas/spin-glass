# notebook_stage_b_escalation — research result log

Recorded UTC: 2026-10-27T15:45:55.720874+00:00

**System: L=4, N=48 spins, periodic boundaries in both directions.**

State: **running_notebook_stage_b_escalation**. Completed results: **109/144**. Recorded failures: **0**.

Manifest signature: `0e910b5e5d6cfbe08460b7f5f45a37e0d34a8f928f8ae91958f730e37cccde7d`.

Process deadline UTC: 2026-10-27T17:46:47.218088+00:00.

All results are feasibility/development evidence. Ground-state convergence and phase-boundary inference remain unqualified. An ancestry alarm means any cycle has descendant ESS below 8 or largest-lineage fraction above 0.5. Every alarm is retained.

## Settings and completion

| Level | Gamma | h | Guide | W | Equilibration | Measurement | Forward | Cycles | Branch | Done / expected | Alarms | Median seconds | Minimum ESS |
|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline_h005 | 0.5 | 0.05 | constant | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 0.2447 | 1 |
| baseline_h005 | 0.5 | 0.05 | network | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 14.33 | 2.723 |
| baseline_h005 | 2.0 | 0.05 | constant | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 0.7594 | 1 |
| baseline_h005 | 2.0 | 0.05 | network | 64 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 0 | 108.9 | 14.63 |
| equilibration4 | 0.5 | 0.0 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 3.67 | 1 |
| equilibration4 | 0.5 | 0.0 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 0.5 | 0.05 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 3.57 | 1 |
| equilibration4 | 0.5 | 0.05 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.0 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 12.36 | 1.022 |
| equilibration4 | 2.0 | 0.0 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| equilibration4 | 2.0 | 0.05 | constant | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 12.83 | 1.002 |
| equilibration4 | 2.0 | 0.05 | network | 1024 | 4.0 | 1.0 | 1.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.0 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 4/4 | 4 | 4.418 | 1 |
| forward2 | 0.5 | 0.0 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 0.5 | 0.05 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 4/4 | 4 | 4.297 | 1 |
| forward2 | 0.5 | 0.05 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.0 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 4/4 | 4 | 15.53 | 1 |
| forward2 | 2.0 | 0.0 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| forward2 | 2.0 | 0.05 | constant | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 4/4 | 4 | 15.65 | 1 |
| forward2 | 2.0 | 0.05 | network | 1024 | 2.0 | 1.0 | 2.0 | 4 | 0.1 | 0/4 | 0 | pending | pending |
| population1024 | 0.5 | 0.0 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 2.844 | 1.004 |
| population1024 | 0.5 | 0.0 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 0 | 414.1 | 19.54 |
| population1024 | 0.5 | 0.05 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 2.859 | 1.002 |
| population1024 | 0.5 | 0.05 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 3/4 | 0 | 209.1 | 21.58 |
| population1024 | 2.0 | 0.0 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 9.9 | 1.046 |
| population1024 | 2.0 | 0.0 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 3/4 | 0 | 3256 | 315.6 |
| population1024 | 2.0 | 0.05 | constant | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 9.847 | 1 |
| population1024 | 2.0 | 0.05 | network | 1024 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 3/4 | 0 | 1638 | 278.1 |
| population256 | 0.5 | 0.0 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 0.7239 | 1 |
| population256 | 0.5 | 0.0 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 1 | 98.93 | 7.841 |
| population256 | 0.5 | 0.05 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 0.7194 | 1 |
| population256 | 0.5 | 0.05 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 2 | 49.13 | 1.617 |
| population256 | 2.0 | 0.0 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 2.339 | 1.032 |
| population256 | 2.0 | 0.0 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 0 | 801.8 | 75.16 |
| population256 | 2.0 | 0.05 | constant | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 4 | 2.333 | 1 |
| population256 | 2.0 | 0.05 | network | 256 | 2.0 | 1.0 | 1.0 | 4 | 0.1 | 4/4 | 0 | 406.3 | 63.02 |

## Individual population results

Values below are individual population estimates, without uncertainty bars. Forward cycles are correlated diagnostics, not independent repeats. Compare four independent repeats within each disorder before forming disorder averages. The escalation subset contains one disorder and cannot estimate disorder uncertainty.

| Task ID | Disorder | Repeat | Level | Gamma | h | Guide | E/N | q2 | chi_conn/N | Seconds | ESS minimum | Largest lineage | Alarm |
|---|---:|---:|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---|
| 874530b32050a5380530a248 | 60000 | 0 | baseline_h005 | 0.5 | 0.05 | network | -1.0511928 | 0.050516764 | 0.091966417 | 14.589 | 4.983 | 0.39062 | True |
| 15cf4c3f3464165b3069ccc2 | 60000 | 0 | baseline_h005 | 0.5 | 0.05 | constant | -0.92469431 | 0.054911296 | 0.0081854926 | 0.25125 | 1 | 1 | True |
| 7d5b433a9f8b223961401dbd | 60000 | 0 | baseline_h005 | 2.0 | 0.05 | network | -2.2086178 | 0.024264865 | 0.024637858 | 108.85 | 15.398 | 0.125 | False |
| ef059526f5605f1be19ae624 | 60000 | 0 | baseline_h005 | 2.0 | 0.05 | constant | -2.153406 | 0.015625 | 0.0085720486 | 0.75824 | 1 | 1 | True |
| 6f2077d6add4762b2a521526 | 60000 | 1 | baseline_h005 | 0.5 | 0.05 | network | -1.0476823 | 0.024447971 | 0.042643229 | 14.239 | 2.7234 | 0.59375 | True |
| bccefe1b7a2c8965976b9c65 | 60000 | 1 | baseline_h005 | 0.5 | 0.05 | constant | -0.87111715 | 0.0065171984 | 0.03640408 | 0.24492 | 1 | 1 | True |
| 005a15bb1b08beffb60137a8 | 60000 | 1 | baseline_h005 | 2.0 | 0.05 | network | -2.2101181 | 0.030266656 | 0.018866645 | 109.03 | 17.504 | 0.125 | False |
| 25d873693cdd94c816747ef7 | 60000 | 1 | baseline_h005 | 2.0 | 0.05 | constant | -2.1495051 | 0.018717448 | 0.025282118 | 0.76047 | 1 | 1 | True |
| 2214c408be7f9335dfdfd861 | 60000 | 2 | baseline_h005 | 0.5 | 0.05 | network | -1.0529991 | 0.11385769 | 0.095194499 | 14.414 | 6.8267 | 0.26562 | True |
| 7968a9c03ecd478f631f8ae5 | 60000 | 2 | baseline_h005 | 0.5 | 0.05 | constant | -0.86429879 | 0.0098605686 | 0.023939345 | 0.24453 | 1 | 1 | True |
| 607449d54d74f216114b6a49 | 60000 | 2 | baseline_h005 | 2.0 | 0.05 | network | -2.2109105 | 0.023003472 | 0.028862847 | 109.34 | 14.629 | 0.17188 | False |
| 6f2b10cb08493b6fa65ce398 | 60000 | 2 | baseline_h005 | 2.0 | 0.05 | constant | -2.192859 | 0.022949219 | 0.016384549 | 0.74054 | 1 | 1 | True |
| ef92cd8d46b31f32f5f932ae | 60000 | 3 | baseline_h005 | 0.5 | 0.05 | network | -1.0561919 | 0.083299425 | 0.012030707 | 14.041 | 4.8188 | 0.42188 | True |
| b4cc708585c98ee8d89d0e4f | 60000 | 3 | baseline_h005 | 0.5 | 0.05 | constant | -0.88847859 | 0.016059028 | 0.034288194 | 0.24306 | 1 | 1 | True |
| 96659c8296584122dcc084af | 60000 | 3 | baseline_h005 | 2.0 | 0.05 | network | -2.2099293 | 0.026048448 | 0.024291992 | 108.76 | 17.655 | 0.14062 | False |
| 5c717ccdd7adceeee2eba626 | 60000 | 3 | baseline_h005 | 2.0 | 0.05 | constant | -2.1679604 | 0.015509711 | 0.027621799 | 0.83727 | 1.0981 | 0.95312 | True |
| 7bf71d15cecdc97755b99810 | 60000 | 0 | population256 | 0.5 | 0.0 | network | -1.0596039 | 0.11931356 | 0.02214898 | 99.052 | 13.991 | 0.15234 | False |
| 672935baa300fc48e484b017 | 60000 | 0 | population256 | 0.5 | 0.0 | constant | -0.90663799 | 0.011413574 | 0.0081481934 | 0.73429 | 1.2636 | 0.88281 | True |
| 8385fef399a581bbf0eb90b0 | 60000 | 0 | population256 | 0.5 | 0.05 | network | -1.0605293 | 0.16023763 | 0.026969062 | 49.47 | 9.2853 | 0.30859 | False |
| 41ccd0073e3bfb4953e2a980 | 60000 | 0 | population256 | 0.5 | 0.05 | constant | -0.90640507 | 0.0363227 | 0.0078108046 | 0.72112 | 1.04 | 0.98047 | True |
| c33eb6a414424cdaaafc69c7 | 60000 | 0 | population256 | 2.0 | 0.0 | network | -2.2106465 | 0.026250203 | 0.022486369 | 801.16 | 75.156 | 0.035156 | False |
| 7fca41413a382cf272448078 | 60000 | 0 | population256 | 2.0 | 0.0 | constant | -2.1957737 | 0.0034162733 | 0.018171522 | 2.3387 | 1.048 | 0.97656 | True |
| e62d3945200d5cf86145392b | 60000 | 0 | population256 | 2.0 | 0.05 | network | -2.2106774 | 0.024427626 | 0.021782769 | 406.25 | 72.818 | 0.035156 | False |
| ad9a9bd93514fa078da52330 | 60000 | 0 | population256 | 2.0 | 0.05 | constant | -2.1808632 | 0.012464735 | 0.0090213352 | 2.3318 | 1 | 1 | True |
| 03e7a2c1f4b97acc664c7cb3 | 60000 | 1 | population256 | 0.5 | 0.0 | network | -1.0577978 | 0.11913045 | 0.35801019 | 98.815 | 9.9599 | 0.25391 | False |
| 3a91afa891d62dfea3d499a8 | 60000 | 1 | population256 | 0.5 | 0.0 | constant | -0.89604745 | 0.027301364 | 0.024056329 | 0.71175 | 1 | 1 | True |
| f75620b90d576845b80fdb76 | 60000 | 1 | population256 | 0.5 | 0.05 | network | -1.0558488 | 0.083650377 | 0.077226427 | 49.784 | 11.115 | 0.26562 | False |
| 445914faf9b8a6f6824939e8 | 60000 | 1 | population256 | 0.5 | 0.05 | constant | -0.86458435 | 0.0062527127 | 0.0085008409 | 0.71943 | 1 | 1 | True |
| f4f5060683b1971c75dc384c | 60000 | 1 | population256 | 2.0 | 0.0 | network | -2.210688 | 0.028464423 | 0.024675157 | 799.17 | 79.727 | 0.03125 | False |
| cff1a5c80911cf8094a2e645 | 60000 | 1 | population256 | 2.0 | 0.0 | constant | -2.2067853 | 0.040703668 | 0.011035495 | 2.3173 | 1.7495 | 0.73047 | True |
| 8a223cf1fbafcd79b421873e | 60000 | 1 | population256 | 2.0 | 0.05 | network | -2.2108839 | 0.025165134 | 0.025426229 | 406.1 | 67.285 | 0.042969 | False |
| 2ca89d745ae0f005c03d2055 | 60000 | 1 | population256 | 2.0 | 0.05 | constant | -2.1766621 | 0.011642456 | 0.06026374 | 2.3351 | 1 | 1 | True |
| 653f440aa231e21d53ab4787 | 60000 | 2 | population256 | 0.5 | 0.0 | network | -1.0610291 | 0.06191508 | 0.025073581 | 94.809 | 8.9408 | 0.29688 | False |
| dd059719d88b337ee391c326 | 60000 | 2 | population256 | 0.5 | 0.0 | constant | -0.90666852 | 0.0099826389 | 0.010235257 | 0.72097 | 1 | 1 | True |
| b8a4ef596f90a53b36806f71 | 60000 | 2 | population256 | 0.5 | 0.05 | network | -1.0630685 | 0.23186069 | 0.021674262 | 48.184 | 1.6166 | 0.78516 | True |
| 3dc6d3c25b8c95fba3d34626 | 60000 | 2 | population256 | 0.5 | 0.05 | constant | -0.93730275 | 0.027253893 | 0.021158854 | 0.71074 | 1 | 1 | True |
| f1e68959aa50e0474d353394 | 60000 | 2 | population256 | 2.0 | 0.0 | network | -2.2102688 | 0.023006863 | 0.025917901 | 805.81 | 84.672 | 0.035156 | False |
| efb6576cddcd458d7b89fcbd | 60000 | 2 | population256 | 2.0 | 0.0 | constant | -2.1774634 | 0.021930271 | 0.0080854628 | 2.3413 | 1.0318 | 0.98438 | True |
| a5d0682ce3dde291b175646b | 60000 | 2 | population256 | 2.0 | 0.05 | network | -2.2107667 | 0.0225118 | 0.024727715 | 411.84 | 66.065 | 0.050781 | False |
| acee10e95e5b62eb8e0ed8eb | 60000 | 2 | population256 | 2.0 | 0.05 | constant | -2.193928 | 0.026041667 | 0.026462131 | 2.3146 | 1 | 1 | True |
| 2d03a0a48c6a0cc175949ae6 | 60000 | 3 | population256 | 0.5 | 0.0 | network | -1.0562411 | 0.10560777 | 0.29248047 | 99.341 | 7.8411 | 0.32031 | True |
| b741e9b8ffb8d89f43afa337 | 60000 | 3 | population256 | 0.5 | 0.0 | constant | -0.92943612 | 0.030047947 | 0.055404663 | 0.72682 | 1 | 1 | True |
| 1f7114f5144e7e03911bbd7f | 60000 | 3 | population256 | 0.5 | 0.05 | network | -1.0569161 | 0.020207723 | 0.026526557 | 48.796 | 5.9611 | 0.29688 | True |
| 4f68d79d7c1a139ec9635df7 | 60000 | 3 | population256 | 0.5 | 0.05 | constant | -0.89260184 | 0.017305162 | 0.018339369 | 0.71947 | 1.0821 | 0.96094 | True |
| 34a9406d6725303634c74400 | 60000 | 3 | population256 | 2.0 | 0.0 | network | -2.2107567 | 0.025100708 | 0.027158949 | 802.36 | 78.58 | 0.042969 | False |
| 311f17f7ca60f76f48750c6b | 60000 | 3 | population256 | 2.0 | 0.0 | constant | -2.1947039 | 0.029937744 | 0.026986016 | 2.3391 | 1.1905 | 0.91406 | True |
| bf7bf0595159c331d4632dd7 | 60000 | 3 | population256 | 2.0 | 0.05 | network | -2.2104722 | 0.026421441 | 0.022354126 | 406.34 | 63.015 | 0.070312 | False |
| c2037b24e228d7af6941852d | 60000 | 3 | population256 | 2.0 | 0.05 | constant | -2.2100409 | 0.0073784722 | 0.010923598 | 2.3431 | 1 | 1 | True |
| 3ce35eb6e095d20f58a735a1 | 60000 | 0 | population1024 | 0.5 | 0.0 | network | -1.0560282 | 0.067187415 | 0.11127175 | 415.67 | 19.543 | 0.16504 | False |
| 8130d8f264c7f5acd23485e9 | 60000 | 0 | population1024 | 0.5 | 0.0 | constant | -0.94598225 | 0.005218082 | 0.010093265 | 2.8791 | 1.0297 | 0.98535 | True |
| f58e1a9e4ea6b236891fec99 | 60000 | 0 | population1024 | 0.5 | 0.05 | network | -1.0586342 | 0.11439641 | 0.029566023 | 216.3 | 21.578 | 0.17969 | False |
| e89debb89b58525aab510a17 | 60000 | 0 | population1024 | 0.5 | 0.05 | constant | -0.94720195 | 0.018461439 | 0.042315165 | 2.9108 | 1.002 | 0.99902 | True |
| 137ffaa59373a4a0d033d8c0 | 60000 | 0 | population1024 | 2.0 | 0.0 | network | -2.2102836 | 0.024118212 | 0.025363498 | 3257.2 | 315.65 | 0.010742 | False |
| 5be3901bd22539b8f3b0334e | 60000 | 0 | population1024 | 2.0 | 0.0 | constant | -2.1970733 | 0.015886095 | 0.027434455 | 10.038 | 1.1624 | 0.92676 | True |
| 2bbdd71228fb0af7712bcd94 | 60000 | 0 | population1024 | 2.0 | 0.05 | network | -2.2101205 | 0.025022719 | 0.024027083 | 1637.8 | 278.14 | 0.014648 | False |
| 51b718fc02e7d0fcb5db8728 | 60000 | 0 | population1024 | 2.0 | 0.05 | constant | -2.1882824 | 0.0051761203 | 0.029265934 | 9.9772 | 1.0524 | 0.97461 | True |
| b34488e04007cb7aa88f97ce | 60000 | 1 | population1024 | 0.5 | 0.0 | network | -1.0601333 | 0.11309094 | 0.038843791 | 415.39 | 25.04 | 0.16992 | False |
| 87507cf4063fcbe45c2f31cb | 60000 | 1 | population1024 | 0.5 | 0.0 | constant | -0.98987383 | 0.030247158 | 0.042798784 | 2.8086 | 1.1552 | 0.92969 | True |
| 8b509ec95a5420f2e7872998 | 60000 | 1 | population1024 | 0.5 | 0.05 | network | -1.0594327 | 0.19509718 | 0.040579902 | 202.58 | 21.72 | 0.1416 | False |
| 841b17d4b4333dcfbf36bd66 | 60000 | 1 | population1024 | 0.5 | 0.05 | constant | -0.9644782 | 0.018364376 | 0.044782427 | 2.941 | 1.9656 | 0.6709 | True |
| 672ddec8dc17ada2d5438bf9 | 60000 | 1 | population1024 | 2.0 | 0.0 | network | -2.2099298 | 0.023200989 | 0.023941464 | 3255.6 | 329.12 | 0.012695 | False |
| f5561e951c0242de0dcfe20d | 60000 | 1 | population1024 | 2.0 | 0.0 | constant | -2.201418 | 0.014545441 | 0.0090997484 | 10.56 | 1.1463 | 0.93262 | True |
| 5090e169a4c29abb369f3cf9 | 60000 | 1 | population1024 | 2.0 | 0.05 | network | -2.2107308 | 0.026169247 | 0.025192685 | 1635.1 | 289.02 | 0.015625 | False |
| 6495191c1dea3badd2311f4b | 60000 | 1 | population1024 | 2.0 | 0.05 | constant | -2.1926439 | 0.014188131 | 0.01221678 | 9.7173 | 1.0258 | 0.9873 | True |
| fb84e2d3f5acb5a06e3ed22a | 60000 | 2 | population1024 | 0.5 | 0.0 | network | -1.0595762 | 0.35397169 | 0.042819553 | 397.89 | 25.993 | 0.16992 | False |
| 7df98f248c65c8042e9b3484 | 60000 | 2 | population1024 | 0.5 | 0.0 | constant | -0.94783094 | 0.070195516 | 0.0090738932 | 2.7796 | 1.0039 | 0.99805 | True |
| 67c185634c9dd194f0cc66af | 60000 | 2 | population1024 | 0.5 | 0.05 | network | -1.0588609 | 0.07628674 | 0.017726474 | 209.13 | 45.162 | 0.089844 | False |
| 1dde286739d2645f721a93ba | 60000 | 2 | population1024 | 0.5 | 0.05 | constant | -0.9380257 | 0.0030640496 | 0.062965393 | 2.7847 | 1.0198 | 0.99023 | True |
| 29b3c5e4cd91f61c5df38d0b | 60000 | 2 | population1024 | 2.0 | 0.0 | network | -2.2104515 | 0.024167803 | 0.023926205 | 3251.5 | 318.91 | 0.011719 | False |
| 59efe007d2a07b47e786f2d7 | 60000 | 2 | population1024 | 2.0 | 0.0 | constant | -2.2085927 | 0.016222636 | 0.031776428 | 9.6396 | 1.0462 | 0.97754 | True |
| 5d0772c3be97ee47d1f5ff49 | 60000 | 2 | population1024 | 2.0 | 0.05 | network | -2.2104425 | 0.023121728 | 0.025429196 | 1638 | 296.04 | 0.011719 | False |
| 3a6b2feace9ed37ac81c5d37 | 60000 | 2 | population1024 | 2.0 | 0.05 | constant | -2.1965702 | 0.025280846 | 0.010047489 | 9.6684 | 1 | 1 | True |
| d0a12a9b920ce84276a328ed | 60000 | 3 | population1024 | 0.5 | 0.0 | network | -1.0614825 | 0.16933695 | 0.20837614 | 412.88 | 49.169 | 0.10547 | False |
| c3da3cf9c01d7a37f1577018 | 60000 | 3 | population1024 | 0.5 | 0.0 | constant | -0.96434713 | 0.033498552 | 0.02090666 | 2.9401 | 1.0943 | 0.95508 | True |
| 32bcda32f2e8ead9200194c1 | 60000 | 3 | population1024 | 0.5 | 0.05 | constant | -0.94952882 | 0.016573588 | 0.021030426 | 2.8074 | 1.0118 | 0.99414 | True |
| 37436fa0fd72309877e46af9 | 60000 | 3 | population1024 | 2.0 | 0.0 | constant | -2.2016166 | 0.017044491 | 0.037823147 | 9.7625 | 1.0629 | 0.96973 | True |
| 7d3dc35529c6e4e07df6922f | 60000 | 3 | population1024 | 2.0 | 0.05 | constant | -2.1904808 | 0.034086863 | 0.052427504 | 10.118 | 1.1542 | 0.92871 | True |
| 0e5da6744adc4384d2b7367c | 60000 | 0 | equilibration4 | 0.5 | 0.0 | constant | -0.96976555 | 0.03832372 | 0.022540198 | 3.5171 | 1.0157 | 0.99219 | True |
| 8226d2eea6eedb9b59a37c2f | 60000 | 0 | equilibration4 | 0.5 | 0.05 | constant | -0.96059927 | 0.051380581 | 0.011739519 | 3.5713 | 1 | 1 | True |
| 4a5d335a73861b601dd69c59 | 60000 | 0 | equilibration4 | 2.0 | 0.0 | constant | -2.2050104 | 0.017712911 | 0.0050150553 | 11.758 | 1.2081 | 0.9082 | True |
| 66ac5d53bd1c3999a4c8d437 | 60000 | 0 | equilibration4 | 2.0 | 0.05 | constant | -2.1927209 | 0.014968448 | 0.038504283 | 12.796 | 2.3609 | 0.50488 | True |
| 311f2cdc7ad177aed69324a4 | 60000 | 1 | equilibration4 | 0.5 | 0.0 | constant | -0.97488716 | 0.036755032 | 0.027003394 | 3.7565 | 1.0079 | 0.99609 | True |
| 3f335f0a24b5f215cc5f70ff | 60000 | 1 | equilibration4 | 0.5 | 0.05 | constant | -0.95549398 | 0.042024824 | 0.0043156942 | 3.5229 | 1.2001 | 0.9082 | True |
| 48bf8cdd648648c7641cb1dd | 60000 | 1 | equilibration4 | 2.0 | 0.0 | constant | -2.1974219 | 0.017176734 | 0.016728719 | 12.578 | 1.4287 | 0.83105 | True |
| 5f196c45a7173d515ce8cca6 | 60000 | 1 | equilibration4 | 2.0 | 0.05 | constant | -2.1908779 | 0.006857978 | 0.020160251 | 12.774 | 1.0039 | 0.99805 | True |
| 8f9d31153f56909cb383d7a7 | 60000 | 2 | equilibration4 | 0.5 | 0.0 | constant | -0.95552011 | 0.019646115 | 0.077963935 | 3.5826 | 1.1239 | 0.94238 | True |
| 2d444e268d13f290e207bbb5 | 60000 | 2 | equilibration4 | 0.5 | 0.05 | constant | -0.98468891 | 0.013165792 | 0.0063256158 | 3.5955 | 1 | 1 | True |
| 98f72a3f7cbe29f744f78e46 | 60000 | 2 | equilibration4 | 2.0 | 0.0 | constant | -2.197955 | 0.017781999 | 0.0079015096 | 12.15 | 1.0237 | 0.98828 | True |
| 7f495ce37b31a3dc0bb9c875 | 60000 | 2 | equilibration4 | 2.0 | 0.05 | constant | -2.1957058 | 0.0045199924 | 0.028821733 | 12.954 | 1.002 | 0.99902 | True |
| 098873bd4f1912f8d2546d97 | 60000 | 3 | equilibration4 | 0.5 | 0.0 | constant | -0.93077459 | 0.064823574 | 0.013562944 | 3.8179 | 1 | 1 | True |
| 8abfc51a4534537fb827bbc5 | 60000 | 3 | equilibration4 | 0.5 | 0.05 | constant | -0.99597252 | 0.0083156162 | 0.032289717 | 3.568 | 1.0098 | 0.99512 | True |
| 65427dfda507de1a1f312294 | 60000 | 3 | equilibration4 | 2.0 | 0.0 | constant | -2.2011946 | 0.014803145 | 0.033241696 | 12.675 | 1.0218 | 0.98926 | True |
| e72acca1eeda24aaa12b495c | 60000 | 3 | equilibration4 | 2.0 | 0.05 | constant | -2.2086321 | 0.018123627 | 0.026233673 | 12.874 | 1.0463 | 0.97754 | True |
| ae1547759bae555aa8649df4 | 60000 | 0 | forward2 | 0.5 | 0.0 | constant | -0.93632562 | 0.033237881 | 0.024511125 | 4.3295 | 1.0039 | 0.99805 | True |
| fbf3e61c58e5db93106522cd | 60000 | 0 | forward2 | 0.5 | 0.05 | constant | -0.95321628 | 0.033841451 | 0.016387092 | 4.3008 | 1 | 1 | True |
| 5b784b758e2115a4ba217ed0 | 60000 | 0 | forward2 | 2.0 | 0.0 | constant | -2.2120692 | 0.030219184 | 0.06710773 | 14.463 | 1 | 1 | True |
| 10c43ed6aa2a507a623aa48b | 60000 | 0 | forward2 | 2.0 | 0.05 | constant | -2.1962882 | 0.012666066 | 0.0054312812 | 15.732 | 1 | 1 | True |
| d14877e6c0b91725761fc339 | 60000 | 1 | forward2 | 0.5 | 0.0 | constant | -0.93900125 | 0.044423421 | 0.067255232 | 4.5228 | 1 | 1 | True |
| d60d56e0e61e4baa287626ba | 60000 | 1 | forward2 | 0.5 | 0.05 | constant | -0.9476684 | 0.029757606 | 0.024543338 | 4.2937 | 1 | 1 | True |
| 26d0aa089c9c7dad780a110f | 60000 | 1 | forward2 | 2.0 | 0.0 | constant | -2.1984957 | 0.0082465278 | 0.041937934 | 15.769 | 1 | 1 | True |
| 9fc2a652dd40e0950538e844 | 60000 | 1 | forward2 | 2.0 | 0.05 | constant | -2.2012402 | 0.012856801 | 0.029638078 | 15.801 | 1 | 1 | True |
| 504f6e7f9ef0f3dadf6ffbc3 | 60000 | 2 | forward2 | 0.5 | 0.0 | constant | -0.96464848 | 0.0060687595 | 0.028666178 | 4.3136 | 1 | 1 | True |
| a8fb58fd6451633487d1b130 | 60000 | 2 | forward2 | 0.5 | 0.05 | constant | -0.94571931 | 0.012588925 | 0.018244002 | 4.3101 | 1 | 1 | True |
| 7d974fb54c9d9fde1607d32c | 60000 | 2 | forward2 | 2.0 | 0.0 | constant | -2.1934928 | 0.00146993 | 0.020370483 | 15.595 | 1 | 1 | True |
| 78924323772d0628ee830b06 | 60000 | 2 | forward2 | 2.0 | 0.05 | constant | -2.2162048 | 0.010222965 | 0.031883664 | 15.566 | 1 | 1 | True |
| dace9edd030d9bcd3c69da21 | 60000 | 3 | forward2 | 0.5 | 0.0 | constant | -0.98397301 | 0.043721517 | 0.047092014 | 4.5073 | 1 | 1 | True |
| fd8a91948a67dcf235f86c82 | 60000 | 3 | forward2 | 0.5 | 0.05 | constant | -0.9439423 | 0.01754125 | 0.0044182671 | 4.2553 | 1 | 1 | True |
| 5059deca3f9c8c8a7b39aff3 | 60000 | 3 | forward2 | 2.0 | 0.0 | constant | -2.2050887 | 0.010445065 | 0.024937948 | 15.475 | 1 | 1 | True |
| c39ef08ef8c2ce0bc0a6d289 | 60000 | 3 | forward2 | 2.0 | 0.05 | constant | -2.1932522 | 0.011856079 | 0.0066257053 | 15.411 | 1 | 1 | True |

## Continuation and interpretation

The authorized next step is to finish the base pilot, then execute the fixed 144-run population/time escalation schedule. Its controller waits for successful base completion and acquires the shared supervisor lock. W256/W1024, separate equilibration4 and forward2 tests, and h=0.05 diagnose population loss, projection dependence and cost.

Ancestry recovery alone does not qualify physical observables. Stage C still requires measured feasible cost, eight development disorders, one-parameter convergence comparisons, alternative starts and combined checks. Any unresolved context remains unqualified. Final anchor scope and acceptance standards are identical for L=4,6,8,10.

At the recorded deadline workers checkpoint and pause; the instance stays on. Machine-readable source results and full cycle diagnostics are in `../notebook_stage_b_escalation/results/`. Snapshot provenance stores SHA-256 for every included result.
