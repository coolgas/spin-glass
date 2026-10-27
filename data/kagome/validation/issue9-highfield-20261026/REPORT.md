# L6 stronger-field energy screen

Flat guide, same sample0; total energies per replica. Independent populations supply uncertainties.

| Gamma/sigma | Walkers | Late energy | SE | Late drift | Drift 95% interval |
|---|---:|---:|---:|---:|---|
| g3_m5000 | | -344.747539 | 0.080377 | -0.151429 | [-0.4841472089570153, 0.18128823542597056] |
| g3_m10000 | | -344.733091 | 0.038299 | -0.079096 | [-0.28392310211611343, 0.12573025450083092] |
| g5_m5000 | | -552.476478 | 0.019745 | -0.024075 | [-0.11411116640425083, 0.06596136930246492] |
| g5_m10000 | | -552.482694 | 0.011308 | -0.023680 | [-0.053251417336519516, 0.005891273132356836] |

Walker contrasts (10,000 minus 5,000), independent-population Welch intervals:
{
  "3.0": {
    "mean": 0.01444767164031191,
    "se": 0.08903512988623205,
    "df": 10.022792493388192,
    "ci95": [
      -0.1838738267528246,
      0.21276917003344842
    ],
    "within_working_tolerance": false
  },
  "5.0": {
    "mean": -0.006216541108074125,
    "se": 0.022753765630061838,
    "df": 11.146007583568235,
    "ci95": [
      -0.0562173175649009,
      0.04378423534875265
    ],
    "within_working_tolerance": true
  }
}

Field-specific references:
{
  "3.0": {
    "all_x_upper_bound": -324.0,
    "saved_product_upper_bound": -155.55648789011275,
    "variational_upper_bound": -324.0,
    "second_order_high_field_energy": -345.1053892533624
  },
  "5.0": {
    "all_x_upper_bound": -540.0,
    "saved_product_upper_bound": -163.6377968310835,
    "variational_upper_bound": -540.0,
    "second_order_high_field_energy": -552.6632335520175
  }
}

Energy stability is not a phase determination. No production settings are qualified by this first screen.
