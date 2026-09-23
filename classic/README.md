# Knight's Tour — classic solutions

Hand-coded solutions to the [Knight's tour problem](https://en.wikipedia.org/wiki/Knight%27s_tour)
on an 8x8 board (with condition that the starting point = left upper corner).

1- knights-tour-Backtracking.py shows the solution using Backtracking method.  
WORST EXECUTION TIME is 22 seconds

2- knights-tour-Warnsdorffs-rule.py shows the solution using Warnsdorff's rule.  
WORST EXECUTION TIME is 0.0008 seconds

Comparing worst case to worst case, Warnsdorff's heuristic improves the performance by 27500 times at least.

```bash
python3 knights-tour-Backtracking.py
python3 knights-tour-Warnsdorffs-rule.py
```
