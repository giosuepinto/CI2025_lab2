# TSP lab2 Computational Intelligence

Author: Giosuè Pinto (s342711@studenti.polito.it)
Github: https://github.com/giosuepinto

In collaboration with Francesco Palmisani s343429


-----------------------
1. Project Overview
-----------------------

This project is a high-performance solver for the Travelling Salesman Problem (TSP).

It is an **Adaptive Hybrid Memetic Algorithm** written in Python, accelerated with **Numba**.

**Attention: only works for python 3.10 - 3.13 because of Numba dependency**

It is designed to solve multiple classes of TSP problems by auto-detecting the problem type:
* **Symmetric TSP (G-type):** `dist(A,B) == dist(B,A)`
* **Asymmetric ATSP (R1 & R2-type):** `dist(A,B) != dist(B,A)`

The algorithm's core feature is its **adaptive strategy**. It analyzes the problem size (N) and a pre-defined `COMPUTATIONAL_BUDGET` to select the most powerful solving strategy that can run within a reasonable time, automatically scaling from a "Pure Memetic" algorithm for small N to a "Hybrid GA-ILS" for large N.

-----------------------
2. The Development Journey
-----------------------

1.  **Initial Heuristic:** We began with a "Nearest Neighbour" heuristic, which was fast but produced poor-quality solutions. This was improved by adding probabilistic selection (GRASP-style) and running it multiple times (Multi-Start).

2.  **Genetic Algorithm (GA):** We evolved the heuristic into a GA, using the "smart init" for the initial population and adding core operators: Tournament Selection, Order Crossover (OX1), and Inversion Mutation (2-Opt).

3.  **Memetic Algorithm (MA):** The GA was enhanced into a Memetic Algorithm by integrating a `local_search_2opt` function. Applying this $O(N^3)$ local search to offspring made the algorithm much more effective at finding high-quality solutions.

4.  **Scaling Crisis & Optimization:** This MA was fast for N=50 but failed for N >= 200. We identified two types of bottlenecks:
    * **$O(N^4)$ Strategic Bottleneck:** Calling the $O(N^3)$ 2-Opt function $O(N)$ times per generation.
    * **$O(N^3)$ Implementation Bottleneck:** Hidden $O(N^2)$ costs in our pure-Python `order_crossover` and a $O(N^2 log N)$ `smart_probabilistic_init` function, which, when called $O(N)$ times, were just as slow.

5.  **Final Solution:**
    * **Code Optimization:** `order_crossover` was re-written to be $O(N)$. A "fast" $O(N)$ `fast_random_init` (permutation) was added for large-scale problems.
    * **Numba JIT:** All critical-path functions (`local_search_2opt`, `calculate_fitness`, `order_crossover`, `inversion_mutation`) were decorated with `@jit(nopython=True)` to compile them to machine code, providing a 10-100x speedup.
    * **Asymmetric Solvers:** New $O(N^3)$-style operators (`local_search_insertion`) and $O(1)$ mutations (`swap_mutation`) were added to handle asymmetric R1 and R2 problems.
    * **Adaptive Strategy:** The final algorithm uses this new speed to check its computational budget and automatically select the best strategy (Pure, Restricted, or Hybrid) that fits the problem size.

-----------------------
3. Algorithm Components
-----------------------

This solver is modular. It selects the right "tool" for the job based on problem type.

* **Problem Analysis:**
    * Checks `is_symmetric` to decide which operators to use.
    * Checks `has_negatives` to disable the `smart_probabilistic_init`, which fails with negative distances.

* **Symmetric Operators (G-type):**
    * `local_search_2opt`: The $O(N^3)$ local search workhorse. Based on path inversion.
    * `inversion_mutation`: The $O(N)$ mutation operator.
    * `perturbation_tsp`: The ILS "kick" (multiple inversions).

* **Asymmetric Operators (R1, R2-type):**
    * `local_search_insertion`: The $O(N^3)$-style local search. Finds the best city to move and re-insert elsewhere.
    * `swap_mutation`: The $O(1)$ mutation operator.
    * `perturbation_atsp`: The ILS "kick" (multiple swaps).

* **Common Components:**
    * `calculate_fitness`: $O(N)$ fitness function (works for all problem types).
    * `order_crossover`: $O(N)$ crossover (works for all problem types).
    * `tournament_selection`: $O(k)$ selection.
    * `smart_probabilistic_init` / `fast_random_init`: Heuristic or fast initializers.



ALGORITHM OPERATOR SUMMARY
-------------------------------------------

inversion_mutation (Inversion Mutation)
-------------------------------------------
* PURPOSE: To introduce a small perturbation (a single random 2-Opt move) into a solution.
* HOW IT WORKS:
    1.  Takes a solution: [1, 2, 3, 4, 5, 6, 7]
    2.  Picks two random indices (e.g., 2 and 5).
    3.  Identifies the segment between them: [3, 4, 5, 6]
    4.  INVERTS the segment: [6, 5, 4, 3]
    5.  Replaces the original segment.
    6.  RESULT: [1, 2, 6, 5, 4, 3, 7]
* COST LOGIC: It breaks the edges (2->3) and (6->7) and reconnects them "crossed" (2->6) and (3->7).  This only works if the cost of path (3-4-5-6) is equal to (6-5-4-3).

local_search_2opt (Systematic 2-Opt Optimizer)
--------------------------------------------------
* PURPOSE: To improve a solution until it reaches a "local optimum" (no 2-Opt move can improve it further). This is a hill-climbing descent algorithm.
* HOW IT WORKS:
    1.  Takes a solution (e.g., Cost 100).
    2.  Starts a double loop 'for i' and 'for j' to test EVERY possible 2-Opt inversion.
    3.  ATTEMPT (i=1, j=4): Vector [1, | 5, 2, 3, 6, | 4, 7]
    4.  Segment: [5, 2, 3, 6]. Inversion: [6, 3, 2, 5]
    5.  New Vector: [1, 6, 3, 2, 5, 4, 7]
    6.  COST CHECK (O(1) Delta): Calculates only the difference:
        (cost(1,6) + cost(5,4)) - (cost(1,5) + cost(6,4))
    7.  If the Delta is negative (e.g., new Cost 95), the move is an improvement.
    8.  ACTION: Keeps the new solution and RESTARTS the entire process from the beginning.
    9.  TERMINATES: Only when a full double-loop finds no improvements.


swap_mutation (Swap Mutation)
--------------------------------
* PURPOSE: To introduce an "asymmetry-safe" mutation.
* HOW IT WORKS:
    1.  Takes a solution: [1, 2, 3, 4, 5, 6, 7]
    2.  Picks two random indices (e.g., 2 and 5).
    3.  Values at indices: 3 and 6
    4.  SWAPS the values:
    5.  RESULT: [1, 2, 6, 4, 5, 3, 7]
* COST LOGIC: Breaks 4 edges (e.g., 2->3, 3->4, 5->6, 6->7) and creates 4 new ones (2->6, 6->4, 5->3, 3->7). The cost must be explicitly recalculated.

local_search_insertion (Insertion Optimizer)
-----------------------------------------------
* PURPOSE: The 2-Opt equivalent for asymmetric problems.
* HOW IT WORKS:
    1.  Takes a solution (e.g., Cost 100).
    2.  Starts a double loop 'for i' (city to move) and 'for j' (position to insert after).
    3.  ATTEMPT (i=1, j=3): Move city '5' (at index 1) and insert it AFTER city '3' (at index 3). 
    4.  INITIAL STATE: [...1 -> 5 -> 2...] and [...3 -> 4...]
    5.  FINAL STATE: [...1 -> 2...] and [...3 -> 5 -> 4...]
    6.  COST CHECK (O(1) Delta):
        * Costs Removed: cost(1,5) + cost(5,2) + cost(3,4)
        * Costs Added:   cost(1,2) + cost(3,5) + cost(5,4)
        * Delta = Added - Removed
    7.  If Delta < 0, the move is an improvement.
    8.  ACTION: Saves the move (if "best improvement") or applies it (if "first improvement") and restarts.
    9.  TERMINATES: When a full double-loop finds no improvements.


order_crossover (OX1)
-------------------------
* PURPOSE: To create a valid child from two parents, preserving a block from P1 and the relative order from P2. 
* HOW IT WORKS:
    1.  Parent 1 (P1): [8, 4, | 7, 3, 6, | 2, 5, 1, 9]
    2.  Parent 2 (P2): [1, 2, 3, 4, 5, 6, 7, 8, 9]
    3.  A random segment is chosen from P1 (e.g., indices 2-4): [7, 3, 6]
    4.  The segment is copied to the child:
        Child: [_, _, 7, 3, 6, _, _, _, _]
    5.  A lookup (boolean array) is created for the copied cities: {7, 3, 6}
    6.  Iterate through P2 from the start, filling the gaps (_) with cities NOT in the lookup:
        * P2[0] = 1 (Not in lookup) -> Child: [1, _, 7, 3, 6, _, _, _, _]
        * P2[1] = 2 (Not in lookup) -> Child: [1, 2, 7, 3, 6, _, _, _, _]
        * P2[2] = 3 (IS in lookup)   -> Skip
        * P2[3] = 4 (Not in lookup) -> Child: [1, 2, 7, 3, 6, 4, _, _, _]
        * ...and so on...
    7.  RESULT: [1, 2, 7, 3, 6, 4, 5, 8, 9]



-----------------------
4. The Final Adaptive Algorithm Explained
-----------------------

The algorithm's core logic is **budget-aware strategy selection**.

1.  **Budget Calculation:** The algorithm estimates the cost of a single local search call (e.g., `(N^3 / 10)`) and calculates the total number of calls (`budget_in_ls_calls`) it can afford within the global `COMPUTATIONAL_BUDGET`.
2.  **Operator Selection:** It checks `is_symmetric` and assigns the correct set of functions (e.g., `local_search_func = local_search_2opt` or `local_search_func = local_search_insertion`).
3.  **Strategy Choice:**
    * **If `N <= 100` AND budget is sufficient:** Selects **"Pure Memetic"**. All $\lambda$ offspring will be optimized by local search. Uses `smart_probabilistic_init`.
    * **Else, if budget is sufficient:** Selects **"Restricted Memetic"**. Only $k=50$ offspring are optimized per generation. Uses `smart_probabilistic_init` (if $N$ is small and non-negative) or `fast_random_init`.
    * **Else (Large N / Low Budget):** Selects **"Hybrid GA-ILS"**. No offspring are optimized during the GA. Uses `fast_random_init`.

4.  **Execution:**
    * **Memetic Modes:** Run the GA loop, applying local search to `k` or `all` offspring until convergence (`STALL_LIMIT`).
    * **Hybrid GA-ILS Mode:**
        * **Phase 1 (GA-Pure):** Runs a very fast "raw" GA loop ($O(N^2)$) for `STALL_LIMIT` generations. This phase explores the solution space to find the best *raw champion* (a promising region).
        * **Phase 2 (ILS):** Takes the best *raw champion* from Phase 1. It then spends its *entire remaining* `budget_in_ls_calls` on an Iterated Local Search loop (`Perturb -> Optimize -> Compare -> ...`) to deeply exploit that region.

-----------------------
5. Conclusions
-----------------------

This adaptive, multi-operator algorithm is robust and scalable.

* **Performance:** By leveraging Numba JIT compilation, the $O(N^3)$ bottlenecks become manageable, allowing the algorithm to find high-quality solutions for problems up to N=1000 in minutes.
* **Adaptability:** The budget-based strategy selection is critical. It automatically uses the "best" algorithm (Memetic) when it can afford to, and gracefully degrades to a "good enough" algorithm (Hybrid GA-ILS) when it can't.
* **Flexibility:** By detecting problem properties (symmetry, negative costs), the algorithm correctly swaps its core logic to handle both TSP and ATSP problems with a single codebase.