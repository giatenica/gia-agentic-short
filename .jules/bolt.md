## 2024-05-20 - Initial Discovery: Caching Redundant Property Lookups

**Learning:** Data-intensive classes, like `TimeTrackingReport`, often have properties that repeatedly filter or compute over the same internal data structures. Accessing these properties multiple times triggers expensive, redundant computations, creating a classic N+1-style performance bottleneck within a single object.

**Action:** For future optimizations, I will actively look for properties that derive data from lists or other collections and apply `functools.cached_property` (or an equivalent memoization technique) to cache the results. This ensures the computation runs only once per instance, significantly improving performance on objects with frequent property access.
