"""M4: Teacher trajectory generation for M5 behavior cloning.

Produces teacher-guided graph reading trajectories. Supports two modes:
- Legacy: per-budget generation with per-step GNN teacher guidance
- Phased: 3-phase pipeline with pre-computed scores + batched classifier eval
"""
