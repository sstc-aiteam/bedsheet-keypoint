```
Single Task LoRA + MAML (Current Setup)
┌──────────────────────────────┐
│ Base Model (Meta CLIP + LoRA)│
└───────────────┬──────────────┘
                │ initialization θ
        ┌───────▼────────┐
        │ Support Split  │  (same keypoint class, e.g., bedsheet corners)
        │   of task T₁   │
        └───────┬────────┘
                │ Inner-loop optimizer (AdamW) for K steps
                ▼
        ┌──────────────┐
        │ Adapted θ'   │
        └───────┬──────┘
                │ Evaluate on Query split of T₁
                ▼
        ┌──────────────┐
        │ Query Loss   │
        └───────┬──────┘
                │ Backprop through inner loop
                ▼
        ┌──────────────┐
        │ Meta Update  │ (AdamW) ← produces improved θ
        └──────────────┘

Key characteristics:
- Only one task T₁ ⇒ minimal task diversity.
- Inner and outer loops see near-duplicate data, so meta-step = standard fine-tuning.
- MAML overhead offers little convergence benefit over direct LoRA fine-tuning.
```

```
Multi-Class LoRA + MAML (Proposed Design)
┌─────────────────────────────────────────┐
│ Shared Base Model (Meta CLIP + LoRA) θ  │
└───────────────┬─────────────────────────┘
                │
                ├─────────────┬─────────────┬─────────────┐
                │             │             │             │
        ┌───────▼───┐ ┌───────▼───┐ ┌───────▼───┐ ┌───────▼───┐
        │ Task T₁   │ │ Task T₂   │ │ Task T₃   │ │ Task T₄   │
        │ bedsheet  │ │ fitted    │ │ pillow    │ │ blanket   │
        │ corners   │ │ corners   │ │ tags      │ │ edges     │
        └───────┬───┘ └───────┬───┘ └───────┬───┘ └───────┬───┘
                │             │             │             │
   For each meta-batch:
   1. Sample tasks (e.g., T₁, T₃, T₄)
   2. For each task, draw support & query splits using the task-specific
      heatmap channels (multi-channel output head).
   3. Inner-loop: adapt θ → θ'_task using support split (task-conditioned).
   4. Query loss per task backpropagates through inner steps.
   5. Aggregate meta-gradient across tasks.
   6. Outer loop optimizer (AdamW) updates θ for better quick adaptation.
```

### Why Multitask MAML Beats Single-Task Fine-Tuning
- **Task diversity drives meta-learning.** When the model sees many related tasks, it learns an initialization that works well after just a few gradient steps for *any* of them. Single-task training can only optimize for that task’s distribution; there is no incentive to learn a broadly adaptable prior.
- **Rapid adaptation to new keypoint types.** Once trained on multiple keypoint classes, the model can adapt to unseen or low-data classes in a handful of inner-loop updates, leveraging the shared features captured during meta-training.
- **Avoids catastrophic forgetting.** The shared base model learns to balance gradients coming from different keypoint styles. Traditional fine-tuning on one class at a time risks overwriting weights learned for earlier classes unless you use careful replay/balancing strategies.
- **Better use of LoRA capacity.** LoRA adapters become a compact, shared parameter space. Meta-training tunes θ so the adapters can bend quickly toward any specific class without large updates, preserving the low-rank adaptation benefits.

### Dataset Considerations
- **Yes, use multiple datasets when available.** Each keypoint class should provide its own labeled samples. You can combine them in a single loader but keep task IDs so support/query splits stay homogeneous per task.
- **Domain-based tasks when classes are limited.** If object classes are few, create pseudo-tasks by domain (lighting setups, camera angles, fabric patterns). MAML only needs that tasks differ in a meaningful way so the model learns to adapt.
- **Balanced sampling matters.** Ensure meta-batches cover the range of tasks evenly. Otherwise the initialization will skew toward the majority class, reducing the multi-task benefit.

With the proposed multi-task setup, MAML acts as a meta-optimizer on top of the inner-loop AdamW, producing an initialization that generalizes across keypoint types and adapts faster than conventional single-task LoRA fine-tuning.
