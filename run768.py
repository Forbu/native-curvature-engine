"""d=768 / 12-layer run: HybridSOAP@1e-2 vs AdamW@3e-3 (TinyStories)."""
import gpt_gpu as g

g.LOG_EVERY = max(25, g.STEPS // 12)

tr, va, V = g.get_data()
print(f"arch {g.N_LAYER}L d={g.D} h={g.N_HEAD} | steps {g.STEPS} "
      f"batch {g.BATCH}", flush=True)

for nm, mk, lr in [
        ("AdamW", lambda lr: g.AdamWOpt(lr), 3e-4),
        ("AdamW", lambda lr: g.AdamWOpt(lr), 1e-3),
        ("HybridSOAP", lambda lr: g.SOAPG(lr, hybrid=True), 1e-3),
        ("HybridSOAP", lambda lr: g.SOAPG(lr, hybrid=True), 3e-3)]:
    h, ms = g.train_run(nm, mk, lr, tr, va, V)
    print(f"DONE {nm}@{lr:g}: last {h['val'][-1]:.4f}  {ms:.0f} ms/step",
          flush=True)
