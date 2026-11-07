import copy, json, time
import torch
from utils.config import *
from utils.log import log
from ptflops import get_model_complexity_info

def compute_complexity(model):
    net = copy.deepcopy(model.module if hasattr(model, "module") else model).to("cpu").eval()
    total_params = sum(p.numel() for p in net.parameters())
    trainable_params = sum(p.numel() for p in net.parameters() if p.requires_grad)
    log.info(f"Total parameters: {total_params:,} | trainable: {trainable_params:,}")

    x = torch.randn(1, 3, *IMAGE_SIZE)
    t0 = time.time()
    with torch.no_grad():
        net(x)
    infer_ms = (time.time() - t0) * 1000       # single CPU forward pass, no warm-up (as in the original notebook)
    log.info(f"Inference time (1 image, CPU, single pass): {infer_ms:.1f} ms")

    complexity = dict(total_params=total_params, trainable_params=trainable_params, inference_ms_cpu_single_pass=infer_ms)
    try:                                        # pip install ptflops
        macs, _ = get_model_complexity_info(net, (3, *IMAGE_SIZE), as_strings=False, print_per_layer_stat=False, verbose=False)
        complexity["macs"] = macs
        log.info(f"MACs: {macs:,}")
    except ImportError:
        log.info("ptflops not installed - MACs skipped")
    (OUT_DIR / "complexity.json").write_text(json.dumps(complexity, indent=2))