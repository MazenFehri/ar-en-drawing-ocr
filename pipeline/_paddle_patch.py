"""Disable Paddle IR optimization to avoid an AVX-512 SIGILL crash.

Intel 12th/13th/14th-gen consumer CPUs (Alder Lake / Raptor Lake, e.g. the
i7-13650HX) have AVX-512 fused off — those instructions do not exist on the
chip. PaddlePaddle 2.6.x runs IR fusion passes during inference-graph
optimization (notably ``SelfAttentionFusePass``) that emit AVX-512
instructions, so the OS kills the process with SIGILL ("Illegal instruction").

Turning off IR optimization skips those passes; inference then runs on the
unoptimized graph using runtime-dispatched AVX2 kernels, which the CPU has.

Importing this module patches ``paddle.inference.create_predictor`` (the call
PaddleOCR/PPStructure use to build their predictors) so every predictor is
created with ir_optim disabled. The patch is idempotent.
"""
import paddle.inference as _pi

if not getattr(_pi.create_predictor, "_ir_optim_patched", False):
    _orig_create_predictor = _pi.create_predictor

    def _create_predictor_no_ir_optim(config):
        try:
            config.switch_ir_optim(False)
        except Exception:
            pass
        return _orig_create_predictor(config)

    _create_predictor_no_ir_optim._ir_optim_patched = True
    _pi.create_predictor = _create_predictor_no_ir_optim
