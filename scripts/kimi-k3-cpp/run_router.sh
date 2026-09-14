#!/bin/bash
# Run after Prefill(34) + Decode(38) are ready. Client hits :8000
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://80.5.17.34:30001 \
  --decode  http://80.5.17.38:30001 \
  --host 0.0.0.0 \
  --port 8000 \
  --mini-lb
