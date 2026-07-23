from __future__ import annotations

import os
import sys
import traceback

if os.environ.get("NATIVE_REPLAY_ADAPTER"):
    try:
        from minireplay.instrumentation import install

        install()
    except Exception:
        traceback.print_exc()
        sys.stderr.flush()
        os._exit(70)
