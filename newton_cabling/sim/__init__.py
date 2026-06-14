"""GPU-backed helpers that call into Newton and Warp.

These modules require the ``sim`` optional dependency (``newton[examples]``) and a
working Warp device, so they are imported explicitly by the runner scripts rather
than re-exported from the top-level :mod:`newton_cabling` package. That keeps
``import newton_cabling`` free of any GPU dependency.

Because Warp's kernel signatures use ``wp.array(dtype=...)`` calls in annotation
position (not standard types), these modules are linted but excluded from the
``ty`` gate; they are verified by execution on a GPU box.
"""
