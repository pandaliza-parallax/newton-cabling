"""Rerun recording helpers that remove two recurring papercuts.

* :func:`open_rrd_recorder` keeps the ``.rrd`` file sink. ``ViewerRerun`` silently
  discards the file sink when it also spawns/serves a viewer; pretending to be a
  notebook skips the server launch so the recording is actually written.

* :func:`write_focused_blueprint` excludes the ground-plane shape so the camera
  frames the cm-scale connector instead of the metre-scale plane. The plane's
  shape index is known at build time (it is the shape returned by
  ``add_ground_plane``), so the caller passes it explicitly rather than the demos'
  earlier ritual of grepping the recording to rediscover that "shape_0 is the
  plane" each time.
"""

from __future__ import annotations

import warnings

import newton
from newton.viewer import ViewerRerun


def _always_notebook() -> bool:
    return True


def open_rrd_recorder(rrd_path: str, *, keep_history: bool = True) -> ViewerRerun:
    """A ViewerRerun that records to ``rrd_path`` instead of serving a viewer."""
    import newton._src.viewer.viewer_rerun as viewer_rerun_module

    viewer_rerun_module.is_jupyter_notebook = _always_notebook
    return ViewerRerun(record_to_rrd=rrd_path, keep_historical_data=keep_history)


def find_ground_plane_shapes(model: newton.Model) -> list[int]:
    """Shape indices of every ground/plane in the *finalized* model.

    Derived from ``model.shape_type`` (geometry type), NOT from the builder's
    shape order: once rods/cables are added, ``finalize`` reorders shapes so the
    builder index no longer matches the rendered ``/model/shapes/shape_N`` index.
    Reading the plane's type off the finalized model is the only robust way to
    find it -- this is the fix for recordings that opened showing "just the floor"
    because a guessed index excluded the wrong shape.
    """
    shape_types = model.shape_type.numpy()
    plane_type = int(newton.GeoType.PLANE)
    return [index for index in range(len(shape_types)) if int(shape_types[index]) == plane_type]


def write_focused_blueprint(
    rbl_path: str,
    *,
    excluded_shape_indices: list[int],
    app_id: str = "newton-viewer",
) -> str:
    """Write an ``.rbl`` that shows the scene with panels open and the given
    ``/model/shapes/shape_N`` entities (typically the ground plane) excluded.
    """
    import rerun.blueprint as rrb

    exclusions = [f"- /model/shapes/shape_{index}" for index in excluded_shape_indices]
    view = rrb.Spatial3DView(origin="/", contents=["+ /**", *exclusions])
    rrb.Blueprint(view, collapse_panels=False).save(app_id, rbl_path)
    return rbl_path


def auto_blueprint(rbl_path: str, model: newton.Model, *, app_id: str = "newton-viewer") -> str:
    """Write a focused blueprint for a recording, warning if a ground plane exists.

    The reliable rule for Newton rerun recordings is to NOT add a ground plane: the
    viewer instances identical shapes, so the plane's rendered index cannot be
    matched to a model index to exclude it, and the recording opens showing "just
    the floor". This guard makes that mistake loud instead of silent -- it still
    attempts the (best-effort) exclusion, but warns so the fix (omit
    ``builder.add_ground_plane()``) is obvious.
    """
    planes = find_ground_plane_shapes(model)
    if planes:
        warnings.warn(
            "model has a ground plane: the rerun recording will likely open showing only the "
            "floor, because the viewer instances shapes so the plane cannot be reliably excluded. "
            "Omit builder.add_ground_plane() in anything that records to .rrd.",
            stacklevel=2,
        )
    return write_focused_blueprint(rbl_path, excluded_shape_indices=planes, app_id=app_id)
