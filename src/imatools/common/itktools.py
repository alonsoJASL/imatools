## Tools for header correction orientation
# fix_header_to_axis_aligned moved to core.spatial (T2a3); re-exported via shim below.


# fix_header_and_save, load_image_as_np, load_nrrd_base, load_nrrd_image
# moved to io.image_io (T2a4); re-exported via shim below.
# set_direction_as moved to core.spatial (T2a3); re-exported via shim below.


# load_image migrated: superseded by imatools.io.image_io.load_image(path,
# return_contract=False) — same sitk.ReadImage passthrough, already existed
# there for the ImageContract-based API (M2a-1 straggler).


# get_nrrd_header migrated: superseded by imatools.io.image_io.get_nrrd_header
# (duplicate; the io.image_io copy is the tested canonical one). Re-exported via
# the io.image_io shim block below so the legacy name still resolves (M2b).
# remove_label deleted (M2b — dead: core.label.exchange_labels(im, label, 0) covers it).
# show_labels deleted (M2b — dead: cli/segmentation `show` / core.label.get_labels covers it).


# convert_to_inr and convert_from_inr moved to io.image_io (T2a4);
# re-exported via shim below.


# save_image migrated: superseded by imatools.io.image_io.save_image(image,
# output_path, overwrite=True) — already existed there for the ImageContract-
# based API, and already accepts a plain sitk.Image (M2a-1 straggler). Callers
# now pre-join dir+name into a single output_path.


# pointfile_to_image moved to io.image_io (T2a4); re-exported via shim below.


# get_mask_array_with_restrictions migrated to core.image (M2a-1 straggler);
# no shim (its only caller, core/image.py, now calls it directly, in-module).
# get_mask_with_restrictions migrated to core.image (M1.6a straggler);
# re-exported via shim below.


# check_for_existing_label migrated to core.label (M2a-1 straggler); no shim
# (its only caller, cli/scar.py, now imports it directly from core.label).


# create_normal_vector_for_plane, create_image_at_plane,
# create_image_at_plane_from_vector moved to core.spatial (T2a3);
# re-exported via shim below.


# get_scarq_boundaries migrated to core.image (M2a-1 straggler);
# no shim (its only caller, core/image.py, now calls it directly, in-module).


# ---------------------------------------------------------------------------
# Re-export shims — MUST be at the very bottom of this file so that itktools
# finishes defining its own helpers (imview, …) BEFORE core.label / core.image
# are imported.
# These bindings make the moved names available in the itktools namespace so
# that any code doing `from imatools.common.itktools import <name>` continues
# to work, and functions defined above that call them resolve correctly.
# ---------------------------------------------------------------------------
from imatools.core.label import (  # noqa: E402,F401,I001
    binarise,
    bwlabeln,
    combine_segmentations,
    compare_images,
    dice_score,
    distance_based_outlier_detection,
    exchange_labels,
    exchange_labels_form_json,
    extract_single_label,
    fill_gaps,
    gaps,
    get_labels,
    get_labels_to_exchange,
    get_labels_volumes,
    merge_label_images,
    multilabel_comparison,
    relabel_image,
    swap_labels,
)
from imatools.core.label import (  # noqa: E402,F401,I001
    exchange_many_labels,
    split_label_into_components as split_labels_on_repeats,  # renamed in migration; legacy name preserved
)
from imatools.core.image import (  # noqa: E402,F401,I001
    add_images,
    array2im,
    cp_image,
    extract_largest,
    find_neighbours,
    generate_scar_image,
    get_indices_from_label,
    get_mask_with_restrictions,
    get_num_nonzero_voxels,
    get_spacing,
    image_operation,
    imarray,
    imview,
    mask_image,
    morph_operations,
    points_to_image,
    regionprops,
    resample_smooth_label,
    segmentation_curvature,
    segmentation_curvature_value,
    simple_mask,
    smooth_label_with_distance,
    smooth_labels,
    swap_axes,
    zeros_like,
)
from imatools.core.spatial import (  # noqa: E402,F401,I001
    create_image_at_plane,
    create_image_at_plane_from_vector,
    create_normal_vector_for_plane,
    fix_header_to_axis_aligned,
    set_direction_as,
)

# io.image_io shim — file-I/O functions migrated to io/image_io (T2a4).
# Direct re-export (object-identity); the earlier lazy-wrapper workaround for the
# io/__init__ MeshType import crash is no longer needed — contracts now exports MeshType.
from imatools.io.image_io import (  # noqa: E402,F401
    convert_from_inr,
    convert_to_inr,
    fix_header_and_save,
    get_nrrd_header,
    load_image_as_np,
    load_nrrd_base,
    load_nrrd_image,
    pointfile_to_image,
)
