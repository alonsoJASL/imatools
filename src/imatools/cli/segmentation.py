import argparse
import json
import logging
import sys
from pathlib import Path

from imatools.core import image as core_image
from imatools.core import label as core_label
from imatools.core import segmentation as core_seg
from imatools.io import image_io

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OP_CHOICES = ['add', 'subtract', 'multiply', 'divide', 'and', 'or', 'xor']


def _add_label(parser, *, multiple=False, required=True, help_str="Label value(s) to process"):
    """Attach the -l/--label argument to a subparser.

    multiple=True  -> nargs="+"  (a list of label ints, e.g. -l 1 2 3)
    multiple=False -> a single label int (e.g. -l 5)

    Modes that take no label simply don't call this, so -l never shows up in
    their --help. This is the project convention for per-subcommand label args.
    """
    kwargs = dict(type=int, default=None, required=required, help=help_str)
    if multiple:
        kwargs["nargs"] = "+"
    parser.add_argument("-l", "--label", **kwargs)


def _build_parser() -> argparse.ArgumentParser:
    # Parent holds ONLY the universal flags; -l/--label is added per-subcommand via _add_label().
    parent_parser = argparse.ArgumentParser(add_help=False)
    parent_parser.add_argument('-in', '--input', type=Path, required=True, help="Input image path")
    parent_parser.add_argument('-out', '--output', type=Path, default=None, help="Output image path (optional)")
    parent_parser.add_argument('--overwrite', action='store_true', help="Overwrite existing output file")

    p = argparse.ArgumentParser(prog="imatools-segmentation", description="Segmentation tools")
    sub = p.add_subparsers(dest="command")

    pr = sub.add_parser(
        "collapse-to-value",
        aliases=['collapse', 'relabel_image'],
        help="Collapse foreground labels in an image to a single value", parents=[parent_parser],
    )
    _add_label(pr)
    pr.set_defaults(func=handle_relabel)

    pm = sub.add_parser("morph-label", help="Morph one label, mask back into image", parents=[parent_parser])
    _add_label(pm)
    pm.add_argument("--operation", required=True)
    pm.add_argument("--radius", type=int, default=3)
    pm.add_argument("--kernel", default="ball")
    pm.set_defaults(func=handle_morph_label)

    pe = sub.add_parser("extract-label", help="Extract one label, binarise, save as new image", parents=[parent_parser])
    _add_label(pe)
    pe.set_defaults(func=handle_extract_label)

    ps = sub.add_parser("show", help="Show labels present in the image", parents=[parent_parser])
    ps.set_defaults(func=handle_show_labels)   # no _add_label: 'show' takes no label

    pmsk = sub.add_parser("mask", help="Masks a label map image with a binary mask (defaults to 0)", parents=[parent_parser])
    pmsk.add_argument('-mask', '--mask-image', type=Path, required=True, help="Mask image path")
    pmsk.add_argument('-mask-value', '--mask-value', type=int, default=0, help="Value to write into the masked region")
    pmsk.add_argument('-ignore', '--ignore-image', type=Path, default=None, help="Path to ignore image (optional)")
    pmsk.set_defaults(func=handle_mask)

    pdel = sub.add_parser(
        "delete-labels",
        aliases=['remove-labels', 'remove'],
        help="Delete one or more labels from an image", parents=[parent_parser]
    )
    _add_label(pdel, multiple=True)
    pdel.set_defaults(func=handle_delete_labels)

    pmrg = sub.add_parser("merge-labels", aliases=['merge'], help="Merge multiple labels into one", parents=[parent_parser])
    _add_label(pmrg, multiple=True)
    pmrg.set_defaults(func=handle_merge_labels)

    psplt = sub.add_parser("split-label", aliases=['split', 'cc-split'], help="Split a label into its connected components", parents=[parent_parser])
    _add_label(psplt)
    psplt.add_argument("--open-image", action="store_true", help="Preprocess with morphological opening to remove small connections")
    psplt.add_argument("--open-radius", type=int, default=3, help="Radius for morphological opening preprocessing")
    psplt.set_defaults(func=handle_split_labels)

    padd = sub.add_parser("add", help="Add two label images voxelwise", parents=[parent_parser])
    padd.add_argument('-in2', '--secondary-image', type=Path, required=True, help="Second image to add")
    padd.set_defaults(func=handle_add)

    pop = sub.add_parser("op", help="Voxelwise operation between two images", parents=[parent_parser])
    pop.add_argument('-in2', '--secondary-image', type=Path, required=True, help="Second image")
    pop.add_argument('--operation', required=True, choices=OP_CHOICES, help="Operation to apply")
    pop.set_defaults(func=handle_op)

    pfill = sub.add_parser("fill", help="Fill gaps in a (multi)label segmentation", parents=[parent_parser])
    pfill.add_argument('-in2', '--secondary-image', type=Path, default=None, help="Optional previous segmentation to fill from")
    pfill.set_defaults(func=handle_fill)

    plrg = sub.add_parser("largest", help="Keep only the largest connected component", parents=[parent_parser])
    plrg.set_defaults(func=handle_largest)

    pgap = sub.add_parser("gaps", help="Show gaps in a (multi)label segmentation", parents=[parent_parser])
    pgap.add_argument('--binary', action='store_true', help="Treat input as already binary (default: multilabel)")
    pgap.set_defaults(func=handle_gaps)

    psw = sub.add_parser("swap", aliases=['swap-labels'], help="Swap labels old->new (lists and/or an inclusive range)", parents=[parent_parser])
    psw.add_argument('--old-labels', type=int, nargs='+', default=[], help="Labels to replace")
    psw.add_argument('--new-labels', type=int, nargs='+', default=[], help="Replacement labels (1:1 with --old-labels)")
    psw.add_argument('--old-label-range', type=str, default='', help="Inclusive range 'a:b' appended to --old-labels")
    psw.add_argument('--range-replace', type=int, default=None, help="Single value to map the whole --old-label-range to")
    psw.set_defaults(func=handle_swap)

    prs = sub.add_parser("resample", aliases=['smooth'], help="Resample/smooth a label image", parents=[parent_parser])
    prs.add_argument('--spacing', type=float, nargs=3, default=[1.0, 1.0, 1.0], help="Target spacing")
    prs.add_argument('--sigma', type=float, default=3.0, help="Gaussian sigma")
    prs.add_argument('--threshold', type=float, default=0.5, help="Smoothing threshold")
    prs.add_argument('--close', action='store_true', help="Morphological close after resampling")
    prs.set_defaults(func=handle_resample)

    pinr = sub.add_parser("inr", help="Convert an image to INR format", parents=[parent_parser])
    pinr.set_defaults(func=handle_inr)

    pwcomp = sub.add_parser("compare", help="Compare two label maps", parents=[parent_parser])
    pwcomp.add_argument('-in2', '--secondary-image', type=Path, required=True, help="Second image to compare")
    pwcomp.add_argument('--swap-axes', action='store_true', help="Swap axes of the second image before comparison")
    pwcomp.add_argument('--save-to-report', action='store_true', help="Save comparison results to a report file")
    pwcomp.set_defaults(func=handle_compare_label_maps)

    pcmb = sub.add_parser("combine", help="Combine all binary segmentations in a FOLDER (-in) into one label image", parents=[parent_parser])
    pcmb.add_argument('--pattern', default='*.nrrd', help="Glob for images in the input folder (default: *.nrrd)")
    pcmb.set_defaults(func=handle_combine)

    pchain = sub.add_parser("chain", help="Apply a sequence of single-label morphological ops in order", parents=[parent_parser])
    pchain.add_argument('--operations', nargs='+', required=True, help="Operations to apply in order (open/close/dilate/erode/fillholes)")
    pchain.add_argument('--labels', type=int, nargs='+', required=True, help="Label for each operation (1:1 with --operations)")
    pchain.add_argument('--radii', type=int, nargs='+', default=[], help="Radius per operation; if omitted, --radius is used for all")
    pchain.add_argument('--radius', type=int, default=3, help="Default radius when --radii is not given")
    pchain.add_argument('--kernel', default='ball')
    pchain.set_defaults(func=handle_chain)

    psharp = sub.add_parser("sharp-regions", help="Detect pointy/outlier regions of a label (distance-based)", parents=[parent_parser])
    _add_label(psharp)
    psharp.add_argument('--gauss-sigma', type=float, default=2.0, help="Gaussian sigma for the smooth reference")
    psharp.set_defaults(func=handle_sharp_regions)

    pcid = sub.add_parser("cc-identify", aliases=['identify'], help="Count a label's connected components + per-region volume", parents=[parent_parser])
    _add_label(pcid)
    pcid.set_defaults(func=handle_cc_identify)

    pcex = sub.add_parser("cc-extract", help="Save a label's distinct connected components as a new image", parents=[parent_parser])
    _add_label(pcex)
    pcex.add_argument('--min-voxel-size', type=int, default=None, help="Drop components smaller than this many voxels")
    pcex.set_defaults(func=handle_cc_extract)

    pcrp = sub.add_parser("cc-regionprops", aliases=['regionprops'], help="Report region shape stats for a label's components", parents=[parent_parser])
    _add_label(pcrp)
    pcrp.set_defaults(func=handle_cc_regionprops)

    pshv = sub.add_parser("show-voxels", help="Report voxel indices / world coords / bbox for a label", parents=[parent_parser])
    _add_label(pshv)
    pshv.set_defaults(func=handle_show_voxels)

    return p


def _load(path: Path):                       # core always gets a plain sitk.Image
    return image_io.load_image(path, return_contract=False)

def _save(image, output_path: Path, overwrite: bool = True):   # raw write
    image_io.save_image(image, output_path, overwrite=overwrite)

def _output_name(input_path: Path, suffix: str, token=None, ext: str = "nrrd") -> Path:
    """Derive '<stem>_<suffix>[_<token>].<ext>' next to the input image.

    `token` may be omitted (label-less modes), an int (zero-padded to 3 digits),
    or a str (used verbatim, e.g. a join of several labels like '1_2_3').
    """
    if not ext.startswith('.'):
        ext = f".{ext}"
    if token is None:
        tail = ""
    elif isinstance(token, int):
        tail = f"_{token:03d}"
    else:
        tail = f"_{token}"
    return input_path.parent / f"{input_path.stem}_{suffix}{tail}{ext}"

def _write_output(image, args, suffix: str, token=None, ext: str = "nrrd") -> int:
    """Resolve the output path (explicit --output, else derived) and save. Returns 0."""
    outpath = args.output or _output_name(args.input, suffix, token, ext)
    _save(image, outpath, args.overwrite)
    return 0


def handle_relabel(args):              # THIN: load -> 1 core call -> save
    if args.command == 'relabel_image':
        logger.warning("'relabel_image' is deprecated, use 'collapse-to-value' instead.")
    out = core_label.relabel_image(_load(args.input), args.label)
    return _write_output(out, args, "relabelled", args.label, ext="nii")

def handle_morph_label(args):          # WORKFLOW: delegates to core/segmentation
    out = core_seg.morph_label(_load(args.input), args.label, args.operation, args.radius, args.kernel)
    return _write_output(out, args, "morphed", args.label)

def handle_extract_label(args):        # WORKFLOW: delegates to core/label
    out = core_label.extract_single_label(_load(args.input), args.label, binarise=True)
    return _write_output(out, args, "extracted", args.label)

def handle_show_labels(args):          # INSPECT: report labels present, no output file
    labels = core_label.get_labels(_load(args.input))
    logger.info("Labels present in %s: %s", args.input, labels)
    return 0

def handle_mask(args):                 # WORKFLOW: delegates to core/image
    ignore = _load(args.ignore_image) if args.ignore_image else None
    out = core_image.mask_image(_load(args.input), _load(args.mask_image), args.mask_value, ignore_im=ignore)
    return _write_output(out, args, "masked", args.mask_value)

def handle_delete_labels(args):        # THIN: replace each given label with 0
    if args.command == 'remove-labels':
        logger.warning("'remove-labels' is deprecated, use 'delete-labels' instead.")
    out = _load(args.input)
    for label in args.label:
        out = core_label.exchange_labels(out, label, 0)
    return _write_output(out, args, "deleted", '_'.join(map(str, args.label)))

def handle_merge_labels(args):         # WORKFLOW: binarise each label then merge
    out = _load(args.input)
    label_images = [core_label.extract_single_label(out, label, binarise=True) for label in args.label]
    out = core_label.merge_label_images(label_images)
    return _write_output(out, args, "merged", '_'.join(map(str, args.label)))

def handle_split_labels(args):         # WORKFLOW: delegates to core/label
    out = core_label.split_label_into_components(
        _load(args.input), args.label, open_image=args.open_image, open_radius=args.open_radius
    )
    return _write_output(out, args, "split", args.label)

def handle_add(args):                  # THIN: add two images
    out = core_image.add_images(_load(args.input), _load(args.secondary_image))
    return _write_output(out, args, "added", args.secondary_image.stem)

def handle_op(args):                   # THIN: voxelwise op between two images
    out = core_image.image_operation(args.operation, _load(args.input), _load(args.secondary_image))
    return _write_output(out, args, "op", args.operation)

def handle_fill(args):                 # THIN: fill gaps in a (multi)label segmentation
    previous = _load(args.secondary_image) if args.secondary_image else None
    out = core_label.fill_gaps(_load(args.input), previous, multilabel_images=True)
    return _write_output(out, args, "filled")

def handle_largest(args):              # THIN: keep largest connected component
    out = core_image.extract_largest(_load(args.input))
    return _write_output(out, args, "largest")

def handle_gaps(args):                 # THIN: show gaps
    out = core_label.gaps(_load(args.input), multilabel=not args.binary)
    return _write_output(out, args, "gaps")

def handle_swap(args):                 # THIN+: parse old/new labels (+optional range), one core call
    old_labels = list(args.old_labels)
    new_labels = list(args.new_labels)
    if args.old_label_range:
        start, end = args.old_label_range.split(':')
        range_labels = list(range(int(start), int(end) + 1))
        old_labels.extend(range_labels)
        if args.range_replace is not None:
            new_labels.extend([args.range_replace] * len(range_labels))
    if not old_labels:
        logger.error("No labels to swap. Use --old-labels/--new-labels and/or --old-label-range.")
        return 1
    if len(old_labels) != len(new_labels):
        logger.error("old labels (%d) and new labels (%d) must match in count.", len(old_labels), len(new_labels))
        return 1
    out = core_label.exchange_many_labels(_load(args.input), old_labels, new_labels)
    return _write_output(out, args, "swapped")

def handle_resample(args):             # THIN: resample/smooth a label image
    out = core_image.resample_smooth_label(
        _load(args.input), spacing=args.spacing, sigma=args.sigma,
        threshold=args.threshold, im_close=args.close,
    )
    return _write_output(out, args, "resampled")

def handle_inr(args):                  # CONVERT: write an .inr file (not a standard image save)
    outpath = args.output or args.input.with_suffix('.inr')
    image_io.convert_to_inr(_load(args.input), outpath)
    logger.info("Wrote INR to %s", outpath)
    return 0

def handle_compare_label_maps(args):   # WORKFLOW: delegates to core/segmentation
    scores, comparison_image = core_seg.compare_label_maps(
        _load(args.input), _load(args.secondary_image), im2_swap_axes=args.swap_axes
    )
    for label, dice in scores.items():
        logger.info("label %s: dice = %s", label, dice)
    if args.save_to_report:
        report_path = args.input.with_suffix('.compare.json')
        with open(report_path, 'w') as f:
            json.dump({str(k): v for k, v in scores.items()}, f, indent=4)
        logger.info("Wrote comparison report to %s", report_path)
    return _write_output(comparison_image, args, "comparison")

def handle_combine(args):              # WORKFLOW: combine all binary segs in a folder
    folder = args.input
    files = sorted(folder.glob(args.pattern))
    if not files:
        logger.error("No images matching %s in %s", args.pattern, folder)
        return 1
    images = [_load(f) for f in files]
    out = core_label.combine_segmentations(images)
    outpath = args.output or (folder / "combined.nrrd")
    _save(out, outpath, args.overwrite)
    logger.info("Combined %d images -> %s", len(files), outpath)
    return 0

def handle_chain(args):                # WORKFLOW: in-memory chain of morph-label ops
    n = len(args.operations)
    if len(args.labels) != n:
        logger.error("--labels (%d) and --operations (%d) must have equal length.", len(args.labels), n)
        return 1
    radii = args.radii if args.radii else [args.radius] * n
    if len(radii) != n:
        logger.error("--radii (%d) must match --operations (%d).", len(radii), n)
        return 1
    out = core_seg.morph_label_chain(_load(args.input), args.labels, args.operations, radii, kernel=args.kernel)
    return _write_output(out, args, "chained")

def handle_sharp_regions(args):        # WORKFLOW: distance-based outlier (pointy bits) detection
    out = core_label.distance_based_outlier_detection(_load(args.input), args.label, gauss_sigma=args.gauss_sigma)
    return _write_output(out, args, "sharp_regions", args.label)

def handle_cc_identify(args):          # INSPECT: count a label's connected components + volumes
    im = _load(args.input)
    cc_image, regions, num = core_seg.connected_components(im, args.label)
    sx, sy, sz = core_image.get_spacing(im)
    voxel_vol = sx * sy * sz
    logger.info("Label %d -> %d connected component(s).", args.label, num)
    for r in regions:
        n_vox = core_image.get_num_nonzero_voxels(cc_image == r)
        logger.info("  region %d: %d voxels (%.3f mm^3)", r, n_vox, n_vox * voxel_vol)
    return 0

def handle_cc_extract(args):           # WORKFLOW: save a label's distinct components as an image
    im = _load(args.input)
    cc_image, regions, _ = core_seg.connected_components(im, args.label)
    if args.min_voxel_size is not None:
        cc_image = core_seg.ignore_small_labels(cc_image, regions, args.min_voxel_size)
    return _write_output(cc_image, args, "components", args.label)

def handle_cc_regionprops(args):       # INSPECT: shape stats of a label's components
    im = _load(args.input)
    cc_image, _, _ = core_seg.connected_components(im, args.label)
    logger.info("Region properties for label %d: %s", args.label, core_image.regionprops(cc_image))
    return 0

def handle_show_voxels(args):          # INSPECT: voxel indices / world coords / bbox of a label
    im = _load(args.input)
    vox, world, bboxes = core_image.get_indices_from_label(im, args.label, get_voxel_bbox=True)
    logger.info("Label %d: %d voxel(s)", args.label, len(vox))
    logger.info("  world coords (first 5): %s", world[:5])
    logger.info("  bbox centres (first 5): %s", bboxes["centres"][:5])
    return 0

def main():
    parser = _build_parser()
    args = parser.parse_args()

    if not getattr(args, "func", None):
        parser.print_help()
        return 1
    return args.func(args)

if __name__ == "__main__":
    sys.exit(main())
