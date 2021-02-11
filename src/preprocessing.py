import os
import json

from shapely import geometry
import utils
from utils import jread, jdump, json_record_to_poly, get_cortex_polygs
from pathlib import Path, PurePath


def filt_glom_by_cortex(img_name: str, json_path_in: str, json_path_out: str, suffix: str = '.json',
                        postfix_anot_in: str = "-anatomical-structure",
                        postfix_img_out: str = "_prep") -> None:
    """ Filter glomerulus by location in Cortex. Load -> Filter -> Save.
    """

    assert json_path_in != json_path_out or postfix_img_out != "", "Please change paths or postfix to " \
                                                                   "avoid overwriting input files"

    gloms_json = jread(PurePath(json_path_in, img_name + suffix))
    anot_structs_json = jread(PurePath(json_path_in, img_name + postfix_anot_in + suffix))

    # Get list of Cortex polygons
    cortex_polygs = utils.get_cortex_polygs(anot_structs_json)
    assert len(cortex_polygs) != 0, "No Cortex"

    filt_gloms_json = []
    for record in gloms_json:
        polygs = utils.json_record_to_poly(record)
        # If at least one polygon intersects with at least one Сortex polygon, then append to filt_gloms_json
        if any([polyg.intersects(cortex_polyg) for cortex_polyg in cortex_polygs for polyg in polygs]):
            filt_gloms_json.append(record)
    assert len(filt_gloms_json) != 0, "No intersections are found"
    print(f"{len(gloms_json) - len(filt_gloms_json)} glomerulus are removed in {img_name} ")

    os.makedirs(Path(json_path_out), exist_ok=True)
    jdump(filt_gloms_json, PurePath(json_path_out, img_name + postfix_img_out + suffix))
