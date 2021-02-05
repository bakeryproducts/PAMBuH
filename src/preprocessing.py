import os
import json

from shapely import geometry
from shapely.geometry import Polygon
from utils import create_dir, jread, jdump


def filt_glom_by_cortex(img: str, json_path_in: str, json_path_out: str) -> None:
    """ Filter glomerulus by location in Cortex. Load -> Preprocess -> Save.
    """

    assert json_path_in != json_path_out, "json_path_in and json_path_out should be different to avoid overwriting"
    # Load
    gloms = jread(f'{json_path_in}{img}.json')
    anot_structure = jread(f'{json_path_in}{img}-anatomical-structure.json')
    # Create Cortex polygons
    polygs_anom = []
    for struct in anot_structure:
        if struct['properties']['classification']['name'] == 'Cortex':
            if struct['geometry']['type'] == 'MultiPolygon':
                for coord in struct['geometry']['coordinates']:
                    polygs_anom.append(geometry.Polygon(coord[0]))
            elif struct['geometry']['type'] == 'Polygon':
                polygs_anom.append(geometry.Polygon(struct['geometry']['coordinates'][0]))
            else:
                raise Exception("Invalid type value")
    assert len(polygs_anom) != 0, "No Cortex"
    # Filter glomerulus
    gloms_in_cortex = []
    for glom in gloms:
        if any([Polygon(glom['geometry']['coordinates'][0]).intersects(polyg) for polyg in polygs_anom]):
            gloms_in_cortex.append(glom)
    print(f"{len(gloms) - len(gloms_in_cortex)} glomerulus are removed in {img} ")
    # Save
    create_dir(json_path_out)
    jdump(gloms_in_cortex, f"{json_path_out}{img}.json")

