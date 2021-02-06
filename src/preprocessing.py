import os
import json

from shapely import geometry
import utils
# do full path imports, no need of create_dir, func there is os.makedirs(exists_ok=bool)
from utils import create_dir, jread, jdump


def filt_glom_by_cortex(img: str, json_path_in: str, json_path_out: str) -> None:
    """ Filter glomerulus by location in Cortex. Load -> Preprocess -> Save.
    """

    assert json_path_in != json_path_out, "json_path_in and json_path_out should be different to avoid overwriting"
    gloms_json = jread(f'{json_path_in}{img}.json')
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

    # Filter glomerulus inside anot_structure
    filtered_json = []
    for record in gloms_json:
        poly = utils.json_record_to_poly(record)
        # TODO: if there are many anatomical polygons, intersect greedy 
        if any([poly.intersects(anatomical_poly) for anatomical_poly in polygs_anom]):
            filtered_json.append(record)

    print(f"{len(gloms_json) - len(filtered_json)} glomerulus are removed in {img} ")

    create_dir(json_path_out)
    jdump(filtered_json, f"{json_path_out}{img}.json")

