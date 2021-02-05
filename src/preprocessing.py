import os
import json

from shapely import geometry


def filt_glom_by_cortex(img: str, json_path_in: str, json_path_out: str) -> None:
    """ Filter glomerulus by location in Cortex. Load -> Preprocess -> Save.
    """

    # Load
    with open(f'{json_path_in}{img}.json') as json_file:
        gloms = json.load(json_file)
    with open(f'{json_path_in}{img}-anatomical-structure.json') as json_file:
        anot_structure = json.load(json_file)
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
    if not os.path.isdir(json_path_out):
        os.makedirs(json_path_out)
    with open(f"{json_path_out}{img}.json", 'w') as outfile:
        json.dump(gloms_in_cortex, outfile)
