import os

import numpy as np
from faunadb import query as q
from faunadb.client import FaunaClient

from simulation import percolation_density

sizes = [16, 36, 64, 121, 256, 529]

client = FaunaClient(
    secret=os.environ.get("FAUNA_SECRET"),
)

def get_densities(size, borders = True):
    response = client.query(q.map_( lambda x: q.get(x), q.paginate(q.match(q.index("jigsawData_by_size"), size ), size=1000)))

    densities = []

    for element in response['data']:
        sequence = element['data']['history']
        perc = percolation_density(sequence, borders=borders)
        densities.append(perc)

    return densities

for size in sizes:
    densities = get_densities(size, borders=True)
    np.savetxt(f"data_jigsaw_borders/size{int(np.sqrt(size))}.txt", densities)
