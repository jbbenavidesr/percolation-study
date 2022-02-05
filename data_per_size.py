import os

import numpy as np
from faunadb import query as q
from faunadb.client import FaunaClient

from simulation import percolation_density


client = FaunaClient(
    secret=os.environ.get("FAUNA_SECRET"),
)


def get_densities(size, borders=True):
    response = client.query(
        q.map_(
            lambda x: q.get(x),
            q.paginate(q.match(q.index("jigsawData_by_size"), size), size=1000),
        )
    )

    densities = []

    for element in response["data"]:
        sequence = element["data"]["history"]
        perc = percolation_density(sequence, borders=borders)
        densities.append(perc)

    return densities


if __name__ == "__main__":
    sizes = [36, 64, 121, 256, 529]
    borders = True

    for size in sizes:
        densities = get_densities(size, borders)
        np.savetxt(
            # f"data/jigsaw{'_borders' if borders else ''}/size{int(np.sqrt(size)) - 2 * (not borders)}.txt",
            f"data/jigsaw_center/size{int(np.sqrt(size))}.txt",
            densities,
        )
