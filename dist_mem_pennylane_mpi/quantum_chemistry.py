import covalent as ct

from subprocess import Popen
from pymatgen.core import Structure
from pymatgen.core.surface import generate_all_slabs


sts = Popen("covalent start", shell=True).wait()

@ct.electron
def relax_structure(structure, relax_cell=True):
    return structure.relax(relax_cell=relax_cell)

@ct.electron
def carve_slabs(structure, max_index=1, min_slab_size=10.0, min_vacuum_size=10.0):
    slabs = generate_all_slabs(
        structure,
        max_index,
        min_slab_size,
        min_vacuum_size,
    )
    return slabs    

@ct.electron
@ct.lattice
def relax_slabs(slabs):
    return [relax_structure(slab, relax_cell=False) for slab in slabs]

@ct.lattice(executor="local")
def workflow(structure):
    relaxed_structure = relax_structure(structure)
    slabs = carve_slabs(relaxed_structure)
    relaxed_slabs = relax_slabs(slabs)
    return relaxed_slabs

structure = Structure(
    lattice=[[0, 2.13, 2.13], [2.13, 0, 2.13], [2.13, 2.13, 0]],
    species=["Mg", "O"],
    coords=[[0, 0, 0], [0.5, 0.5, 0.5]],
)            


dispatch_id = ct.dispatch(workflow)(structure)
results = ct.get_result(dispatch_id, wait=True)
