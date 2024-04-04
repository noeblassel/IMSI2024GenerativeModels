from ase.io import read


def read_ase(filename):
    return read(
        filename=filename, 
        format="lammps-data", 
        Z_of_type={1: 78}, 
        style="atomic"
    )
