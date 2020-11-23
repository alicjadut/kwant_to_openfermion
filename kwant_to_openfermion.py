import openfermion

def grid_ix_to_lattice(ix, grid, lat, spinless):
    grid_ix = grid.grid_indices(ix, spinless = spinless)
    lat_ix = lat(grid_ix[0], grid_ix[1])
    return lat_ix

def builder_to_FermionOperator(syst, hoppings_iterator):
    ham = openfermion.FermionOperator()

    #on site terms
    for ix in syst.id_by_site.values():
        ham = ham + openfermion.FermionOperator(f'{ix}^ {ix}', syst.hamiltonian(ix, ix))

    #hopping terms    
    for s1, s2 in hoppings_iterator:
        ix1 = syst.id_by_site[s1]
        ix2 = syst.id_by_site[s2]
        hop = openfermion.FermionOperator(f'{ix1}^ {ix2}', syst.hamiltonian(ix1, ix2))
        ham = ham + hop + openfermion.hermitian_conjugated(hop)
    return(ham)