import openfermion


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