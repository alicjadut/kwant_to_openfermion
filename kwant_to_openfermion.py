import openfermion


def builder_to_FermionOperator(syst):
    
    ham = openfermion.FermionOperator()

    #on site terms
    for ix in syst.id_by_site.values():
        ham = ham + openfermion.FermionOperator(f'{ix}^ {ix}', syst.hamiltonian(ix, ix))

    #hopping terms
    for edge in range(syst.graph.num_edges):
        ix1 = syst.graph.head(edge)
        ix2 = syst.graph.tail(edge)
        ham = ham + openfermion.FermionOperator(f'{ix1}^ {ix2}', syst.hamiltonian(ix1, ix2))
        
    return(ham)