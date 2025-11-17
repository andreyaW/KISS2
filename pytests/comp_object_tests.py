from component_objects import component_weibull
from component_objects import component_exponential

import pytests

def test_make_one_comp():
    # make an exponential component
    MTTF= 10
    MTTR= 2
    exp_comp = component_exponential(MTTF, MTTR)
    
    assert(exp_comp.MTTF) == MTTF
    assert(exp_comp.MTTR) ==MTTR
    
    # make a weibull component
        
        