from component_objects.component_weibull import WeibullComponent
from system_objects.series_system import SeriesSystem
from system_objects.parallel_system import ParallelSystem

import pandas as pd
import numpy as np
from copy import deepcopy

#------------------------------------------------------------
# DEFINE COMPONENTS FROM DATA
#------------------------------------------------------------
def define_comps_from_data():
    # hard coded data from TA-O alternate propulsion report
    data = {
        "Component": [
            "Diesel Engine",
            "Clutch",
            "Reduction Gear",
            "Shaft and Bearings",
            "CRP Propellers",
            "Fuel Oil Motor",
            "Fuel Oil Pump",
            "Fuel Oil Purifier",
            "Lube Oil Motor",
            "Lube Oil Pump",
            "Jacket Water Pump",
            "Fresh Water Pump",
            "F.O. Booster Pump",
        ],
        "MTBF_hrs": [
            8000, 50000, 200000, 200000, 25000,
            7500, 5500, 10000, 7500, 4000,
            27000, 12500, 5500
        ],
        "MTTR_hrs": [
            8, np.nan, np.nan, np.nan, 15,
            18, 4.5, 4, 7.8, 5,
            7.6, 12, 4.5
        ],
        "Reliability": [
            0.999, 0.986, 0.996, 0.996, 0.999,
            0.998, 0.999, 1.000, 0.999, 0.999,
            0.999, 0.999, 0.999
        ],
        "Availability": [
            0.999, 1.000, 1.000, 1.000, 0.999,
            0.997, 0.999, 0.999, 0.998, 0.998,
            0.999, 0.999, 0.999
        ],
    }

    df = pd.DataFrame(data)
    
    # initialize components dictionary    
    components = {}
    for _, row in df.iterrows():
        name = row["Component"]
        components[name] = WeibullComponent(
            name=name,
            MTTF=row["MTBF_hrs"],
            MTTR=row["MTTR_hrs"] if not np.isnan(row["MTTR_hrs"]) else None,
            shape=1.5
        )
        
    return components

#------------------------------------------------------------
# ENGINE SYSTEM
#------------------------------------------------------------
def engine_sys(number=1):
    components = define_comps_from_data()
    
    engine_sys = SeriesSystem(name=f"Engine #{number} System",
                                    components=[deepcopy(components["Lube Oil Motor"]), 
                                                deepcopy(components["Lube Oil Pump"]), 
                                                deepcopy(components["Jacket Water Pump"]), 
                                                deepcopy(components["Fresh Water Pump"]), 
                                                deepcopy(components["F.O. Booster Pump"]), 
                                                deepcopy(components["Fuel Oil Pump"]),
                                                deepcopy(components["Diesel Engine"]),
                                                deepcopy(components["Clutch"])
                                                ])
    return engine_sys

#------------------------------------------------------------
# TRANSMISSION SYSTEM
#------------------------------------------------------------
def transmission_sys():
    components = define_comps_from_data()
    
    transmission_sys= SeriesSystem(name="Transmission System",
                            components=[deepcopy(components["Reduction Gear"]),
                                        deepcopy(components["Shaft and Bearings"]),
                                        deepcopy(components["CRP Propellers"])
                                        ])
    return transmission_sys

#------------------------------------------------------------
# FUEL OIL SYSTEM
#------------------------------------------------------------
def fuel_oil_sys():
    components = define_comps_from_data()
    
    subsys1A = SeriesSystem(name="Fuel Oil Subsystem 1A",
                                components=[deepcopy(components["Fuel Oil Motor"]),
                                            deepcopy(components["Fuel Oil Pump"])
                                            ])
    subsys1B = SeriesSystem(name="Fuel Oil Subsystem 1A",
                                components=[deepcopy(components["Fuel Oil Motor"]),
                                            deepcopy(components["Fuel Oil Pump"])
                                            ])
    
    subsys1 = ParallelSystem(name="Fuel Oil Subsystem 1",
                                components=[subsys1A,
                                            subsys1B])
    
    subsys2 = ParallelSystem(name="Fuel Oil Subsystem 2",
                                components=[deepcopy(components["Fuel Oil Purifier"]),
                                            deepcopy(components["Fuel Oil Purifier"])
                                            ])
    
    fuel_oil_system = SeriesSystem(name="Fuel Oil System",
                                components=[subsys1,
                                            subsys2])
    
    return fuel_oil_system

#------------------------------------------------------------
# OVERALL TAO SYSTEM
#------------------------------------------------------------
def build_TAO():
    """ builds the systems of the example TAO from paper"""
        
    # Build subsystems
    engine_1_sys = engine_sys(1)
    engine_2_sys = engine_sys(2)
    trans_sys = transmission_sys()
    fo_sys = fuel_oil_sys()
    
    # Build overall TAO system
    parallel_engines = ParallelSystem(name="Parallel Engines System",
                                        components=[engine_1_sys,
                                                    engine_2_sys])
    TAO_system = SeriesSystem(name="TAO Overall System",
                                components=[fo_sys, 
                                            parallel_engines,
                                            trans_sys])                                          
    return TAO_system