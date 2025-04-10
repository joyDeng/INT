# Inverse Neutron Transport Simple Test Code
Before running the code, install the [mitsuba](https://mitsuba.readthedocs.io/en/stable/index.html) and [dr.jit](https://mitsuba.readthedocs.io/en/stable/src/quickstart/drjit_quickstart.html) using
```
pip install mitsuba
```
<!-- simple_bounce.py -->

Ray csg intersection is build upon mitsuba ray scene intersection api. 
Example of creating a csg material can be find in csg.py test_1()

Neutron transport simulation is implemented in area_tally.py.

Shape optimizer without csg is written in optimization.py
Shape optimizer with csg is written in opt_csg.py

Driver: 560.81
