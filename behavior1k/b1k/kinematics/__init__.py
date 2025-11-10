import os

from .chain import Chain
from .urdf import *

from b1k.paths import r1pro_urdf_path

import pinocchio as pin
from pinocchio import casadi as cpin


def build_chain_from_file(filename: str) -> Chain:
    return build_chain_from_urdf(open(filename).read())


def build_r1pro_chain() -> Chain:
    return build_chain_from_file(str(r1pro_urdf_path.absolute()))


def build_r1pro_casadi_pinocchio_model_and_data():
    urdf = str(r1pro_urdf_path.absolute())
    model = pin.buildModelFromUrdf(urdf)  # assumes fixed base
    cmodel = cpin.Model(model)
    cdata = cmodel.createData()
    return cmodel, cdata
