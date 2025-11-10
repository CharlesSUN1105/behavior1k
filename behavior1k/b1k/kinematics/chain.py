from functools import cached_property
from typing import Dict, Iterator, List, Optional, Union
import random

import casadi as cs
import spatial_casadi as sc

from . import frame, jacobian


class Chain:
    """Chain is a class that represents a kinematic chain."""

    def __init__(self, root_frame: frame.Frame) -> None:
        self._root: Optional[frame.Frame] = root_frame

    def __str__(self) -> str:
        return str(self._root)

    def __iter__(self) -> Iterator[frame.Frame]:
        assert self._root is not None, "Root frame is None"
        yield from self._root.walk()

    @cached_property
    def dof(self):
        return len(self.get_joint_parameter_names())

    @staticmethod
    def _find_frame_recursive(name: str, frame: frame.Frame) -> Optional[frame.Frame]:
        for child in frame.children:
            if child.name == name:
                return child
            ret = Chain._find_frame_recursive(name, child)
            if ret is not None:
                return ret
        return None

    def find_frame(self, name: str) -> Optional[frame.Frame]:
        """Find a frame by name.

        Parameters
        ----------
        name : str
            Frame name.

        Returns
        -------
        Optional[frame.Frame]
            Frame if found, None otherwise.
        """
        assert self._root is not None, "Root frame is None"
        if self._root.name == name:
            return self._root
        return self._find_frame_recursive(name, self._root)

    @staticmethod
    def _find_link_recursive(name: str, frame: frame.Frame) -> Optional[frame.Link]:
        for child in frame.children:
            if child.link.name == name:
                return child.link
            ret = Chain._find_link_recursive(name, child)
            if ret is not None:
                return ret
        return None

    def find_link(self, name: str) -> Optional[frame.Link]:
        """Find a link by name.

        Parameters
        ----------
        name : str
            Link name.

        Returns
        -------
        Optional[frame.Link]
            Link if found, None otherwise.
        """
        assert self._root is not None, "Root frame is None"
        if self._root.link.name == name:
            return self._root.link
        return self._find_link_recursive(name, self._root)

    @staticmethod
    def _get_joint_parameter_names(
        frame: frame.Frame, exclude_fixed: bool = True
    ) -> List[str]:
        joint_names = []
        if not (exclude_fixed and frame.joint.joint_type == "fixed"):
            joint_names.append(frame.joint.name)
        for child in frame.children:
            joint_names.extend(Chain._get_joint_parameter_names(child, exclude_fixed))
        return joint_names

    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        """Get joint parameter names.

        Parameters
        ----------
        exclude_fixed : bool, optional
            Exclude fixed joints, by default True

        Returns
        -------
        List[str]
            Joint parameter names.
        """
        assert self._root is not None, "Root frame is None"
        names = self._get_joint_parameter_names(self._root, exclude_fixed)
        return list(sorted(set(names), key=names.index))

    @staticmethod
    def _get_joint_limits(frame):
        joint_limits = {}
        if frame.joint.joint_type != "fixed":
            joint_limits[frame.joint.name] = frame.joint.limit
        for child in frame.children:
            joint_limits = {**joint_limits, **Chain._get_joint_limits(child)}
        return joint_limits

    def get_joint_limits(self):
        assert self._root is not None, "Root frame is None"
        return self._get_joint_limits(self._root)

    def get_link_names(self):
        """Get the names of all links in the chain.

        Returns
        -------
        List[str]
            A list containing the names of all links in the chain.
        """
        assert self._root is not None, "Root frame is None"
        link_names = []

        def _collect_link_names(frame: frame.Frame):
            link_names.append(frame.link.name)
            for child in frame.children:
                _collect_link_names(child)

        _collect_link_names(self._root)
        return link_names

    def add_frame(self, frame: frame.Frame, parent_name: str) -> None:
        parent_frame = self.find_frame(parent_name)
        if parent_frame is not None:
            parent_frame.add_child(frame)

    @staticmethod
    def _forward_kinematics(
        root: frame.Frame,
        th_dict: Dict[str, float],
        world: Optional[sc.Transformation] = None,
    ) -> Dict[str, sc.Transformation]:
        world = world or sc.Transformation.identity()
        link_transforms = {}
        trans = world * root.get_transform(th_dict.get(root.joint.name, 0.0))
        link_transforms[root.link.name] = trans * root.link.offset
        for child in root.children:
            link_transforms.update(Chain._forward_kinematics(child, th_dict, trans))
        return link_transforms

    def forward_kinematics(
        self,
        th: Union[Dict[str, float]],
        world: Optional[sc.Transformation] = None,
        **kwargs: Dict,
    ) -> Dict[str, sc.Transformation]:
        """Forward kinematics.

        Parameters
        ----------
        th : Union[Dict[str, float]]
            Joint parameters.
        world : Optional[transform.Transform], optional
            World transform, by default None

        Returns
        -------
        Dict[str, transform.Transform]
            Link transforms.
        """
        assert self._root is not None, "Root frame is None"
        world = world or sc.Transformation.identity()
        return self._forward_kinematics(self._root, th, world)

    @staticmethod
    def _visuals_map(root: frame.Frame) -> Dict[str, List[frame.Visual]]:
        vmap = {root.link.name: root.link.visuals}
        for child in root.children:
            vmap.update(Chain._visuals_map(child))
        return vmap

    def visuals_map(self):
        return self._visuals_map(self._root)

    def sample_joint_position(self, feasible=None, max_attempts=100):
        """Returns a random joint configuration within position limits.

        If feasible is given, it should be a method feasible(chain, joint_state) that returns true
        when the joint_position is feasible for some additional criteria, and false otherwise.
        """

        if feasible is None:
            feasible = lambda chain, joint_state: True

        def sample():
            return {
                name: random.uniform(limit.lower, limit.upper)
                for name, limit in self.get_joint_limits().items()
                if limit is not None
            }

        for _ in range(max_attempts):
            position = sample()
            if feasible(self, position):
                break
        else:
            raise RuntimeError("max attempts reached")

        return position

    def get_serial(self, end_link_name, root_link_name=""):
        return SerialChain(
            self,
            end_link_name + "_frame",
            root_frame_name="" if root_link_name == "" else root_link_name + "_frame",
        )


class SerialChain(Chain):
    """SerialChain is a class that represents a serial kinematic chain."""

    def __init__(
        self, chain: Chain, end_frame_name: str, root_frame_name: str = ""
    ) -> None:
        assert chain._root is not None, "Chain root frame is None"
        if root_frame_name == "":
            self._root = chain._root
        else:
            self._root = chain.find_frame(root_frame_name)
            if self._root is None:
                raise ValueError("Invalid root frame name %s." % root_frame_name)
        frames = self._generate_serial_chain_recurse(self._root, end_frame_name)
        if frames is None:
            raise ValueError("Invalid end frame name %s." % end_frame_name)
        self._serial_frames = [self._root] + frames

    @staticmethod
    def _generate_serial_chain_recurse(
        root_frame: frame.Frame, end_frame_name: str
    ) -> Optional[List[frame.Frame]]:
        for child in root_frame.children:
            if child.name == end_frame_name:
                return [child]
            else:
                frames = SerialChain._generate_serial_chain_recurse(
                    child, end_frame_name
                )
                if frames is not None:
                    return [child] + frames
        return None

    def get_joint_parameter_names(self, exclude_fixed: bool = True) -> List[str]:
        assert self._serial_frames is not None, "Serial chain not initialized."
        names = []
        for f in self._serial_frames:
            if exclude_fixed and f.joint.joint_type == "fixed":
                continue
            names.append(f.joint.name)
        return names

    def forward_kinematics(  # type: ignore[override]
        self,
        th: Union[Dict[str, float], List[float]],
        world: Optional[sc.Transformation] = None,
        end_only: bool = True,
    ) -> Union[sc.Transformation, Dict[str, sc.Transformation]]:
        assert self._serial_frames is not None, "Serial chain not initialized."
        if isinstance(th, dict):
            link_transforms = super().forward_kinematics(th, world)
            if end_only:
                return link_transforms[self._serial_frames[-1].link.name]
            else:
                return link_transforms
        world = world or sc.Transformation.identity()
        cnt = 0
        link_transforms = {}
        trans = world
        for f in self._serial_frames:
            if f.joint.joint_type != "fixed":
                trans = trans * f.get_transform(th[cnt])
            else:
                trans = trans * f.get_transform()
            link_transforms[f.link.name] = trans * f.link.offset
            if f.joint.joint_type != "fixed":
                cnt += 1
        return (
            link_transforms[self._serial_frames[-1].link.name]
            if end_only
            else link_transforms
        )

    def jacobian(self, th: List[float], end_only: bool = True):
        assert self._serial_frames is not None, "Serial chain not initialized."

        assert end_only, "end_only=True is only currently supported"

        th_sym = cs.SX.sym("th", self.dof)
        # th_dict = {n: th_sym[i] for i, n in enumerate(self.get_joint_parameter_names())}
        fk = self.forward_kinematics(th_sym)
        p = fk.translation().as_vector()
        r = fk.rotation().as_rotvec()
        J = cs.jacobian(cs.vertcat(p, r), th_sym)
        # print(J)
        # raise ValueError("gotcha")
        J_fun = cs.Function("J", [th_sym], [J])

        J = J_fun(th)

        # print(J.shape)
        # print("---")
        # raise ValueError("gotchat")

        return J

        # if end_only:
        #     return jacobian.calc_jacobian(self, th)
        # else:
        #     jacobians = {}
        #     for serial_frame in self._serial_frames:
        #         jac = jacobian.calc_jacobian_frames(
        #             self, th, link_name=serial_frame.link.name
        #         )
        #         jacobians[serial_frame.link.name] = jac
        #     return jacobians

    def get_joint_limits(self):
        limits = super().get_joint_limits()
        return [limits[name] for name in self.get_joint_parameter_names()]
