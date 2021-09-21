from typing import List, Tuple, Union
from copy import deepcopy

import numpy as np

rounding = 3


class Track:
    points: List[np.ndarray]

    def __init__(self, points: List[np.ndarray]):
        self.points = [point.round(rounding) for point in points]

    def get_nc(self, power: float, power_map_track: Union[List[float], None],
               fly_in_out: bool, fly_in_out_length: float) -> List[str]:

        if power_map_track is None:
            lines_nc = [f"X{self.points[0][0]} Y{self.points[0][1]}\n",
                        f"DO $A_OUTA[6]={power}*10000/5000\n"]
            lines_nc += [f"X{point[0]} Y{point[1]}\n"
                         for point in self.points[1:]]
            lines_nc += [f"DO $A_OUTA[6]=0\n"]
        else:
            lines_nc = []
            power_map_track += [0]
            for idx, point in enumerate(self.points):
                lines_nc += [f"X{point[0]} Y{point[1]}\n",
                             f"DO $A_OUTA[6]={power_map_track[idx]}*10000/5000\n"]

        if fly_in_out:
            vec_fly_in = self.points[0] - self.points[1]
            vec_fly_in = vec_fly_in / np.linalg.norm(vec_fly_in)
            vec_fly_out = self.points[-1] - self.points[-2]
            vec_fly_out = vec_fly_out / np.linalg.norm(vec_fly_out)

            fly_in_start = (self.points[0] +
                            vec_fly_in * fly_in_out_length).round(rounding)
            fly_out_end = (self.points[-1] +
                           vec_fly_out * fly_in_out_length).round(rounding)

            lines_nc = [f"X{fly_in_start[0]} Y{fly_in_start[1]}\n"] + lines_nc
            lines_nc += [f"X{fly_out_end[0]} Y{fly_out_end[1]}\n"]

        return lines_nc


class Contour(Track):
    pass


class Layer:
    tracks: List[Track]
    contour: Contour
    height: float

    def __init__(self, shape: Tuple[float, float], track_spacing: float,
                 bidirectional: bool, track_direction: str, hatch_direction: str,
                 contour_opening: str, contour_rotation: str, height: float):

        def track_division(value, divisor):
            if round(value % divisor, 12) == 0:
                return int(value // divisor - 1)
            else:
                return int(value // divisor)

        self.height = round(height, rounding)

        if track_direction[0] == "X":
            hatch_length = shape[1]
        else:
            hatch_length = shape[0]

        num_tracks = track_division(hatch_length, track_spacing)

        edge_x_max = shape[0] * 0.5
        edge_x_min = -edge_x_max
        edge_y_max = shape[1] * 0.5
        edge_y_min = -edge_y_max

        contour_points = [np.array((edge_x_max, edge_y_max)),
                          np.array((edge_x_max, edge_y_min)),
                          np.array((edge_x_min, edge_y_min)),
                          np.array((edge_x_min, edge_y_max))]

        if contour_opening == "X+Y-":
            contour_points = contour_points[1:] + contour_points[:1]
        elif contour_opening == "X-Y-":
            contour_points = contour_points[2:] + contour_points[:2]
        elif contour_opening == "X-Y+":
            contour_points = contour_points[3:] + contour_points[:3]

        if contour_rotation == "CCW":
            contour_points.reverse()
            contour_points = contour_points[3:] + contour_points[:3]

        contour_points.append(contour_points[0])

        offset = (hatch_length - (num_tracks - 1) * track_spacing) * 0.5

        track_coords = np.linspace(-0.5 * hatch_length + offset,
                                   0.5 * hatch_length - offset,
                                   num_tracks)

        if hatch_direction[0] == "X":
            track_points = [[np.array((coord, edge_y_min)),
                             np.array((coord, edge_y_max))]
                            for coord in track_coords]
        else:
            track_points = [[np.array((edge_x_min, coord)),
                             np.array((edge_x_max, coord))]
                            for coord in track_coords]

        if hatch_direction[1] == "-":
            track_points.reverse()

        if track_direction[1] == "-":
            for track in track_points:
                track.reverse()

        if bidirectional:
            for track in track_points[1::2]:
                track.reverse()

        self.tracks = [Track(points) for points in track_points]
        self.contour = Contour(contour_points)

    def get_nc(self, power: float,
               power_map_layer: Union[List[List[float]], None],
               contour_first: bool, fly_in_out: bool,
               fly_in_out_length: float) -> List[str]:

        tracks_nc = []
        power_map_track = None
        for idx, track in enumerate(self.tracks):
            if power_map_layer is not None:
                power_map_track = power_map_layer[idx]
            tracks_nc += track.get_nc(power=power,
                                      power_map_track=power_map_track,
                                      fly_in_out=fly_in_out,
                                      fly_in_out_length=fly_in_out_length)

        contour_nc = self.contour.get_nc(power=power,
                                         power_map_track=None,
                                         fly_in_out=fly_in_out,
                                         fly_in_out_length=fly_in_out_length)

        layer_nc = [f"Z{self.height}\n", "\n"]
        if contour_first:
            layer_nc += contour_nc + ["\n"] + tracks_nc
        else:
            layer_nc += tracks_nc + ["\n"] + contour_nc

        return layer_nc


class Cuboid:
    layers: List[Layer]
    _power_map: List[List[List[float]]]

    def __init__(self, shape: Tuple[float, float, float], track_spacing: float,
                 layer_height: float, bidirectional: bool = True,
                 track_direction: str = "X+", hatch_direction: str = "Y+",
                 crosshatching: bool = False, crosshatching_rotation: str = "CCW",
                 contour_opening: str = "X-Y+", contour_rotation: str = "CCW",
                 power_map: List[List[List[float]]] = None):
        """
        Builds a cuboid sample. Origin is located at sample center

        Args:
            shape: X, Y and Z dimensions of sample
            track_spacing: Distance between hatch tracks
            layer_height: Distance between layers
            bidirectional: Create bidirectional hatch tracks
            track_direction: Direction of first track in hatch.
                Possible Values: X+, X-, Y+, Y-
            hatch_direction: Direction of hatch. Must be perpendicular to
                track direction.
                Possible Values: X+, X-, Y+, Y-
            crosshatching: Alternate hatch direction
            crosshatching_rotation:
            contour_opening: Corner position of contour opening.
                Possible Values: X+Y+, X+Y-, X-Y+, X-Y-
            contour_rotation: Direction of contour rotation.
                Possible Values: CW, CCW
            power_map: Nested list specifying power values. Must be list
                containing lists for each layer containing lists of power
                values for every track in layer. Power values will be applied
                evenly spaced along each track

        Raises:
            ValueError:
                If track_direction, hatch_direction, contour_opening or
                    contour_rotation specifiers are invalid.
                If track_direction and hatch_direction are not
                    perpendicular.
                If power_map has invalid dimensions.
        """

        if track_direction not in ("X+", "X-", "Y+", "Y-"):
            raise ValueError("Invalid track direction specified")
        if hatch_direction not in ("X+", "X-", "Y+", "Y-"):
            raise ValueError("Invalid hatch direction specified")
        if crosshatching_rotation not in ("CW", "CCW"):
            raise ValueError("Invalid crosshatching rotation specified")
        if contour_opening not in ("X+Y+", "X+Y-", "X-Y+", "X-Y-"):
            raise ValueError("Invalid contour opening specified")
        if contour_rotation not in ("CW", "CCW"):
            raise ValueError("Invalid contour rotation specified")
        if track_direction[0] == hatch_direction[0]:
            raise ValueError("Track direction and hatch direction must be "
                             "perpendicular")

        heights = np.arange(0, shape[2], layer_height)

        if crosshatching:
            directions = ("X+", "Y+", "X-", "Y-")
            track_dir_idx = directions.index(track_direction)
            hatch_dir_idx = directions.index(hatch_direction)

            corners = ("X+Y+", "X-Y+", "X-Y-", "X+Y-")
            opening_idx = corners.index(contour_opening)

            if crosshatching_rotation == "CW":
                track_dir_cross = directions[track_dir_idx - 1]
                hatch_dir_cross = directions[hatch_dir_idx - 1]
                opening_cross = corners[opening_idx - 1]
            else:
                track_dir_cross = directions[(track_dir_idx + 1) % 4]
                hatch_dir_cross = directions[(hatch_dir_idx + 1) % 4]
                opening_cross = corners[(opening_idx + 1) % 4]

            layer_vals = [(track_direction, hatch_direction,
                           contour_opening, height)
                          if idx % 2 == 0 else
                          (track_dir_cross, hatch_dir_cross,
                           opening_cross, height)
                          for idx, height in enumerate(heights)]
        else:
            layer_vals = [(track_direction, hatch_direction,
                           contour_opening, height) for height in heights]

        self.layers = [Layer(shape=shape[0:2],
                             track_spacing=track_spacing,
                             bidirectional=bidirectional,
                             track_direction=track_dir,
                             hatch_direction=hatch_dir,
                             contour_opening=opening,
                             contour_rotation=contour_rotation,
                             height=height)
                       for track_dir, hatch_dir, opening, height in layer_vals]

        self._power_map = power_map
        if self._power_map is not None:
            if len(power_map) != len(self.layers):
                raise ValueError(
                    f"Power map has invalid layer count "
                    f"({len(power_map)}, must be {len(self.layers)})"
                )
            for layer_idx, layer in enumerate(self.layers):
                if len(power_map[layer_idx]) != len(layer.tracks):
                    raise ValueError(
                        f"Power map layer {layer_idx} has invalid track count "
                        f"({len(power_map[layer_idx])}, must be {len(layer.tracks)})"
                    )

            for layer_idx, layer in enumerate(self.layers):
                for track_idx, track in enumerate(layer.tracks):
                    map_track = power_map[layer_idx][track_idx]
                    vec_track = track.points[1] - track.points[0]
                    spacing = np.linalg.norm(vec_track) / len(map_track)
                    vec_track = vec_track / np.linalg.norm(vec_track)
                    track_start = track.points[0]
                    track.points = [(track_start +
                                     vec_track * spacing * idx).round(rounding)
                                    for idx in range(len(map_track) + 1)]

    def generate_nc(self, power: float, contour_first: bool = True,
                    save_layers: bool = True, save_time: float = 0.5,
                    fly_in_out: bool = False, fly_in_out_length: float = 5,
                    path_out: str = None, path_header: str = None,
                    path_footer: str = None) -> List[str]:
        """
        Generates nc code of cuboid sample

        Args:
            power: Laser power. Applies only to contour if power map is used
                during initialization
            contour_first: Weld contour before hatch
            save_layers: Turn DATATRIGGER off and on after each layer
            save_time: Dwell time between DATATRIGGER of and on in seconds
            fly_in_out: Linearly extend track beginnings and ends
            fly_in_out_length: Fly in/out length in mm
            path_out: Output file path. If None, no output is written
            path_header: Header file path. If None, no header is included
            path_footer: Footer file path. If None, no footer is included

        Returns:
            List of strings containing nc code
        """

        cuboid_nc = []
        power_map_layer = None
        for idx, layer in enumerate(self.layers):
            if self._power_map is not None:
                power_map_layer = self._power_map[idx]
            cuboid_nc += layer.get_nc(power=power,
                                      power_map_layer=power_map_layer,
                                      contour_first=contour_first,
                                      fly_in_out=fly_in_out,
                                      fly_in_out_length=fly_in_out_length)

            if save_layers and idx != len(self.layers) - 1:
                cuboid_nc += [f"\n",
                              f"DATATRIGGER = 0\n",
                              f"G4 F{save_time}\n",
                              f"DATATRIGGER = 1\n",
                              f"\n"]

        if path_header is not None:
            with open(path_header, "r") as header_file:
                header = header_file.readlines()
                cuboid_nc = header + ["\n", "\n"] + cuboid_nc

        if path_footer is not None:
            with open(path_footer, "r") as footer_file:
                footer = footer_file.readlines()
                cuboid_nc += ["\n", "\n"] + footer

        if path_out is not None:
            with open(path_out, "w") as out_file:
                out_file.writelines(cuboid_nc)

        return cuboid_nc
