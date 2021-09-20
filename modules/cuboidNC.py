from typing import List, Tuple, Union
from copy import deepcopy

import numpy as np


class Track:
    points: List[np.ndarray]

    def __init__(self, points: List[np.ndarray]):
        self.points = [point.round(3) for point in points]

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
                            vec_fly_in * fly_in_out_length).round(3)
            fly_out_end = (self.points[-1] +
                           vec_fly_out * fly_in_out_length).round(3)

            lines_nc = [f"X{fly_in_start[0]} Y{fly_in_start[1]}\n"] + lines_nc
            lines_nc += [f"X{fly_out_end[0]} Y{fly_out_end[1]}\n"]

        return lines_nc


class Contour(Track):
    pass


class Layer:
    tracks: List[Track]
    contour: Contour

    def __init__(self, tracks: List[Track], contour: Contour, height: float):
        self.tracks = tracks
        self.contour = contour
        self.height = round(height, 3)

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
                 layer_height: float, bidirectional: bool, crosshatching: bool,
                 track_direction: str, hatch_direction: str,
                 contour_opening: str, contour_rotation: str,
                 power_map: List[List[List[float]]] = None):
        """
        Builds a cuboid sample. Origin is located at sample center

        Args:
            shape: X, Y and Z dimensions of sample
            track_spacing: Distance between hatch tracks
            layer_height: Distance between layers
            bidirectional: Create bidirectional hatch tracks
            crosshatching: Alternate hatch direction by 90°
            track_direction: Direction of first track in hatch.
                Possible Values: X+, X-, Y+, Y-
            hatch_direction: Direction of hatch. Must be perpendicular to
                track direction.
                Possible Values: X+, X-, Y+, Y-
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

        def track_division(value, divisor):
            if round(value % divisor, 12) == 0:
                return int(value // divisor - 1)
            else:
                return int(value // divisor)

        if track_direction not in ("X+", "X-", "Y+", "Y-"):
            raise ValueError("Invalid track direction specified")
        if hatch_direction not in ("X+", "X-", "Y+", "Y-"):
            raise ValueError("Invalid hatch direction specified")
        if contour_opening not in ("X+Y+", "X+Y-", "X-Y+", "X-Y-"):
            raise ValueError("Invalid contour opening specified")
        if contour_rotation not in ("CW", "CCW"):
            raise ValueError("Invalid contour rotation specified")
        if track_direction[0] == hatch_direction[0]:
            raise ValueError("Track direction and hatch direction must be "
                             "perpendicular")

        num_tracks_x = track_division(shape[0], track_spacing)
        num_tracks_y = track_division(shape[1], track_spacing)

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

        offset_x = (shape[0] - (num_tracks_x - 1) * track_spacing) * 0.5
        offset_y = (shape[1] - (num_tracks_y - 1) * track_spacing) * 0.5

        track_coords_x = np.linspace(-0.5 * shape[0] + offset_x,
                                     0.5 * shape[0] - offset_x,
                                     num_tracks_x)

        track_points_x = [[np.array((x, edge_y_min)), np.array((x, edge_y_max))]
                          for x in track_coords_x]

        track_coords_y = np.linspace(-0.5 * shape[1] + offset_y,
                                     0.5 * shape[1] - offset_y,
                                     num_tracks_y)

        track_points_y = [[np.array((edge_x_min, y)), np.array((edge_x_max, y))]
                          for y in track_coords_y]

        if hatch_direction[0] == "X":
            track_points_even = track_points_x
            track_points_odd = track_points_y
        else:
            track_points_even = track_points_y
            track_points_odd = track_points_x
        if hatch_direction[1] == "-":
            track_points_even.reverse()
            track_points_odd.reverse()

        if track_direction[1] == "-":
            for track in track_points_even:
                track.reverse()
            for track in track_points_odd:
                track.reverse()

        if bidirectional:
            for track in track_points_even[1::2]:
                track.reverse()
            for track in track_points_odd[1::2]:
                track.reverse()

        tracks_even = [Track(points) for points in track_points_even]
        tracks_odd = [Track(points) for points in track_points_odd]
        contour = Contour(contour_points)

        heights = np.arange(0, shape[2], layer_height)
        if crosshatching:
            self.layers = [Layer(deepcopy(tracks_even), deepcopy(contour), height)
                           if idx % 2 == 0 else
                           Layer(deepcopy(tracks_odd), deepcopy(contour), height)
                           for idx, height in enumerate(heights)]
        else:
            self.layers = [Layer(deepcopy(tracks_even), deepcopy(contour), height)
                           for height in heights]

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
                                     vec_track * spacing * idx).round(3)
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
