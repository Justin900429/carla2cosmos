from math import pi, sin, cos
from lxml import etree

from opendriveparser import OpenDrive, parse_opendrive
from opendriveparser.elements.roadLanes import LaneOffset
from opendriveparser.elements.roadPlanView import (
    Line,
    Arc,
    Spiral,
    Poly3,
    ParamPoly3,
)
from opendriveparser.elements.roadLanes import Lane
from opendriveparser.elements.roadLanes import LaneWidth, LaneSection
from opendriveparser.elements.road import Road


Geometry = Line | Arc | Spiral | Poly3 | ParamPoly3


class LaneOffsetCalculate:
    def __init__(self, lane_offsets: list[LaneOffset]):
        lane_offsets = list(sorted(lane_offsets, key=lambda x: x.sPos))
        lane_offsets_dict = dict()
        for lane_offset in lane_offsets:
            a = lane_offset.a
            b = lane_offset.b
            c = lane_offset.c
            d = lane_offset.d
            s_start = lane_offset.sPos
            lane_offsets_dict[s_start] = (a, b, c, d)
        self.lane_offsets_dict = lane_offsets_dict

    def calculate_offset(self, s: float) -> float:
        for s_start, (a, b, c, d) in reversed(self.lane_offsets_dict.items()):  # e.g. 75, 25
            if s >= s_start:
                ds = s - s_start
                offset = a + b * ds + c * ds**2 + d * ds**3
                return offset
        return 0


def load_xodr_and_parse(xodr_file: str) -> OpenDrive:
    with open(xodr_file, "r") as f:
        parser = etree.XMLParser()
        root_node = etree.parse(f, parser).getroot()
        road_network = parse_opendrive(root_node)
    return road_network


def calculate_reference_points_of_one_geometry(
    geometry: Geometry, length: float, step: float = 0.01
) -> list[dict[str, any]]:
    nums = int(length / step)
    res = []
    for i in range(nums):
        s_ = step * i
        pos_, tangent_ = geometry.calcPosition(s_)
        x, y = pos_
        one_point = {
            "position": (x, y),  # The location of the reference point
            "tangent": tangent_,  # Orientation of the reference point
            "s_geometry": s_,  # The distance between the start point of the geometry and current point along the reference line
        }
        res.append(one_point)
    return res


def get_geometry_length(geometry: Geometry) -> float:
    if hasattr(geometry, "length"):
        length = geometry.length
    elif hasattr(geometry, "_length"):
        length = geometry._length  # Some geometry has the attribute "_length".
    else:
        raise AttributeError("No attribute length found!!!")
    return length


def get_all_reference_points_of_one_road(
    geometries: list[Geometry], step: float = 0.01
) -> list[dict[str, any]]:
    reference_points = []
    s_start_road = 0
    for geometry_id, geometry in enumerate(geometries):
        geometry_length = get_geometry_length(geometry)

        # Calculate all the reference points of current geometry.
        pos_tangent_s_list = calculate_reference_points_of_one_geometry(geometry, geometry_length, step=step)

        # As for every reference points, add the distance start by road and its geometry index.
        pos_tangent_s_s_list = [
            {**point, "s_road": point["s_geometry"] + s_start_road, "index_geometry": geometry_id}
            for point in pos_tangent_s_list
        ]
        reference_points.extend(pos_tangent_s_s_list)

        s_start_road += geometry_length
    return reference_points


def get_width(widths: list[LaneWidth], s: float) -> float:
    assert isinstance(widths, list), TypeError(type(widths))
    widths.sort(key=lambda x: x.sOffset)
    current_width = None
    # EPS = 1e-5
    milestones = [width.sOffset for width in widths] + [float("inf")]

    control_mini_section = [
        (start, end) for (start, end) in zip(milestones[:-1], milestones[1:], strict=True)
    ]
    for width, start_end in zip(widths, control_mini_section, strict=True):
        start, end = start_end
        if start <= s < end:
            ds = s - width.sOffset
            current_width = width.a + width.b * ds + width.c * ds**2 + width.d * ds**3
    return current_width


def get_lane_offset(lane_offsets: list[LaneOffset], section_s: float, length: float = float("inf")) -> float:
    assert isinstance(lane_offsets, list), TypeError(type(lane_offsets))
    if not lane_offsets:
        return 0
    lane_offsets.sort(key=lambda x: x.sPos)
    current_offset = 0
    EPS = 1e-5
    milestones = [lane_offset.sPos for lane_offset in lane_offsets] + [length + EPS]

    control_mini_section = [
        (start, end) for (start, end) in zip(milestones[:-1], milestones[1:], strict=True)
    ]
    for offset_params, start_end in zip(lane_offsets, control_mini_section, strict=True):
        start, end = start_end
        if start <= section_s < end:
            ds = section_s - offset_params.sPos
            current_offset = (
                offset_params.a + offset_params.b * ds + offset_params.c * ds**2 + offset_params.d * ds**3
            )
    return current_offset


def calculate_area_of_one_left_lane(
    left_lane: Lane, points: list[dict[str, any]], most_left_points: list[tuple[float, float]]
) -> tuple[dict[str, any], list[tuple[float, float]]]:
    inner_points = most_left_points[:]

    widths = left_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points, strict=True):
        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_left = tangent + pi / 2
        x_inner, y_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_left) * lane_width_offset
        y_outer = y_inner + sin(normal_left) * lane_width_offset

        update_points.append((x_outer, y_outer))

    outer_points = update_points[:]
    most_left_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_left_points


def calculate_area_of_one_right_lane(
    right_lane: Lane, points: list[dict[str, any]], most_right_points: list[tuple[float, float]]
) -> tuple[dict[str, any], list[tuple[float, float]]]:
    inner_points = most_right_points[:]

    widths = right_lane.widths
    update_points = []
    for reference_point, inner_point in zip(points, inner_points, strict=True):
        tangent = reference_point["tangent"]
        s_lane_section = reference_point["s_lane_section"]
        lane_width = get_width(widths, s_lane_section)

        normal_eight = tangent - pi / 2
        x_inner, y_inner = inner_point

        lane_width_offset = lane_width

        x_outer = x_inner + cos(normal_eight) * lane_width_offset
        y_outer = y_inner + sin(normal_eight) * lane_width_offset

        update_points.append((x_outer, y_outer))

    outer_points = update_points[:]
    most_right_points = outer_points[:]

    current_ara = {
        "inner": inner_points,
        "outer": outer_points,
    }
    return current_ara, most_right_points


def calculate_lane_area_within_one_lane_section(
    lane_section: LaneSection, points: list[dict[str, any]]
) -> tuple[dict[str, any], dict[str, any], list[tuple[float, float]], list[tuple[float, float]]]:
    all_lanes = lane_section.allLanes

    # Process the lane indexes.
    left_lanes = [lane for lane in all_lanes if int(lane.id) > 0]
    right_lanes = [lane for lane in all_lanes if int(lane.id) < 0]
    left_lanes.sort(key=lambda x: x.id)
    right_lanes.sort(reverse=True, key=lambda x: x.id)

    # Get the lane area of left lanes and the most left lane line.
    left_lanes_area = dict()
    most_left_points = [point["position_center_lane"] for point in points][:]
    for left_lane in left_lanes:
        current_area, most_left_points = calculate_area_of_one_left_lane(left_lane, points, most_left_points)
        left_lanes_area[left_lane.id] = current_area

    # Get the lane area of right lanes and the most right lane line.
    right_lanes_area = dict()
    most_right_points = [point["position_center_lane"] for point in points][:]
    for right_lane in right_lanes:
        current_area, most_right_points = calculate_area_of_one_right_lane(
            right_lane, points, most_right_points
        )
        right_lanes_area[right_lane.id] = current_area

    return left_lanes_area, right_lanes_area, most_left_points, most_right_points


def calculate_points_of_reference_line_of_one_section(points: list[dict[str, any]]) -> list[dict[str, any]]:
    res = []
    for point in points:
        tangent = point["tangent"]
        x, y = point["position"]  # Points on reference line.
        normal = tangent + pi / 2
        lane_offset = point["lane_offset"]  # Offset of center lane.

        x += cos(normal) * lane_offset
        y += sin(normal) * lane_offset

        point = {
            **point,
            "position_center_lane": (x, y),
        }
        res.append(point)
    return res


def calculate_s_lane_section(
    reference_points: list[dict[str, any]], lane_sections: list[LaneSection]
) -> list[dict[str, any]]:
    res = []
    for point in reference_points:
        for lane_section in reversed(lane_sections):
            if point["s_road"] >= lane_section.sPos:
                res.append(
                    {
                        **point,
                        "s_lane_section": point["s_road"] - lane_section.sPos,
                        "index_lane_section": lane_section.idx,
                    }
                )
                break
    return res


def uncompress_dict_list(dict_list: list[dict[str, any]]) -> dict[str, any]:
    assert isinstance(dict_list, list), TypeError("Keys")
    if not dict_list:
        return dict()

    keys = set(dict_list[0].keys())
    for dct in dict_list:
        cur = set(dct.keys())
        assert keys == cur, "Inconsistency of dict keys! {} {}".format(keys, cur)

    res = dict()
    for sample in dict_list:
        for k, v in sample.items():
            if k not in res:
                res[k] = [v]
            else:
                res[k].append(v)

    keys = list(sorted(list(keys)))
    res = {k: res[k] for k in keys}
    return res


def get_lane_line(section_data: dict[str, any]) -> dict[str, any]:
    left_lanes_area = section_data["left_lanes_area"]
    right_lanes_area = section_data["right_lanes_area"]

    lane_line_left = dict()
    if left_lanes_area:
        indexes = list(left_lanes_area.keys())
        for index_inner, index_outer in zip(indexes, indexes[1:] + ["NAN"], strict=True):
            lane_line_left[(index_inner, index_outer)] = left_lanes_area[index_inner]["outer"]

    lane_line_right = dict()
    if right_lanes_area:
        indexes = list(right_lanes_area.keys())
        for index_inner, index_outer in zip(indexes, indexes[1:] + ["NAN"], strict=True):
            lane_line_right[(index_inner, index_outer)] = right_lanes_area[index_inner]["outer"]

    return {"lane_line_left": lane_line_left, "lane_line_right": lane_line_right}


def get_lane_area_of_one_road(road: Road, step: float = 0.01) -> dict[str, any]:
    geometries = road.planView._geometries
    # Lane offset is the offset between center lane (width is 0) and the reference line.
    lane_offsets = road.lanes.laneOffsets
    lane_offset_calculate = LaneOffsetCalculate(lane_offsets=lane_offsets)
    lane_sections = road.lanes.laneSections
    lane_sections = list(
        sorted(lane_sections, key=lambda x: x.sPos)
    )  # Sort the lane sections by start position.

    reference_points = get_all_reference_points_of_one_road(
        geometries, step=step
    )  # Extract the reference points.

    # Calculate the offsets of center lane.
    reference_points = [
        {**point, "lane_offset": lane_offset_calculate.calculate_offset(point["s_road"])}
        for point in reference_points
    ]

    # Calculate the points of center lane based on reference points and offsets.
    reference_points = calculate_points_of_reference_line_of_one_section(reference_points)

    # Calculate the distance of each point starting from the current section along the direction of the reference line.
    reference_points = calculate_s_lane_section(reference_points, lane_sections)

    total_areas = dict()
    for lane_section in lane_sections:
        section_start = lane_section.sPos  # Start position of the section in current road.
        section_end = lane_section.sPos + lane_section.length  # End position of the section in current road.

        # Filter out the points belonging to current lane section.
        current_reference_points = list(
            filter(lambda x: section_start <= x["s_road"] < section_end, reference_points)
        )

        # Calculate the boundary point of every lane in current lane section.
        area = calculate_lane_area_within_one_lane_section(lane_section, current_reference_points)
        left_lanes_area, right_lanes_area, most_left_points, most_right_points = area

        # Extract types and indexes.
        types = {lane.id: lane.type for lane in lane_section.allLanes if lane.id != 0}
        index = (road.id, lane_section.idx)

        # Convert dict list to list dict of the reference points information.
        uncompressed_lane_section_data = uncompress_dict_list(current_reference_points)

        # Integrate all the information of current lane section of current road.
        section_data = {
            "left_lanes_area": left_lanes_area,
            "right_lanes_area": right_lanes_area,
            "most_left_points": most_left_points,
            "most_right_points": most_right_points,
            "types": types,
            "reference_points": uncompressed_lane_section_data,
        }

        # Get all lane lines with their left and right lanes.
        lane_line = get_lane_line(section_data)
        section_data.update(lane_line)

        total_areas[index] = section_data

    return total_areas


def get_all_lanes(
    road_network: OpenDrive, step: float = 0.1, ignore_junction: bool = False
) -> dict[str, any]:
    roads = road_network.roads
    total_areas_all_roads = dict()

    for road in roads:
        if ignore_junction and road.junction is not None:
            continue
        lanes_of_one_road = get_lane_area_of_one_road(road, step=step)
        total_areas_all_roads = {**total_areas_all_roads, **lanes_of_one_road}

    return total_areas_all_roads
