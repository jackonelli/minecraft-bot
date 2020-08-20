# This file uses the environment variable MINERL_PATH_TO_LTH_WORLD
# to create private mission-xmls in the folder my_mission_xmls
import os

INTERESTING_COORDS = {
    "outside_house":                (-272, 64, 10253, -90),
    "outside_close_to_entrance":    (-258, 64, 10238, -90),
    "inside_entrance":              (-250, 64, 10238, -90),
    "bottom_of_first_staircase":    (-240, 64, 10229,   0),
    "top_of_first_staircase":       (-240, 72, 10242,   0),
    "bottom_of_second_staircase":   (-240, 72, 10229,   0),
    "top_of_second_staircase":      (-240, 79, 10242,   0),
    "an_office_entrance":           (-244, 79, 10236,  90),
    "inside_office":                (-249, 79, 10236,  90),
}

def create_mission(start_coords, goal_coords, mission_name="tmpMission", dense=False):
    """
        Creates a mission-xml placed in the mission_xmls folder
        :start_coords: on the form (x, y, z, yaw)
        :goal_coords: (placement of diamond block) on the form (x, y, z)

    """
    if dense:
        template_path = "mission_xmls/templateDense.xml"
    else:
        template_path = "mission_xmls/template.xml"

    xml_file = open(template_path, "r")
    xml = xml_file.read()
    xml_file.close()

    start_str = f'x="{start_coords[0]}" y="{start_coords[1]}" z="{start_coords[2]}" yaw="{start_coords[3]}"'
    goal_str = f'x="{goal_coords[0]}" y="{goal_coords[1]}" z="{goal_coords[2]}"'

    xml = xml.replace("$(START_COORDS)", start_str)
    xml = xml.replace("$(GOAL_COORDS)", goal_str)

    write_path = os.path.join("mission_xmls", mission_name + ".xml")

    write_file = open(write_path, "w+")
    write_file.write(xml)
    write_file.close()
