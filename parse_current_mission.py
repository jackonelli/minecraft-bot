import os

# below should be your path to the 11.2 LTH world
PATH_TO_LTH_WORLD = os.environ['MINERL_PATH_TO_LTH_WORLD']

# this path will be overwritten
CURRENT_MISSION = "./mission_xmls/currentMission.xml"

# path to the mission file wanted
MISSION_FILE = "./mission_xmls/navigationFixedLTH.xml"

xml_file = open(MISSION_FILE, "r")
xml = xml_file.read()
xml_file.close()

xml = xml.replace("$(MINERL_PATH_TO_LTH_WORLD)", PATH_TO_LTH_WORLD)
write_file = open(CURRENT_MISSION, "w+")
write_file.write(xml)
write_file.close()
