# This file uses the environment variable MINERL_PATH_TO_LTH_WORLD
# to create private mission-xmls in the folder my_mission_xmls
import os
import glob

# below should be your path to the 11.2 LTH world
PATH_TO_LTH_WORLD = os.environ['MINERL_PATH_TO_LTH_WORLD']

if not os.path.isdir("my_mission_xmls"):
    os.mkdir("my_mission_xmls")

for xml_path in glob.glob("mission_xmls/*.xml"):
    xml_file = open(xml_path, "r")
    xml = xml_file.read()
    xml_file.close()

    xml = xml.replace("$(MINERL_PATH_TO_LTH_WORLD)", PATH_TO_LTH_WORLD)
    write_file = open("my_" + xml_path, "w+")
    write_file.write(xml)
    write_file.close()
