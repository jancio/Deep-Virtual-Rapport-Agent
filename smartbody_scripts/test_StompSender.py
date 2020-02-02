#######################################################################################################################
# Project: Deep Virtual Rapport Agent
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Trial script used to test the communication with the Smartbody. 
#######################################################################################################################


from StompSender import *

char_name = "ChrBrad"
stompSender = StompSender()
stompSender.init_network()


str_text = "hi, my name is "
msg = "<speech>" + str_text + "</speech>"

# bml.execBML('*', '<head type="SHAKE"/>')
msg = "<head type=\"SHAKE\"/>"
msg_xml = stompSender.add_xml(msg)

print("sending: " + msg_xml)
stompSender.send_BML(char_name, msg_xml)

time.sleep(2)

# cmd = "character.getSkeleton().getJointByName(\"Neck\").setPostrotation(SrQuat(-3, -0.0004, -0.0, -0.0));"
# print("sending: " + cmd)
# stompSender.send_SB(cmd)

time.sleep(2)

stompSender.finit_network()

##########################################
# to test, run
# c:\Python27\Scripts>..\python.exe stomp
# subscribe /topic/DEFAULT_SCOPE
# stomp subscribe /topic/DEFAULT_SCOPE
