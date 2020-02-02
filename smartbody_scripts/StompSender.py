#######################################################################################################################
# Project: Deep Virtual Rapport Agent
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# StompSender class to send control messages to SmartBody via ActiveMQ
#
#   Adapted from https://github.com/bernuly/StompSender
#######################################################################################################################


import time
import sys
import urllib
import urllib.parse
import random
import datetime
import stomp

class StompSender():
    headers = {}

    def __init__(self):
        self.headers['ELVISH_SCOPE'] = 'DEFAULT_SCOPE'
        self.headers['MESSAGE_PREFIX'] = 'vrSpeak'
        host_and_ports = [('localhost', 61613)]
        # host_and_ports = [('localhost', 61616)]
        self.conn = stomp.Connection(host_and_ports=host_and_ports)

    def init_network(self):
        self.conn.start()
        self.conn.connect(username='admin', passcode='password', wait=True)
        self.conn.auto_content_length = False

    def finit_network(self):
        self.conn.disconnect()

    def add_xml(self, msg):
        """ convenience function to add xml prefix and suffix to message """
        return "<?xml version=\"1.0\" ?><act><bml>" + msg + "</bml></act>"

    def send_BML(self, char_name, msg):
        """ send BML message. Assumes message is complete xml statement including prefix/suffix """
        # need to prefix with  ALL <message UID>
        str_ID="%03d%s" % (random.randint(0,999), '{:%M%S}'.format(datetime.datetime.now()))
        msg = char_name + " ALL " + str_ID + " " + msg
        self.send_msg("vrSpeak", msg)

    def send_SB(self, msg):
        """ send SmartBody python command """
        self.send_msg("sb", msg)

    def send_msg(self, prefix, msg):
        msg = prefix + " " + urllib.parse.quote_plus(msg)
        #print(msg)
        self.conn.send(body=msg, headers=self.headers, destination='/topic/DEFAULT_SCOPE')
