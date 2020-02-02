#######################################################################################################################
# Project: Deep Virtual Rapport Agent (rapport model)
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# Dual logger to print outputs to both the terminal and a log file.
#######################################################################################################################


import logging
import sys


class DualLogger(object):
	"""
	Dual logger to print outputs to both the terminal and a log file. 

	Handles stdout as well as stderr streams.
	"""

	def __init__(self, name=None):
		if name == 'stdout':
			self.terminal = sys.stdout
			self.level = logging.INFO
		elif name == 'stderr':
			self.terminal = sys.stderr
			self.level = logging.ERROR
		self.logger = logging.getLogger(name)

	def write(self, msg):
		self.terminal.write(msg)
		self.logger.log(self.level, msg)

	def flush(self):
		self.terminal.flush()
		for handler in self.logger.handlers:
			handler.flush()
