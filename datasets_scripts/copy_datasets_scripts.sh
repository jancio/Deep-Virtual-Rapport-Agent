#!/bin/bash
#######################################################################################################################
# Project: Deep Virtual Rapport Agent
#
#     Jan Ondras (jo951030@gmail.com)
#     Institute for Creative Technologies, University of Southern California
#     April-October 2019
#
#######################################################################################################################
# [no longer needed]
# Copy all .py, .ipynb, .sh, .conf scripts and .md files from datasets directory to the 
# deep-virtual-rapport-agent repository, preserving the directory structure.
#######################################################################################################################


echo "Copying files ..."

cd ~/datasets
find . -name '*.py' | cpio -pdm ~/deep-virtual-rapport-agent/datasets_scripts
find . -name '*.ipynb' | cpio -pdm ~/deep-virtual-rapport-agent/datasets_scripts
find . -name '*.sh' | cpio -pdm ~/deep-virtual-rapport-agent/datasets_scripts
find . -name '*.conf' | cpio -pdm ~/deep-virtual-rapport-agent/datasets_scripts
find . -name '*.md' | cpio -pdm ~/deep-virtual-rapport-agent/datasets_scripts

echo "DONE"
