# CMAKE generated file: DO NOT EDIT!
# Generated by "Unix Makefiles" Generator, CMake Version 2.8

#=============================================================================
# Special targets provided by cmake.

# Disable implicit rules so canonical targets will work.
.SUFFIXES:

# Remove some rules from gmake that .SUFFIXES does not remove.
SUFFIXES =

.SUFFIXES: .hpux_make_needs_suffix_list

# Suppress display of executed commands.
$(VERBOSE).SILENT:

# A target that is always out of date.
cmake_force:
.PHONY : cmake_force

#=============================================================================
# Set environment variables for the build.

# The shell in which to execute make rules.
SHELL = /bin/sh

# The CMake executable.
CMAKE_COMMAND = /usr/bin/cmake

# The command to remove a file.
RM = /usr/bin/cmake -E remove -f

# Escaping for special characters.
EQUALS = =

# The top-level source directory on which CMake was run.
CMAKE_SOURCE_DIR = /home/mohits1/Projects/HCRF2.0b/apps/testHCRF

# The top-level build directory on which CMake was run.
CMAKE_BINARY_DIR = /home/mohits1/Projects/HCRF2.0b/apps/testHCRF

# Include any dependencies generated for this target.
include hCRF/uncOptim/CMakeFiles/uncoptim.dir/depend.make

# Include the progress variables for this target.
include hCRF/uncOptim/CMakeFiles/uncoptim.dir/progress.make

# Include the compile flags for this target's objects.
include hCRF/uncOptim/CMakeFiles/uncoptim.dir/flags.make

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o: hCRF/uncOptim/CMakeFiles/uncoptim.dir/flags.make
hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o: /home/mohits1/Projects/HCRF2.0b/libs/3rdParty/uncOptim/src/uncoptim.cpp
	$(CMAKE_COMMAND) -E cmake_progress_report /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/CMakeFiles $(CMAKE_PROGRESS_1)
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Building CXX object hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o"
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && /usr/bin/c++   $(CXX_DEFINES) $(CXX_FLAGS) -o CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o -c /home/mohits1/Projects/HCRF2.0b/libs/3rdParty/uncOptim/src/uncoptim.cpp

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.i: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Preprocessing CXX source to CMakeFiles/uncoptim.dir/src/uncoptim.cpp.i"
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -E /home/mohits1/Projects/HCRF2.0b/libs/3rdParty/uncOptim/src/uncoptim.cpp > CMakeFiles/uncoptim.dir/src/uncoptim.cpp.i

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.s: cmake_force
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --green "Compiling CXX source to assembly CMakeFiles/uncoptim.dir/src/uncoptim.cpp.s"
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && /usr/bin/c++  $(CXX_DEFINES) $(CXX_FLAGS) -S /home/mohits1/Projects/HCRF2.0b/libs/3rdParty/uncOptim/src/uncoptim.cpp -o CMakeFiles/uncoptim.dir/src/uncoptim.cpp.s

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.requires:
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.requires

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.provides: hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.requires
	$(MAKE) -f hCRF/uncOptim/CMakeFiles/uncoptim.dir/build.make hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.provides.build
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.provides

hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.provides.build: hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o

# Object files for target uncoptim
uncoptim_OBJECTS = \
"CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o"

# External object files for target uncoptim
uncoptim_EXTERNAL_OBJECTS =

hCRF/uncOptim/libuncoptim.a: hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o
hCRF/uncOptim/libuncoptim.a: hCRF/uncOptim/CMakeFiles/uncoptim.dir/build.make
hCRF/uncOptim/libuncoptim.a: hCRF/uncOptim/CMakeFiles/uncoptim.dir/link.txt
	@$(CMAKE_COMMAND) -E cmake_echo_color --switch=$(COLOR) --red --bold "Linking CXX static library libuncoptim.a"
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && $(CMAKE_COMMAND) -P CMakeFiles/uncoptim.dir/cmake_clean_target.cmake
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && $(CMAKE_COMMAND) -E cmake_link_script CMakeFiles/uncoptim.dir/link.txt --verbose=$(VERBOSE)

# Rule to build all files generated by this target.
hCRF/uncOptim/CMakeFiles/uncoptim.dir/build: hCRF/uncOptim/libuncoptim.a
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/build

hCRF/uncOptim/CMakeFiles/uncoptim.dir/requires: hCRF/uncOptim/CMakeFiles/uncoptim.dir/src/uncoptim.cpp.o.requires
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/requires

hCRF/uncOptim/CMakeFiles/uncoptim.dir/clean:
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim && $(CMAKE_COMMAND) -P CMakeFiles/uncoptim.dir/cmake_clean.cmake
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/clean

hCRF/uncOptim/CMakeFiles/uncoptim.dir/depend:
	cd /home/mohits1/Projects/HCRF2.0b/apps/testHCRF && $(CMAKE_COMMAND) -E cmake_depends "Unix Makefiles" /home/mohits1/Projects/HCRF2.0b/apps/testHCRF /home/mohits1/Projects/HCRF2.0b/libs/3rdParty/uncOptim /home/mohits1/Projects/HCRF2.0b/apps/testHCRF /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim /home/mohits1/Projects/HCRF2.0b/apps/testHCRF/hCRF/uncOptim/CMakeFiles/uncoptim.dir/DependInfo.cmake --color=$(COLOR)
.PHONY : hCRF/uncOptim/CMakeFiles/uncoptim.dir/depend

