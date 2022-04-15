
Targets = to_metis_s1 to_metis_s2 to_metis_s3 to_metis_grp to_metis_grp4 part show_res eval_res repart

INC_PATH = ./include
LIB_PATH = ./lib
LIB_NAME = metis
C++ = g++
FLAGS = --std=c++17 -O3 -g

all: $(Targets)

$(Targets): %: %.cpp mem.cpp
	$(C++) $(FLAGS) $^ -o $@ -I$(INC_PATH) -L$(LIB_PATH) -l$(LIB_NAME)

.PHONY: clean

clean: 
	rm $(Targets)
