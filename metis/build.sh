#!/bin/bash
g++ --std=c++11 -O3 to_metis.cpp -o to_metis -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_2.cpp -o to_metis_2 -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_3.cpp mem.cpp -o to_metis_3 -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_nodelay.cpp mem.cpp -o to_metis_nodelay -I./include/ -L./lib -lmetis

g++ --std=c++11 -O3 to_metis_s1.cpp mem.cpp -o to_metis_s1 -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_s2.cpp mem.cpp -o to_metis_s2 -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_s3.cpp mem.cpp -o to_metis_s3 -I./include/ -L./lib -lmetis

g++ --std=c++11 -O3 to_metis_grp.cpp mem.cpp -o to_metis_grp -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 to_metis_grp4.cpp mem.cpp -o to_metis_grp4 -I./include/ -L./lib -lmetis

g++ --std=c++11 -O3 part.cpp mem.cpp -o part -I./include/ -L./lib -lmetis

g++ --std=c++11 -O3 show_res.cpp -o show_res -I./include/ -L./lib -lmetis
g++ --std=c++11 -O3 eval_res.cpp -o eval_res -I./include/ -L./lib -lmetis

g++ --std=c++11 -O3 to_metis_all.cpp -o to_metis_all -I./include/ -L./lib -lmetis
