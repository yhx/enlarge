
#include <stdio.h>
#include <string.h>
#include "../../msg_utils/helper/helper_c.h"
#include "../../src/base/type.h"

using std::string;
using std::to_string;

int main(int argc, char **argv)
{

	if (argc != 3) {
		printf("Usuage: %s file_pop_size file_node_size\n", argv[0]);
		return -1;
	}

	FILE *p_f = fopen_c(argv[1], "r");

	int pop_num = 0;
	fscanf(p_f, "%d", &pop_num);

	int *pop_size = malloc_c<int>(pop_num);

	for (int i=0; i<pop_num; i++) {
		fscanf(p_f, "%d", &(pop_size[i]));
	}
	fclose_c(p_f);

	printf("Population number: %d\n", pop_num);
	printf("Population size: \n");
	size_t sum = 0;
	for (int i=0; i<pop_num; i++) {
		sum += pop_size[i];
		printf("%d ", pop_size[i]);
	}
	printf("\n");
	printf("Total size: %ld\n", sum);

	string path = argv[2];

	int node_num = 0;
	string name = path + "/meta.data";
	FILE *f = fopen_c(name.c_str(), "r");
	fread_c(&(node_num), 1, f);
	fclose_c(f);

	int *node_size = malloc_c<int>(node_num);

	for (int i=0; i<node_num; i++) {
		name = path + "/" + std::to_string(i) + "/main.net";
		size_t n_t_num = 0, s_t_num = 0;
		FILE *f = fopen_c(name.c_str(), "r");
		fread_c(&(n_t_num), 1, f);
		fread_c(&(s_t_num), 1, f);
		Type * n_t = malloc_c<Type>(n_t_num);
		Type * s_t = malloc_c<Type>(s_t_num);
		fread_c(n_t, n_t_num, f);
		fread_c(s_t, s_t_num, f);
		size_t * n_nums = malloc_c<size_t>(n_t_num + 1);
		fread_c(n_nums, n_t_num+1, f);
		node_size[i] = n_nums[n_t_num];
		free_c(n_t);
		free_c(s_t);
		free_c(n_nums);
		fclose_c(f);
	}

	sum = 0;
	printf("Node number: %d\n", node_num);
	printf("Node size: \n");
	for (int i=0; i<node_num; i++) {
		sum += node_size[i];
		printf("%d ", node_size[i]);
	}
	printf("\n");
	printf("Total size: %ld\n", sum);

	size_t pop_idx = 0, node_idx = 0, idx = 0;
	size_t pop_count = 0, node_count = 0;
	printf("Node %ld: ", node_idx);
	for (int i=0; i<pop_num; i++) {
		size_t tmp = pop_count + pop_size[i]/2;
		while (node_count + node_size[node_idx] < tmp) {
			node_count += node_size[node_idx];
			node_idx += 1;
			printf("\n");
			if (node_idx >= node_num ) {
				break;
			}
			printf("Node %ld: ", node_idx);
		}
		if (node_idx >= node_num) {
			printf("\nTotal num: %d\n", i);
			break;
		}
		printf("%ld ", tmp - node_count);
		pop_count += pop_size[i];
	}
	printf("\n");




	return 0;
}


