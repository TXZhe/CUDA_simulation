
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <cstdlib>
#include <vector>

#define USRNUM 50

__global__ void SIContagionProcessOnGPU(const int *head, const int *len, const int *edges, int *Inum, int N)
{
	int i = threadIdx.x;
	int countI = 1;
	int step = 0;
	int *infected = new int[N];
	for (int i = 0; i < N; i++)
	{
		infected[i] = -1;
	}
	infected[0] = 0;

	Inum[step] = 1;
	while (countI < N)
	{
		step++;
		for (int i = 0; i < N; i++)
		{
			if (infected[i] < step && infected[i]!=-1)
			{
				for (int j = 0; j < len[i]; j++)
				{
					if (infected[edges[head[i] + j]] == -1)
					{
						infected[edges[head[i] + j]] = step;
						countI++;
					}
				}
			}
		}
		Inum[step] = countI;
	}
}

int main()
{
	std::vector< std::vector<int> > h_adjlist;
	h_adjlist.resize(USRNUM);

	printf("Start to read the network...\n");
	FILE *p1;
	p1 = fopen("F:\\private\\17210720150\\cuda_simulation\\data\\WS_50_4_0.3.txt", "r");
	int edge_num = 0;
	while (!feof(p1))
	{
		int a, b;
		fscanf(p1, "%d %d", &a, &b);
		h_adjlist[a].push_back(b);
		h_adjlist[b].push_back(a);
		edge_num += 2;
	}

	// change adj_list into head matrix & len matrix & edges matrix
	int *h_head, *h_len, *h_edges, *h_Inum;
	h_head = (int *)malloc(USRNUM * sizeof(int));
	h_len = (int *)malloc(USRNUM * sizeof(int));
	h_edges = (int *)malloc(edge_num * sizeof(int));
	h_Inum = (int *)malloc(USRNUM * sizeof(int));
	int nownodes = 0;
	for (int i = 0; i < h_adjlist.size(); i++)
	{
		h_head[i] = nownodes;
		h_len[i] = h_adjlist[i].size();
		for (int j = 0; j < h_len[i]; j++)
		{
			h_edges[nownodes] = h_adjlist[i][j];
			nownodes++;
		}
	}

	printf("Bulid network finished\n");

	// set up device
	int dev = 0;
	cudaSetDevice(dev);

	// malloc device global memory
	int *d_head, *d_len, *d_edges, *d_Inum;
	cudaMalloc((int **)&d_head, USRNUM * sizeof(int));
	cudaMalloc((int **)&d_len, USRNUM * sizeof(int));
	cudaMalloc((int **)&d_edges, edge_num * sizeof(int));
	cudaMalloc((int **)&d_Inum, USRNUM * sizeof(int));

	// transfer data from host to device
	cudaMemcpy(d_head, h_head, USRNUM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_len, h_len, USRNUM * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_edges, h_edges, edge_num * sizeof(int), cudaMemcpyHostToDevice);

	//invoke kernel at host side
	dim3 block(1);
	dim3 grid(1);

	SIContagionProcessOnGPU << < grid, block >> > (d_head, d_len, d_edges, d_Inum, USRNUM);

	cudaDeviceSynchronize();

	//copy kernel result back to host
	cudaMemcpy(h_Inum, d_Inum, USRNUM * sizeof(int), cudaMemcpyDeviceToHost);

	for (int i = 0; i < USRNUM; i++)
	{
		if (h_Inum[i] == 0)
		{
			break;
		}
		printf("%d\n", h_Inum[i]);
	}

	//free
	cudaFree(d_head);
	cudaFree(d_len);
	cudaFree(d_edges);
	cudaFree(d_Inum);

	free(h_head);
	free(h_edges);
	free(h_len);
	free(h_Inum);

	system("pause");
	return 0;
}
