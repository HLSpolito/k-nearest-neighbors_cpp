/* 
======================================================
 Copyright 2016 Fahad Bin Muslim
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
       http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
======================================================
*
* Author:   Fahad Bin Muslim (fahad.muslim@polito.it)
*
* This OpenCL code has a kernel to calculate the distances     
* between the query point and all the points in reference data set.
* A second kernel is used to identify the nearest neighbors.  
* 
*----------------------------------------------------------------------------
*/

#include "params.h"


// This computes the resultsCount nearest neighbors, one per work item.
extern "C"	void nearestNeighbor (float*dist, float *d_distances,
		int *indices,
		const float numRecords,
		const float resultsCount
		) {
#pragma HLS interface m_axi port = d_distances bundle = gmem
#pragma HLS interface s_axilite port = d_distances bundle = control
#pragma HLS interface m_axi port = dist bundle = gmem
#pragma HLS interface s_axilite port = dist bundle = control
#pragma HLS interface m_axi port = indices bundle = gmem0
#pragma HLS interface s_axilite port = indices bundle = control

#pragma HLS interface s_axilite port = numRecords bundle = gmem
#pragma HLS interface s_axilite port = resultsCount bundle = gmem
#pragma HLS interface s_axilite port = numRecords bundle = control
#pragma HLS interface s_axilite port = resultsCount bundle = control

#pragma HLS interface s_axilite port = return bundle = control
	float dmin1 = 0; 
	int numR = *(int*)&numRecords;

	int count = *(int*)&resultsCount;
	for(int localId=0;localId < count; localId++) {
		float dist1;
		float dmin = INFINITY;
		int count = 0;

		for (int k = 0; k < numR; k++) {
#pragma HLS PIPELINE
			dist1 = dist[k];
			if (dist1 < dmin && dist1 > dmin1) {
				dmin = dist1;
				count = k;
			}
			if (k == numR-1) {
				dmin1 = dmin;
				indices[localId] = count;
				d_distances[localId] = dmin;
			}
		}
	} 
}
