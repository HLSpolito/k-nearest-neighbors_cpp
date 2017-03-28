#include "params.h"

/* 
======================================================
 Copyright 2017 Liang Ma
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
* Author:   Liang Ma 
*
* This C++ code is based on the OpenCL code by Fahad Bin Muslim at https://github.com/fahadmuslim/KNN.
* This kernel aims to computes the distance of all points in d_locations from the given
* longitude and latitude, and returns it in the on-chip global buffer dist.
* 
*----------------------------------------------------------------------------
*/

extern "C"
void distance_calc(float2 *d_locations,
                     const float lat,
		     const float lng,
				 float *dist,
				 const float numRecords) {
#pragma HLS DATA_PACK variable=d_locations
#pragma HLS interface m_axi port = d_locations bundle = gmem0
#pragma HLS interface s_axilite port = d_locations bundle = control
#pragma HLS interface m_axi port = dist bundle = gmem
#pragma HLS interface s_axilite port = dist bundle = control

#pragma HLS interface s_axilite port = lat bundle = gmem
#pragma HLS interface s_axilite port = lng bundle = gmem
#pragma HLS interface s_axilite port = numRecords bundle = gmem
#pragma HLS interface s_axilite port = lat bundle = control
#pragma HLS interface s_axilite port = numRecords bundle = control
#pragma HLS interface s_axilite port = lng bundle = control

#pragma HLS interface s_axilite port = return bundle = control
	int count = *(int*)&numRecords;
	for(int globalId = 0; globalId < count; globalId++){
#pragma HLS PIPELINE
		float lat_tmp, lng_tmp, dist_lat, dist_lng;

		// using temporaries for the latitude and longitude
		lat_tmp = d_locations[globalId].x;
		lng_tmp = d_locations[globalId].y;

		dist_lat = lat-lat_tmp;
		dist_lng = lng-lng_tmp;

		//squared euclidean distance calculation
		dist[globalId] = (dist_lat*dist_lat) + (dist_lng*dist_lng);
	}
}
