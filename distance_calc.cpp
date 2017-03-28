#include "params.h"


// This computes the distance of all points in d_locations from the given
// longitude and latitude, and returns it in the on-chip global buffer
// dist.
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
