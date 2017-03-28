/*----------------------------------------------------------------------------
*
* Author:   Liang Ma (liang-ma@polito.it)
*
*----------------------------------------------------------------------------
*/
#define __CL_ENABLE_EXCEPTIONS
#include "ML_cl.h"
#include <cstdlib>
#include <iostream>
#include <fstream>
#include <cmath>
#include <unistd.h>
#include "params.h"
using namespace std;

typedef struct record
{
  char recString[REC_LENGTH];
  float distance;
} Record;

void findLowest(std::vector<Record> &records,vector<float>&distances,vector<int> &minLocations,int topN) {
  int minLoc1;
  Record *tempRec;
  
  for(int i=0;i<topN;i++) {
    minLoc1 = minLocations[i];
   
    // swap locations 
    tempRec = &records[i];
    records[i] = records[minLoc1];
    records[minLoc1] = *tempRec;
    
    // add distance to the min we found
    records[i].distance = distances[i];
  }
}
namespace Params 
{
	int platform = 0;								// -p
	int device = 0;									// -d
	int numRecords = 0;
	char *kernel_name=NULL;     // -k
	char *binary_name=NULL;     // -b
	char *filePath = NULL;
	bool flagf=false, flagb = false;
	void usage(char* name)
	{
		cout<<"Usage: "<<name
			<<" -b opencl_binary_name"
			<<" -k kernel_name"
			<<" -f file path"
			<<" [-p platform]"
			<<" [-d device]"
			<<endl;
	}
	bool parse(int argc, char **argv){
		int opt;
		while((opt=getopt(argc,argv,"n:f:p:d:k:b:"))!=-1){
			switch(opt){
				case 'n':
					numRecords=atoi(optarg);
					break;
				case 'p':
					platform=atoi(optarg);
					break;
				case 'd':
					device=atoi(optarg);
					break;
				case 'k':
					kernel_name=optarg;
					break;
				case 'f':
					filePath=optarg;
					flagf=true;
					break;
				case 'b':
					binary_name=optarg;
					flagb=true;
					break;
				default:
					usage(argv[0]);
					return false;
			}
		}
		return true;
	}
}
int main(int argc, char** argv)
{
	// parse arguments
	if(!Params::parse(argc, argv))
		return EXIT_FAILURE;
 	if(!Params::flagb)
		return EXIT_FAILURE;
	try
	{
		// load binary files
		ifstream ifstr(Params::binary_name);
		const string programString(istreambuf_iterator<char>(ifstr),
				(istreambuf_iterator<char>()));
		
		// create platform, device, program
		vector<cl::Platform> platforms;
		cl::Platform::get(&platforms);
		cl_context_properties properties[] =
		{ CL_CONTEXT_PLATFORM,
			(cl_context_properties)(platforms[Params::platform])(),
			0 };
		cl::Context context(CL_DEVICE_TYPE_ACCELERATOR, properties);
		vector<cl::Device> devices=context.getInfo<CL_CONTEXT_DEVICES>();

		cl::Program::Binaries binaries(1, make_pair(programString.c_str(), programString.length()));
		cl::Program program(context,devices,binaries);
		try
		{
			program.build(devices);
		}
		catch (cl::Error err)
		{
			if (err.err() == CL_BUILD_PROGRAM_FAILURE)
			{
				string info;
				program.getBuildInfo(devices[Params::device],CL_PROGRAM_BUILD_LOG, &info);
				cout << info << endl;
				return EXIT_FAILURE;
			}
			else throw err;
		}

		cl::CommandQueue commandQueue(context, devices[Params::device]);
		
		
		// input and output parameters
		std::vector<Record> h_records(2048);
		std::vector<float2> h_locations(2048);
		int resultsCount=NUM_NEIGHBORS;
		std::vector<float> h_distances(resultsCount);
		vector<int> h_indices(resultsCount);
		bool quiet=0;
		float lat=QUERY_LAT,lng=QUERY_LNG;


		FILE *fp;
		int num=0;
		fp = fopen(Params::filePath, "r");
		if(!fp) {
			perror("error opening the data file\n");
			exit(1);
		}
		// read each record
		while(!feof(fp) && num < Params::numRecords){
			Record record;
			float2 latLong;
			fgets(record.recString,REC_LENGTH,fp);
			fgetc(fp); // newline
			if (feof(fp)) break;

			// parse for lat and long
			char str[REC_LENGTH];
			strncpy(str,record.recString,sizeof(str));
			int year, month, date, hour, num, speed, press;
			float lat, lon;
			char name[REC_LENGTH];
			sscanf(str, "%d %d %d %d %d %s %f %f %d %d", &year, 
					&month, &date, &hour, &num, name, &lat,   &lon, &speed, &press);    
			latLong.x = lat;
			latLong.y = lon;   

			h_locations[num]=latLong;
			h_records[num]=record;
			num++;
		}
		fclose(fp);

		vector<float> h_dist(Params::numRecords);
		if (!quiet) {
			printf("Number of points in reference data set: %d\n",Params::numRecords);
			printf("latitude: %f\n",lat);
			printf("longitude: %f\n",lng);
			printf("Finding the %d closest neighbors.\n",resultsCount);
		}

		// create buffer
		cl::Buffer d_locations = cl::Buffer(context, h_locations.begin(),h_locations.end(),true);
		cl::Buffer d_dist = cl::Buffer(context, CL_MEM_READ_WRITE, sizeof(float)*Params::numRecords);
		cl::Buffer d_distances = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(float)*resultsCount);
		cl::Buffer d_indices = cl::Buffer(context, CL_MEM_WRITE_ONLY, sizeof(int)*resultsCount);
		// create kernel	
		typedef cl::make_kernel<cl::Buffer, float, float, cl::Buffer, int> kernelType;
		kernelType kernelFunctor = kernelType(program, "distance_calc");

		cl::EnqueueArgs enqueueArgs(commandQueue,cl::NDRange(1),cl::NDRange(1));
		cl::Event event = kernelFunctor(enqueueArgs,
				d_locations,
				lat,
				lng,
				d_dist,
				Params::numRecords
				);

		commandQueue.finish();
		event.wait();

		cout<<"calculation of distances are finished."<<endl;
		cl::copy(commandQueue, d_dist, h_dist.begin(), h_dist.end());
		for(vector<float>::iterator it = h_dist.begin();it!=h_dist.end();it++){
//			cout<<"dist["<<it-h_dist.begin()<<"]="<<*it<<endl;
		}
		typedef cl::make_kernel<cl::Buffer,  cl::Buffer, cl::Buffer, int, int> kernelType1;
		kernelType1 kernelFunctor1 = kernelType1(program, "nearestNeighbor");

		cl::EnqueueArgs enqueueArgs1(commandQueue,cl::NDRange(1),cl::NDRange(1));
		cl::Event event1 = kernelFunctor1(enqueueArgs1,
				d_dist,
				d_distances,
				d_indices,
				Params::numRecords,
				resultsCount
				);

		commandQueue.finish();
		event1.wait();

		// load result
		cl::copy(commandQueue, d_distances, h_distances.begin(), h_distances.end());
		cl::copy(commandQueue, d_indices, h_indices.begin(), h_indices.end());

  // find the resultsCount least distances
  findLowest(h_records,h_distances,h_indices,resultsCount);

  // print out results
  if (!quiet)
    for(int i=0;i<resultsCount;i++) {
      printf("%s --> Distance=%f\n",h_records[i].recString,h_records[i].distance);
    }



		int err = 0;
		cout<<"There is/are "<<err<<" error(s)."<<endl;
		if(err!=0)
			return EXIT_FAILURE;
		return EXIT_SUCCESS;
	}
	catch (cl::Error err)
	{
		cerr
			<< "Error:\t"
			<< err.what()
			<< endl;

		return EXIT_FAILURE;
	}
}
