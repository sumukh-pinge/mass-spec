#include "host.h"

void datasetBinaryRead(vector<int> &data, string path){
	ifstream file_(path, ios::in | ios::binary);
	int32_t size;
	file_.read((char*)&size, sizeof(size));
	int32_t temp;
	for(int i = 0; i < size; i++){
		file_.read((char*)&temp, sizeof(temp));
		data.push_back(temp);
	}
	file_.close();
}

int main(int argc, char** argv)
{
    if (argc != 2) {
        cout << "Usage: " << argv[0] << " <XCLBIN File>" << std::endl;
		return EXIT_FAILURE;
	}
    string binaryFile = argv[1];
    cl_int err;
    unsigned fileBufSize;

	auto t_start = chrono::high_resolution_clock::now();
   
	vector<int> X_data;
	vector<int> y_data;
	datasetBinaryRead(X_data, X_data_path);
	datasetBinaryRead(y_data, y_data_path);
	
		//Shuffle the data (and labels)
		vector<int> X_data_shuffled(X_data.size());
		vector<int> y_data_shuffled(y_data.size());
		int shuffle_arr[y_data.size()];
		srand (time(NULL));

	if(shuffled){
	
		for(int i = 0; i < y_data.size(); i++)
			shuffle_arr[i] = i;
		//shuffle
		for(int i = y_data.size()-1; i != 0; i--){
			int j = rand()%i;
			int temp = shuffle_arr[i];
			shuffle_arr[i] = shuffle_arr[j];
			shuffle_arr[j] = temp;
		}

		for(int i = 0; i < y_data.size(); i++){
			y_data_shuffled[i] = y_data[shuffle_arr[i]];
			for(int j = 0; j < N_FEAT; j++){
				X_data_shuffled[i*N_FEAT + j] = X_data[shuffle_arr[i]*N_FEAT + j];
			}
		}
	}	
	
	int N_SAMPLE = y_data_shuffled.size();
	int input_int = X_data_shuffled.size();
	 
	vector<int, aligned_allocator<int>> input_gmem(input_int);
	for(int i = 0; i < input_int; i++){
		if(shuffled)
			input_gmem[i] = X_data_shuffled[i];
		else
			input_gmem[i] = X_data[i];
	}
	
	vector<int, aligned_allocator<int>> labels_gmem(N_SAMPLE);

	//We need a seed ID. To generate in a random yet determenistic (for later debug purposes) fashion, we use bits of log2 as some random stuff.
	vector<int, aligned_allocator<int>> ID_gmem(Dhv/32);
	srand(time(NULL));
	for(int i = 0; i < Dhv/32; i++){
		//long double temp = log2(i+2.5) * pow(2, 31);
		//long long int temp2 = (long long int)(temp);
		//temp2 = temp2 % int(pow(2, 31));
		ID_gmem[i] = int(rand());
	}

	auto t_elapsed = chrono::high_resolution_clock::now() - t_start;
	long mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	long mSec1 = mSec;
	///cout << "Reading data took " << mSec << " mSec" << endl;

// OPENCL HOST CODE AREA START
	
// ------------------------------------------------------------------------------------
// Step 1: Get All PLATFORMS, then search for Target_Platform_Vendor (CL_PLATFORM_VENDOR)
//	   Search for Platform: Xilinx 
// Check if the current platform matches Target_Platform_Vendor
// ------------------------------------------------------------------------------------	
    std::vector<cl::Device> devices = get_devices("Xilinx");
    devices.resize(1);
    cl::Device device = devices[0];

// ------------------------------------------------------------------------------------
// Step 1: Create Context
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::Context context(device, NULL, NULL, NULL, &err));
	
// ------------------------------------------------------------------------------------
// Step 1: Create Command Queue
// ------------------------------------------------------------------------------------
    OCL_CHECK(err, cl::CommandQueue q(context, device, CL_QUEUE_PROFILING_ENABLE, &err));

// ------------------------------------------------------------------
// Step 1: Load Binary File from disk
// ------------------------------------------------------------------		
    char* fileBuf = read_binary_file(binaryFile, fileBufSize);
    cl::Program::Binaries bins{{fileBuf, fileBufSize}};
	
// -------------------------------------------------------------
// Step 1: Create the program object from the binary and program the FPGA device with it
// -------------------------------------------------------------	
    OCL_CHECK(err, cl::Program program(context, devices, bins, NULL, &err));

// -------------------------------------------------------------
// Step 1: Create Kernels
// -------------------------------------------------------------
    OCL_CHECK(err, cl::Kernel krnl_hd(program,"hd", &err));

// ================================================================
// Step 2: Setup Buffers and run Kernels
// ================================================================
//   o) Allocate Memory to store the results 
//   o) Create Buffers in Global Memory to store data
// ================================================================

// .......................................................
// Allocate Global Memory for sources
// .......................................................	

    OCL_CHECK(err, cl::Buffer buf_input      (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*input_gmem.size(),  input_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_ID         (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*ID_gmem.size(), ID_gmem.data(), &err));
	OCL_CHECK(err, cl::Buffer buf_labels     (context,CL_MEM_USE_HOST_PTR | CL_MEM_READ_ONLY, sizeof(int)*labels_gmem.size(), labels_gmem.data(), &err));

// ============================================================================
// Step 2: Set Kernel Arguments and Run the Application
//         o) Set Kernel Arguments
//         o) Copy Input Data from Host to Global Memory on the device
//         o) Submit Kernels for Execution
//         o) Copy Results from Global Memory, device to Host
// ============================================================================	

	OCL_CHECK(err, err = krnl_hd.setArg(0, buf_input));
    OCL_CHECK(err, err = krnl_hd.setArg(1, buf_ID));
    OCL_CHECK(err, err = krnl_hd.setArg(2, buf_labels));
    OCL_CHECK(err, err = krnl_hd.setArg(3, EPOCH));
    OCL_CHECK(err, err = krnl_hd.setArg(4, N_SAMPLE));

// ------------------------------------------------------
// Step 2: Copy Input data from Host to Global Memory on the device
// ------------------------------------------------------
	OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_input, buf_ID}, 0 ));
	cout << "Data copied to the device global memory" << endl;	
// ----------------------------------------
// Step 2: Submit Kernels for Execution
// ----------------------------------------
	t_start = chrono::high_resolution_clock::now();
	
    OCL_CHECK(err, err = q.enqueueTask(krnl_hd));
	
// --------------------------------------------------
// Step 2: Copy Results from Device Global Memory to Host
// --------------------------------------------------
    OCL_CHECK(err, err = q.enqueueMigrateMemObjects({buf_labels}, CL_MIGRATE_MEM_OBJECT_HOST));
    q.finish();
	t_elapsed = chrono::high_resolution_clock::now() - t_start;
	
	mSec = chrono::duration_cast<chrono::milliseconds>(t_elapsed).count();
	
// OPENCL HOST CODE AREA END

    // Evaluate  the results

	/*
	long double distance = 0;
	int count = 0;

	for(int i = 0; i < N_SAMPLE; i++){
		//cout << labels_gmem[i] << endl;
		long double sum1 = 0;
		if(labels_gmem[i] < N_CENTER){
			count++;
			for(int j = 0; j < N_FEAT; j++){
				long double temp;
				if(shuffled)
					temp = X_data_shuffled[labels_gmem[i]*N_FEAT + j] - X_data_shuffled[i*N_FEAT + j];
				else
					temp = X_data[labels_gmem[i]*N_FEAT + j] - X_data[i*N_FEAT + j];
				sum1 += temp*temp;
			}
		}
		distance += sqrt(sum1);
	}
	*/
	cout << "\nReading data took " << mSec1 << " mSec" << endl;    
    cout << "Execution (" << EPOCH << " epochs)  took " << mSec << " mSec" << endl;
	
	ofstream myfile("out.txt");
	for(int i = 0; i < N_SAMPLE; i++){
		if(shuffled)
			myfile << y_data_shuffled[i] << " " << labels_gmem[i] << endl;
		else
			myfile << y_data[i] << " " << labels_gmem[i] << endl;
	}
	//calculate score
	//string command = "python -W ignore mutual_info.py";
	//system(command.c_str());
    
 //   cout << "\nNormalized distance:\t" << int(distance / count / Dhv) << endl;
    //cout << "\nAccuracy = " << float(correct)/N_SAMPLE << endl;
	
// ============================================================================
// Step 3: Release Allocated Resources
// ============================================================================
    delete[] fileBuf;

}

