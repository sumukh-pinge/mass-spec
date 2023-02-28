
#include "hd.h"

/*
 * inputStream fetches input features as ints, and streames to the next functions.
 *
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * feature_stream (output): N_FEAT_PAD parallel streams to stream the data to the next module.
 * size (input): number of data sampels.
 */
void inputStream(int *input_gmem, hls::stream<int> feature_stream[N_FEAT_PAD], int EPOCH, int size){

	loop_epoch:
	for(int iter_epoch = 0; iter_epoch < EPOCH; iter_epoch++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1 MAX=1
		loop_inputs:
		for(int iter_read = 0; iter_read < size; iter_read++){
			#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000
			 //Need to move the pointer by intPerInput after each input
			int offset = iter_read * N_FEAT;
			loop_features:
			for(int i = 0; i < N_FEAT; i++){
				feature_stream[i] << input_gmem[offset + i];
			}
			for(int i = 0; i < PAD; i++){
				feature_stream[N_FEAT + i] << 0;
			}
		}
	}

}

/*
 * encodeUnit reads input features from the stream and obtains encoding hypervector using random projection (RP) algorithm.
 * RP is all about a matrix-vector multplicatoin. The matrix is ID hypervectors arrangaed as Dhv*Div (Dhv = HV dimensions, Div = #input vector elements).
 * We break this A=(Dhv*Div) * B=(Div*1) multiplicaton into slinding windows of ROW*COL on the matrix A.
 * It takes Div/COL cycles for the sliding window to reach the right-most column, when we accomplish ROW dimensions. Total cycles = (Dhv*Div)/(ROW*COL)*(latency of accumulating COL partials).
 * Note that we only have a single seed ID (Dhv bits) in a (Dhv/ROW) x ROW array for column 0, and generate the ID of column k by k-bit rotating (circular shift).
 *
 * feature_stream (input): N_FEAT_PAD parallel streams from the previous module to read input features of a data sample.
 * ID (input): seed ID hypervector, packed as Dhv/ROW (rows) of ROW bit (total Dhv bits).
 * enc_stream (output): streams ROW encoded dimensions per (Div/COL) cycles to the next module.
 * size (input): number of data samples.
 */
template<int ROW_, int COL_>
void encodeUnit(hls::stream<int> feature_stream[N_FEAT_PAD], ap_int<ROW> ID[Dhv/ROW], hls::stream<int> enc_stream[ROW], int EPOCH, int size){

	//Operate on ROW encoding dimension per cycle.
	int encHV_partial[ROW];
	#pragma HLS array_partition variable=encHV_partial

	int feature_array[N_FEAT_PAD];
	//Factor the feature memory into COL, as we read COL elements of it in parallel.
	#pragma HLS array_partition variable=feature_array cyclic factor=COL_

	//ID register to keep ROW+COL bits for a ROW*COL window.
	//ID memory has ROW bits per cell, so we use 2*ROW bit register (extra bits will be used in the next window).
	//It might look a little tricky. See the report for visualization.
	ap_int<2*ROW> ID_reg;

	loop_repeat:
	for(int iter_epoch = 0; iter_epoch < EPOCH; iter_epoch++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1 MAX=1
	loop_inputs:
		for(int iter_read = 0; iter_read < size; iter_read++){
			#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000
			//Read all features into the feature_array
			loop_stream:
			for(int i = 0; i < N_FEAT_PAD; i++){
				#pragma HLS UNROLL factor=COL_
				feature_stream[i] >> feature_array[i];
			}

			//Probe ROW rows simultanously for mat-vec multplication (result = r encoding dimension).
			//Each row block has Dhv/ROW rows.
			loop_mat_row:
			for(int r = 0; r < Dhv/ROW; r++){
				//Clear the partial encoding buffer when the window starts the new rows.
				loop_clear:
				for(int i = 0; i < ROW; i++){
					#pragma HLS UNROLL factor=ROW_
					encHV_partial[i] = 0;
				}
				//We need to figure out which ID bits should be read.
				//At the beginning of row block r, we read bits of the block r and r+1 (each block has Dhv/ROW bits).
				int cycle = 0;
				int addr1 = r;
				int addr2 = r+1;
				//In the last block, r+1 becomes Dhv/ROW, so we start from 0 (ID bits are stored circular).
				if(addr2 == Dhv/ROW)
					addr2 = 0;
				ID_reg.range(ROW-1, 0) = ID[addr1];
				ID_reg.range(2*ROW-1, ROW) = ID[addr2];

				//Divide each of row blocks into columns (tiles) of COL, i.e., multiply a ROW*COL tile to COL features at a given cycle.
				loop_mat_col:
				for(int c = 0; c < N_FEAT_PAD/COL; c++){
					#pragma HLS PIPELINE

					//Iterate over the rows and columns of the ROW*COL tile to perform matrix-vector multplication.
					loop_tile_row:
					for(int i = 0; i < ROW; i++){
						#pragma HLS UNROLL factor=ROW_
						//In each ID register of 2*ROW bits, bits [0-COL) are for the first row, [1, COL+1) for the second, and so on.
						ap_int<COL> ID_row = ID_reg.range(i+COL-1, i);
						loop_tile_col:
						for(int j = 0; j < COL; j++){
							#pragma HLS UNROLL factor=COL_
							//For column group c, we read features c*COL to (c+1)*COL.
							int feature = feature_array[c*COL + j];
							if(ID_row[j] == 1)
								encHV_partial[i] += feature;
							else
								encHV_partial[i] -= feature;
						}
					}
					//After the first window, we move the window to right.
					//The initial 2*ROW ID block has enough bits for ROW/COL consecutive windows (as each window needs ROW+COL bits, not 2*ROW bits).
					//Otherwise, we update the ID address to get the new required ID bits.
					cycle += 1;
					if(cycle == ROW/COL){
						cycle = 0;
						addr1 = addr1 + 1;
						addr2 = addr2 + 1;
						if(addr1 == Dhv/ROW)
							addr1 = 0;
						if(addr2 == Dhv/ROW)
							addr2 = 0;
						ID_reg.range(ROW-1, 0) = ID[addr1];
						ID_reg.range(2*ROW-1, ROW) = ID[addr2];
					}
					//We have not reached the bound of ROW/COL, so the ID register contains the needed bits.
					//Just shift right by COL, so 'ID_reg.range(i+COL-1, i)' gives the correct ID bits per each row i of the ID block.
					//E.g., in a 4x2 window, in the first cycle we need bits 0-1 for row 1, while in the next cycle we need bits 2-3, so shift by COL=2 is needed.
					else{
						ID_reg = (ID_reg >> COL);
					}
				}
				//Output the ROW generated dimensions for subsequent pipelined search.
				//Note that we use quantized random projection. Otherwise, we will need higher bit-width for classes (and tmp resgiter during dot-product).
				loop_enc_stream:
				for(int i = 0; i < ROW; i++){
					#pragma HLS UNROLL factor=ROW_
					enc_stream[i] << encHV_partial[i];
					//if(iter_epoch == 0 && iter_read == 1)
						//cout << encHV_partial[i] << endl;
//					if(encHV_partial[i] >= 0)
//						enc_stream[i] << 1;
//					else
//						enc_stream[i] << -1;
				}
			}
		}
		//Iterating over inputs ends here.
	}
}


/*
 * searchUnit is the major component of our implementation. It runs EPOCH times over the data.
 * First, it reads all the encoding hypervectors ROW by ROW (ROW elements in Div/COL cycles) from the encodeUnit unit, and stores them in the global memory to reuse in clustering.
 * Thereafter, it reads the encoded hypervector from the global memory.
 *
 * enc_stream (input): ROW parallel stream of bipolar (+1, -1) dimensions from encoding unit.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * labels_gmem (output): final cluster index of each data sample.
 * EPOCH (input): number of iteration over data.
 * size (input): number of data samples.
 */

template<int ROW_>
void searchUnit(hls::stream<int> enc_stream[ROW], int *labels_gmem, int EPOCH, int size){

	//Explained previously: to read ROW encoding dimensions per cycle.
	int encHV_partial[ROW];
	#pragma HLS array_partition variable=encHV_partial

	//To store the dot-product of the centroid classes with the encoding hypervector.
	ap_int<64> dotProductRes[N_CENTER];
	#pragma HLS array_partition variable=dotProductRes

	//To store an encoding hypervector.
	int encHV_full[Dhv];
	#pragma HLS array_partition variable=encHV_full cyclic factor=ROW_

	//Centroid hypervectors.
	int centerHV[N_CENTER][Dhv];
	#pragma HLS array_partition variable=centerHV cyclic factor=ROW_ dim=2

	//Temporary centroid class hypervectors to bundle the encoded hypervectors belonging to the same centroid.
	int centerHV_temp[N_CENTER][Dhv];
	//We partition each class dimensions into ROW elements to match the ROW generated dimensions.
	#pragma HLS array_partition variable=centerHV_temp cyclic factor=ROW_ dim=2

	float norm2_inv[N_CENTER];

	//Iterate over the encoded data for EPOCH times.
	loop_repeat:
	for(int iter_epoch = 0; iter_epoch < EPOCH; iter_epoch++){
		#pragma HLS LOOP_TRIPCOUNT MIN=1 MAX=1

		//update the centroids in the next epochs
		if(iter_epoch > 0){
			loop_centers:
			for(int i_center = 0; i_center < N_CENTER; i_center++){
				ap_int<64> total = 0;
				for(int dim = 0; dim < Dhv; dim++){
					#pragma HLS UNROLL factor=ROW_
					ap_int<64> temp = centerHV_temp[i_center][dim];
					//centerHV[i_center][dim] = temp >= 0 ? 1 : -1;
					centerHV[i_center][dim] = temp;
					centerHV_temp[i_center][dim] = 0;
					total += temp*temp;
				}
				if(total == 0)
					norm2_inv[i_center] = 0;
				else{
					norm2_inv[i_center] = 1.0 / float(total);
				}
			}
		}

		loop_inputs_:
		for(int iter_read = 0; iter_read < size; iter_read++){
			#pragma HLS LOOP_TRIPCOUNT MIN=1000 MAX=1000

			//Reset the dotProductRes (score buffer) before each input sample.
			loop_clear:
			for(int i = 0; i < N_CENTER; i++){
				#pragma HLS UNROLL
				dotProductRes[i] = 0;
			}

			//read one encoded HV
			loop_readEnc:
			for(int i_dim = 0; i_dim < Dhv; i_dim += ROW){
				loop_stream:
				for(int j_sub = 0; j_sub < ROW; j_sub++){
					#pragma HLS UNROLL factor=ROW_
						enc_stream[j_sub] >> encHV_partial[j_sub];
						encHV_full[i_dim + j_sub] = encHV_partial[j_sub];
				}
			}
			//Initialize the centroids in the first epoch
			//Choose first N_CENTER points as centroids
			if((iter_epoch == 0) && (iter_read < N_CENTER)) {
				ap_int<64> total = 0;
				for(int dim = 0; dim < Dhv; dim++){
					#pragma HLS UNROLL factor=ROW_
					ap_int<64> temp = encHV_full[dim];
					centerHV[iter_read][dim] = temp;
					centerHV_temp[iter_read][dim] = 0;
					total += (temp*temp);
				}
				if(total == 0)
					norm2_inv[iter_read] = 0;
				else
					norm2_inv[iter_read] = 1.0/float(total);
				//cout << norm2_inv[iter_read] << endl;
			}
			else {//In the next epochs, or after the centorids are initialized
				//Compare the hypervector with all centroid hypervectors.
				loop_score:
				for(int i_center = 0; i_center < N_CENTER; i_center++){
					for(int dim = 0; dim < Dhv; dim++){
						#pragma HLS UNROLL factor=ROW_
						dotProductRes[i_center] += ap_int<64>(encHV_full[dim]) * ap_int<64>(centerHV[i_center][dim]);
					}
				}
				//Find out the max index
				int maxIndex = 0;
				float maxVal = -(1 << 20);
				loop_max:
				for(int i_center = 0; i_center < N_CENTER; i_center++){
					float temp = dotProductRes[i_center]*norm2_inv[i_center];
					float score = temp * dotProductRes[i_center];
					if(dotProductRes[i_center] < 0)
						score = -score;
					if(score > maxVal){
						maxIndex = i_center;
						maxVal = score;
					}
				}
				//Add the encoded hypervector to the nearest temporary centroid
				for(int dim = 0; dim < Dhv; dim++){
					#pragma HLS UNROLL factor=ROW_
					centerHV_temp[maxIndex][dim] += encHV_full[dim];
				}
				//If it is the last epoch, output the index of the centroid as the label of this input sample.
				if(iter_epoch == EPOCH-1){
					labels_gmem[iter_read] = maxIndex;
				}
			}
		}
	}
}

template<int ROW_, int COL_>
void top(int *input_gmem, int *ID_gmem, int *labels_gmem, int EPOCH, int size){

	static hls::stream<int> feature_stream[N_FEAT_PAD];
	#pragma HLS STREAM variable=feature_stream depth=2

	//For now, the encoding stream is integer while we are using bipolar (+1, -1) encoding. Fix it later.
	static hls::stream<int> enc_stream[ROW];
	#pragma HLS STREAM variable=enc_stream depth=2

	//We have a seed ID of Dhv length, and we partition it to Dhv/ROW pieces of ROW bits as we operate on ROW rows at the same time.
	ap_int<ROW> ID[Dhv/ROW];
	#pragma HLS array_partition variable=ID cyclic factor=4

	//Initialize the seed ID hypervector
	int offset = 0;
	loop_initID:
	for(int i = 0; i < Dhv/32; i++){
		ap_int<32> ID_int = ID_gmem[i];
		//If ROW is smaller than 32, each IDarray will fill several ID elements.
		if(ROW < 32){
			for(int j = 0; j < 32/ROW; j++){
				ID[i*32/ROW + j] = ID_int.range((j+1)*ROW - 1, j*ROW);
			}
		}//Otherwise, for each ID element, we need to read several IDarray elements.
		else{
			ID[i*32/ROW].range(32*offset + 31, 32*offset) = ID_int;
			offset += 1;
			if(offset == ROW/32)
				offset = 0;
		}
	}

	#pragma HLS dataflow
	inputStream(input_gmem, feature_stream, EPOCH, size);
	encodeUnit<ROW, COL>(feature_stream, ID, enc_stream, EPOCH, size);
	searchUnit<ROW>(enc_stream, labels_gmem, EPOCH, size);

}

/*
 * input_gmem (input): input data port; each feature is quantized to an integer.
 * ID_gmem (input): seed ID hypervector, packed to ints.
 * encHV_gmem (input/output): interface to write/read encoded hypervectors to/from the DRAM to reuse encoded data.
 * labels_gmem (output): final cluster of each data sample.
 * EPOCH (input): number of iteration over data.
 * size (input): number of data samples.
 */

extern "C" {
void hd(int *input_gmem, int *ID_gmem, int *labels_gmem, int EPOCH, int size){

	#pragma HLS INTERFACE m_axi port=input_gmem   offset=slave bundle=gmem0
	#pragma HLS INTERFACE m_axi port=ID_gmem      offset=slave bundle=gmem1
	#pragma HLS INTERFACE m_axi port=labels_gmem  offset=slave bundle=gmem2


	#pragma HLS INTERFACE s_axilite port=input_gmem   bundle=control
	#pragma HLS INTERFACE s_axilite port=ID_gmem      bundle=control
	#pragma HLS INTERFACE s_axilite port=labels_gmem  bundle=control
	#pragma HLS INTERFACE s_axilite port=EPOCH        bundle=control
	#pragma HLS INTERFACE s_axilite port=size         bundle=control

	#pragma HLS INTERFACE s_axilite port=return       bundle=control

	top<ROW, COL> (input_gmem, ID_gmem, labels_gmem, EPOCH, size);

}
}

