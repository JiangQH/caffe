#include <vector>
#include <cmath>
#include <string>
#include "caffe/layers/discrete_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
__global_ void ForwardLabel(const int nthreads,
	const Dtype* bottom_data, const int k, const int h, const int w,
	const int discrete_num, string discrete_space,
	bool transform, const double delta, const double discrete_min
	) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int top_offset = index;
		// get the label
		int lable = 0;
		for (int kk = 0; kk < k; ++k) {
			const int bottom_offset = top_offset + kk * h * w;
			const Dtype value = bottom_data[bottom_offset];
			if (transform) {
				value = (value / 2 + 0.5) * 255;
			}
			if (discrete_space == "log") {
				value += 1;
				value = log(value);
			}
			int indicate = (value - discrete_min) / delta;
			if (indicate < 0) {
				indicate = 0;
			}
			if (indicate >= discrete_num) {
				indicate = discrete_num - 1;
			}
			label = lable * discrete_num + indicate;

		}
		top_data[top_offset] = lable;
	}
}



template <typename Dtype>
void DiscreteLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      
      Dtype* top_data = top[0]->mutable_gpu_data();
      const Dtype* bottom_data = bottom[0]->gpu_data();
      caffe_gpu_set(top[0]->count(), Dtype(0), top_data);

      const int count = bottom[0]->count();
      ForwardLable<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
      		count, bottom_data, K_, H_, W_, discrete_num_, discrete_space_, transform_,
      		delta_, discrete_min_
      );
}

template <typename Dtype>
void DiscreteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
      /** do nothing **/
}

}// namespace caffe