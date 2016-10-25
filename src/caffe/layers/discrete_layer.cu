#include <vector>
#include <cmath>
#include <string>
#include "caffe/layers/discrete_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {



template <typename Dtype>
__global__ void ForwardLabel(const int nthreads,
	Dtype* const top_data, const Dtype* bottom_data, const int k, const int h, const int w,
	const int discrete_num, bool log_space,
	bool transform, const double delta, const double discrete_min
	) {
	CUDA_KERNEL_LOOP(index, nthreads) {
		const int top_offset = index;
		// get the label
		int lable = 0;
		for (int kk = 0; kk < k; ++kk) {
			const int bottom_offset = top_offset + kk * h * w;
			Dtype value = bottom_data[bottom_offset];
			if (transform) {
				value = (value / 2 + 0.5) * 255;
			}
			if (log_space) {
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
            lable = lable * discrete_num + indicate;
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

      if (discrete_method_ == "oridnary") {
            bool log_space = false;
            if (discrete_space_ == "log") {
                log_space = true;
            }
            ForwardLabel<Dtype><<<CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS>>>(
            count, top_data, bottom_data, K_, H_, W_, discrete_num_, log_space, transform_,
            delta_, discrete_min_
            );
      }
      else {
            // todo
      }

}

template <typename Dtype>
void DiscreteLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
        const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
      /** do nothing **/
}

INSTANTIATE_LAYER_GPU_FUNCS(DiscreteLayer);
}// namespace caffe