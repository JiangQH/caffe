#include <vector>
#include <map>

#include "caffe/layers/discrete_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void DiscreteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// layer setup
}


template <typename Dtype>
void DiscreteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// reshape
}


template <typename Dtype>
void DiscreteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// forward
}

/**
#ifdef CPU_ONLY
	STUB_GPU(DiscreteLayer);
#endif
**/
INSTANTIATE_CLASS(DiscreteLayer);
REGISTER_LAYER_CLASS(Discrete);
}// namespace caffe