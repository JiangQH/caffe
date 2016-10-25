#include <vector>
#include <cmath>
#include <string>
#include "caffe/layers/discrete_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {
/**
 * This is the discrete layer used to discrete the continuous value into discrete values.
 * to allow classification method applied to regression values. two method provided here.
 * one is using the ordinary space split to discrete into bins, another is using the clustering method like k-means to do this job.
 * specified by the param method.
 * */
template <typename Dtype>
void DiscreteLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// layer setup. get all the info
    discrete_num_ = this->layer_param_.discrete_param().num();
    discrete_space_ = this->layer_param_.discrete_param().space();
    discrete_method_ = this->layer_param_.discrete_param().method();
    discrete_min_ = this->layer_param_.discrete_param().mins();
    discrete_max_ = this->layer_param_.discrete_param().maxs();
    N_ = bottom[0]->shape(0);
    K_ = bottom[0]->shape(1);
    H_ = bottom[0]->shape(2);
    W_ = bottom[0]->shape(3);
    // do the check
    if (discrete_method_ == "ordinary") {
        // check the num and space if legal
       CHECK_GT(discrete_num_, 0) << "The discrete num must be larger than 0";
       CHECK(discrete_space_ == "linear" || discrete_space_ == "log") <<
           "illegal discrete_space. Be linear or log";
       CHECK(discrete_method_ == "ordinary" || discrete_method_ == "clustering")
           << "illegal discrete_method_. Be omZdinary or clustering";

       // get the channels of the bottom
       discrete_per_channel_ = round(exp(log(discrete_num_) / K_));
       LOG(INFO) << "The actually discrete num is " << pow(discrete_per_channel_, K_);
       // compute the delta according to the discrete space
       // if min is less than 0. then it should be the normal. transform it to values between 0 and 255
       if (discrete_min_ < 0) {
       		discrete_min_ = (discrete_min_ / 2 + 0.5) * 255;
       		discrete_max_ = (discrete_max_ / 2 + 0.5) * 255;
       		transform_ = true;
       }
       if (discrete_space_ == "linear") {
       		delta_ = (discrete_max_ - discrete_min_) / discrete_num_;
       }
       else if (discrete_space_ == "log"){
       		// add one to the value to avoid log(0)
       		discrete_min_ += 1;
       		discrete_max_ += 1;
       		discrete_max_ = log(discrete_max_);
       		discrete_min_ = log(discrete_min_);
       		delta_ = (discrete_max_ - discrete_min_) / discrete_num_;
       }
       else {
           LOG(ERROR) << "Unrecognized discrete space. be linear or log";
       }
    }
    else if (discrete_method_ == "clustering"){
        // using the clustering. todo

    }
    else {
        LOG(ERROR) << "Unrecognized discrete method, be ordinary or clustering.";
    }
}


template <typename Dtype>
void DiscreteLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
	const vector<Blob<Dtype>*>& top) {
	// reshape
	vector<int> top_shape;
	top_shape.push_back(N_);
	top_shape.push_back(1);// the lable channel is 1
	top_shape.push_back(H_);
	top_shape.push_back(W_);
	top[0]->Reshape(top_shape);
}


template <typename Dtype>
void DiscreteLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
	// forward
	// discrete the bottom according to the channel to get the finnal result
	Dtype* top_data = top[0]->mutable_cpu_data();
	const Dtype* bottom_data = bottom[0]->cpu_data();
	caffe_set(top[0]->count(), Dtype(0), top_data);
	if (discrete_method_ == "ordinary") {
		for (int n = 0; n < N_; ++n) {
			for (int h = 0; h < H_; ++h) {
				for (int w = 0; w < W_; ++w) {
					const int top_offset = (n * H_ + h ) * W_ + w;
					// get the value and index accoding to the bottom value
					int label = 0;
					for (int k = 0; k < K_; ++k) {
						const int bottom_offset = top_offset + k * H_ * W_;
						Dtype value = bottom_data[bottom_offset];
						if (transform_) {
							value = (value / 2 + 0.5) * 255; // when it is normal . do transform
						}
						if (discrete_space_ == "log") {
							value += 1; // if it is log. add 1
							value = log(value);
						}
						int indicate = (value - discrete_min_) / delta_;
                        /**
                        if (value != 0) {
                            LOG(INFO) << value << " " << indicate;
                        }
                         **/
						if (indicate < 0) {
							indicate = 0;
						}
						if (indicate >= discrete_num_) {
							indicate = discrete_num_ - 1;
						}
						label = label * discrete_num_ + indicate;// assign the label				}
					}
					// assign the top with label
					top_data[top_offset] = label;
				}
			}
		}
	}
    else {
        // todo
    }
}
	


#ifdef CPU_ONLY
    STUB_GPU(DiscreteLayer);
#endif

INSTANTIATE_CLASS(DiscreteLayer);
REGISTER_LAYER_CLASS(Discrete);

}// namespace caffe
