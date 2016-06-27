#include <vector>
#include <map>
#include <cmath>
#include <algorithm>

#include "caffe/layers/hypercolumns_layer.hpp"
#include "caffe/util/math_functions.hpp"

namespace caffe {

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    // forward step
    Dtype* top_normal = top[1]->mutable_gpu_data();
    Dtype* top_hypercolumns = top[0]->mutable_gpu_data();
    const Dtype* bottom_normal = bottom[0]->gpu_data();
    vector<int> sampling_list;
    caffe_set(top[1]->count(), Dtype(-1.0), top_normal);
    caffe_set(top[0]->count(), Dtype(0), top_hypercolumns);

    // for each batch, do the sampling job
    for (int n = 0; n < N_; ++n) {
        generate_list(sampling_list, bottom[0], n);
        if (is_train_) {
            selected_points_.insert(selected_points_.end(),
                    sampling_list.begin(), sampling_list.end());
        }
        // for every sampling point
        for (int id = 0; id < sampling_list.size(); ++id) {
            const int index = sampling_list[id];
            //LOG(INFO) << "for batch " << n << " index is " << index;
            // normal first. Here due to caffe's BGR channel change. need to change the channels back
            for (int c = 0; c < top[1]->shape(1); ++c) {
                const int top_index = top[1]->offset(n * sample_num_ + id, c);
                const int bottom_index = bottom[0]->offset(n,2-c) + index; // here the channel change
                top_normal[top_index] = get_true_normal(bottom_normal[bottom_index]);
            }
            // hyperfeature next
            int hyper_channel = 0;
            for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
                const Dtype* bottom_data = bottom[bottom_id]->gpu_data();
                const int channel = bottom[bottom_id]->shape(1);
                // get the correponding point in the corresponding bottom
                vector<int> original_size;
                original_size.push_back(bottom[bottom_id]->shape(2));
                original_size.push_back(bottom[bottom_id]->shape(3));
                std::map<int, double> weights;
                get_map_point(weights, index, original_size);
                // for debug usage, output the last and first to check
                /**if (id == 0 || id == sampling_list.size() - 1) {
                    LOG(INFO) << "for sample point " << id << " the selected point index is " << index << " the bottom is " << bottom_id;
                    for (std::map<int, double>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
                        LOG(INFO) << "the coresponding to bottom " << bottom_id << " point is " << iter->first << " weights is " << iter->second << std::endl;
                    }

                } **/
                for (int c = 0; c < channel; ++c){
                    // compute the value, according to the point
                    double value = 0;
                    for (std::map<int, double>::iterator iter = weights.begin(); iter != weights.end(); ++iter) {
                        const int bottom_index = bottom[bottom_id]->offset(n, c) + iter->first;
                        value += bottom_data[bottom_index] * iter->second;
                    }
                    const int top_index = top[0]->offset(n * sample_num_  + id, hyper_channel);
                    top_hypercolumns[top_index] = value;
                    ++hyper_channel;
                }
            }
        }
    }
}

template <typename Dtype>
void HyperColumnsLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top,
                                            const vector<bool>& propagate_down,
                                            const vector<Blob<Dtype>*>& bottom) {
    // backward step
    const Dtype* top_diff = top[0]->gpu_diff();
    // first set the value to zero for every bottom diff
    for (int i = 1; i < bottom.size(); ++i) {
        caffe_set(bottom[i]->count(), Dtype(0), bottom[i]->mutable_gpu_diff());
    }
    
    // do backward
    for (int index = 0; index < selected_points_.size(); ++index) {
        const int selected_index = selected_points_[index];
        const int n = index / sample_num_;
        int hyper_channel = 0;
        for (int bottom_id = 1; bottom_id < bottom.size(); ++bottom_id) {
            Dtype* bottom_diff = bottom[bottom_id]->mutable_gpu_diff();
            const int channel = bottom[bottom_id]->shape(1);
            vector<int> original_size;
            original_size.push_back(bottom[bottom_id]->shape(2));
            original_size.push_back(bottom[bottom_id]->shape(3));
            std::map<int, double> weights;
            get_map_point(weights, selected_index, original_size);
            for (int c = 0; c < channel; ++c) {
                const int top_index = top[0]->offset(index, hyper_channel);
                for (std::map<int, double>::iterator iter = weights.begin();
                     iter != weights.end(); ++iter) {
                    const int bottom_index = bottom[bottom_id]->offset(n, c) + iter->first;
                    bottom_diff[bottom_index] += top_diff[top_index] * iter->second;
                }
                ++hyper_channel;
            }
        }
    }
    selected_points_.clear();
}

INSTANTIATE_LAYER_GPU_FUNCS(HyperColumnsLayer)

} // namespace caffe
