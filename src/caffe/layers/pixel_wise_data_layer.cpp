/*************************************************************************
	> File Name: pixel_wise_data_layer.cpp
	> Author: Jiang Qinhong
	> Mail: 
	> Created Time: 2016年06月03日 星期五 14时07分02秒
 ************************************************************************/
#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>

#include <fstream>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/pixel_wise_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

namespace caffe {

template <typename Dtype>
PixelWiseDataLayer<Dtype>::~PixelWiseDataLayer<Dtype>() {
    this->StopInternalThread();
}

template <typename Dtype>
void PixelWiseDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
        const vector<Blob<Dtype>*>& top) {
    const int new_height = this->layer_param_.pixel_wise_data_param().new_height();
    const int new_width = this->layer_param_.pixel_wise_data_param().new_width();
    const bool is_color = this->layer_param_.pixel_wise_data_param().is_color();
    string root_folder = this->layer_param_.pixel_wise_data_param().root_folder();
    const int label_channel = this->layer_param_.pixel_wise_data_param().label_channel();
    CHECK((new_height == 0 && new_width == 0) || 
            (new_height > 0 && new_width >0)) << "new_height and new_width should be set at the same time";
    // read the file with filenames and read the coresponding labels both
    const string& source = this->layer_param_.pixel_wise_data_param().source();
    LOG(INFO) << "opeing file " << source;
    std::ifstream infile(source.c_str());
    string line;
    size_t pos;
    while (std::getline(infile, line)) {
        pos = line.find_last_of(' ');
        lines_.push_back(std::make_pair(line.substr(0, pos), line.substr(pos+1)));
    }

    if(this->layer_param_.pixel_wise_data_param().shuffle()) {
        //randomly shuffle data
        LOG(INFO) << "Shuffing data";
        const unsigned int prefetch_rgn_seed = caffe_rng_rand();
        prefetch_rgn_.reset(new Caffe::RNG(prefetch_rgn_seed));
        ShuffleImages();
    }
    LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    //read an image .use it to init the top blobs(rgb and label with same size)
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                    new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    // use data_transformer to infer the expected blob shape from a cv_image
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // reshape prefetch_data and top[0], top[1] according to the batch_size
    const int batch_size = this->layer_param_.pixel_wise_data_param().batch_size();
    CHECK_GT(batch_size, 0) << "Positive batch size required";
    top_shape[0] = batch_size;
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
        this->prefetch_[i].label_.Reshape(top_shape);
    }
    top[0]->Reshape(top_shape);
    top_shape[1] = label_channel;
    top[1]->Reshape(top_shape);
    LOG(INFO) << "output image data size: " << top[0]->num() << ","
        << top[0]->channels() << "," << top[0]->height() << ","
        << top[0]->width();
    LOG(INFO) << "output label data size: " << top[1]->num() << ","
        << top[1]->channels() << "," << top[1]->height() << ","
        << top[1]->width();
}

template <typename Dtype>
void PixelWiseDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
      static_cast<caffe::rng_t*>(prefetch_rgn_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// this function is called on prefetch thread
template <typename Dtype>
void PixelWiseDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());
    PixelWiseDataParameter pixel_wise_data_param = this->layer_param_.pixel_wise_data_param();
    const int batch_size = pixel_wise_data_param.batch_size();
    const int new_height = pixel_wise_data_param.new_height();
    const int new_width = pixel_wise_data_param.new_width();
    const bool is_color = pixel_wise_data_param.is_color();
    const int label_channel = pixel_wise_data_param.label_channel();
    string root_folder = pixel_wise_data_param.root_folder();
    //Reshape according to the first image of each batch
    //on single input btaches allows for inputs of varying dimension.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
            new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
    vector<int> top_shape = this->data_transformer_->InferBlobShape(cv_img);
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);
    top_shape[1] = label_channel;
    batch->label_.Reshape(top_shape);

    Dtype* prefetch_data = batch->data_.mutable_cpu_data();
    Dtype* prefetch_label = batch->label_.mutable_cpu_data();

    //datum scales
    const long lines_size = lines_.size();
    for (int item_id = 0; item_id < batch_size; ++item_id) {
        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);
        int rows, cols;
        cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                new_height, new_width, is_color, rows, cols);
        cv::Mat cv_label = ReadLabelToCVMat(root_folder + lines_[lines_id_].second, rows, cols,
                new_height, new_width, label_channel);
       // LOG(INFO) << cv_label.at<cv::Vec3f>(120,213)[1] << " outside";
        CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;
        CHECK(cv_label.data) << "Could not load " << lines_[lines_id_].second;
        // for debug
        //LOG(INFO) << "load image " << lines_[lines_id_].first << " load the label " << lines_[lines_id_].second;
        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations to the image and label
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(prefetch_data + offset);
        bool do_mirror = this->data_transformer_->TransformImg(cv_img, &(this->transformed_data_));
        offset = batch->label_.offset(item_id);
        this->transformed_data_.set_cpu_data(prefetch_label + offset);
        this->data_transformer_->TransformLabel(cv_label, &(this->transformed_data_), do_mirror);
        trans_time += timer.MicroSeconds();
        // go to the next iter
        lines_id_++;
        if (lines_id_ >= lines_size) {
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.pixel_wise_data_param().shuffle()) {
                ShuffleImages();
            }
        }
    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << " Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000<< " ms.";
}

INSTANTIATE_CLASS(PixelWiseDataLayer);
REGISTER_LAYER_CLASS(PixelWiseData);


}//namespace caffe
#endif //USE_OPENCV

