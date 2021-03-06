/**
 * load the image and its corresponding normalmap
 * if do flipping. do both
 * */
#ifndef CAFFE_PIXEL_WISE_DATA_LAYER_HPP_
#define CAFFE_PIXEL_WISE_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
/**
 * @load image and normal both
 * @Jiang Qinhong. mail: mylivejiang@gmail.com
 * */
template <typename Dtype>
class PixelWiseDataLayer: public BasePrefetchingDataLayer<Dtype> {
public:
    explicit PixelWiseDataLayer(const LayerParameter& param)
        : BasePrefetchingDataLayer<Dtype>(param) {}

    virtual ~PixelWiseDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                               const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "PixelWiseData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 2; }

protected:
    shared_ptr<Caffe::RNG> prefetch_rgn_;
    virtual void ShuffleImages();
    virtual void load_batch(Batch<Dtype>* batch);
    vector<std::pair<std::string, std::string> > lines_; // store the image and label path
    int lines_id_;
};

}// namespace caffe
#endif //CAFFE_PIXEL_WISE_DATA_LAYER_HPP_
