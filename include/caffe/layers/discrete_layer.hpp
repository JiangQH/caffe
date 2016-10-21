#ifndef CAFFE_DISCRETE_LAYER_HPP_
#define CAFFE_DISCRETE_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"
 
 namespace caffe {
/**
* this is the self define layer. used to do feature discretization.
* using this to convert the regression problem to a classification problem
*/
template <typename Dtype>
class DiscreteLayer: public Layer<Dtype> {

public:
	explicit DiscreteLayer<const LayerParameter& param> : 
		Layer(param) {/*some thing*/}

	virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom,
                           const vector<Blob<Dtype>*>& top);

    virtual void Reshape(const vector<Blob<Dtype>*>& bottom,
                        const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "Discrete"; }
    virtual inline int ExactNumBottomBlobs() const { return 1; }
  	virtual inline int ExactNumTopBlobs() const { return 1; }

protected:

	virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top);

	virtual void Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
		/* since it often used behind the data layer. so no backworad is needed*/
	}



};// end of class

 }//namespace caffe
#endif // CAFFE_DISCRETE_LAYER_HPP_