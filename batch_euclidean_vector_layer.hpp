#ifndef CAFFE_BATCH_EUCLIDEAN_VECTOR_LAYER_HPP_
#define CAFFE_BATCH_EUCLIDEAN_VECTOR_LAYER_HPP_

#include <vector>

#include "caffe/blob.hpp"
#include "caffe/layer.hpp"
#include "caffe/proto/caffe.pb.h"

namespace caffe {
    template <typename Dtype>
    class BatchEuclideanVectorLayer : public Layer<Dtype> {
    public:
        explicit BatchEuclideanVectorLayer(const LayerParameter& param) : Layer<Dtype>(param) {}
        virtual void LayerSetUp(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
		virtual void Reshape(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual inline const char* type() const { return "BatchEuclideanVector"; }
        virtual inline int ExactNumBottomBlobs() const { return 2; }
        virtual inline int MinTopBlobs() const { return 1; }
        virtual inline int MaxTopBlobs() const { return 1; }
    protected:
        virtual void Forward_cpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Backward_cpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        virtual void Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top);
        virtual void Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom);
        int channels_;
        int num_;
        Blob<Dtype> diff_feat;
		Dtype max_d;
	};
}  // namespace caffe

#endif //CAFFE_BatchEuclideanVector_LAYER_HPP_
