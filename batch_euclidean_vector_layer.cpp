#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/batch_euclidean_vector_layer.hpp"

namespace caffe {
  
  template <typename Dtype>
  void BatchEuclideanVectorLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  	num_ = bottom[0]->num();
  	channels_ = bottom[0]->channels();
  	CHECK_EQ(bottom[0]->height(), 1);
  	CHECK_EQ(bottom[0]->width(), 1);
    CHECK_EQ(bottom[1]->height(), 1);
    CHECK_EQ(bottom[1]->width(), 1);
    CHECK_EQ(channels_, bottom[1]->channels());
  }

  template <typename Dtype>
  void BatchEuclideanVectorLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  	diff_feat.Reshape(1, channels_, 1, 1);
    top[0]->Reshape(1, 1, 1, num_);
  }

  template <typename Dtype>
  void BatchEuclideanVectorLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    const Dtype* bottom_data1 = bottom[0]->cpu_data();
    const Dtype* bottom_data2 = bottom[1]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* diff_feat_data = diff_feat.mutable_cpu_data();
    max_d = Dtype(0.0);
    for(int n = 0; n < num_; ++n){
      caffe_sub(channels_, bottom_data1+n*channels_, bottom_data2+n*channels_, diff_feat_data);
      Dtype distance = caffe_cpu_dot(channels_, diff_feat_data, diff_feat_data);
      if (distance > max_d){
        max_d = distance;
      }
      caffe_set(1, distance, top_data+n);
    }
    caffe_scal(num_, Dtype(1.0) / max_d, top_data);
  }

  template <typename Dtype>
  void BatchEuclideanVectorLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  	if (!propagate_down[0]){return;}
  	const Dtype* top_diff = top[0]->cpu_diff();
  	Dtype* bottom_diff1 = bottom[0]->mutable_cpu_diff();
    Dtype* bottom_diff2 = bottom[1]->mutable_cpu_diff();
  	caffe_set(num_*channels_, Dtype(0.0), bottom_diff1);
    caffe_set(num_*channels_, Dtype(0.0), bottom_diff2);
  	Dtype* diff_feat_data = diff_feat.mutable_cpu_data();
  	const Dtype* bottom_data1 = bottom[0]->cpu_data();
    const Dtype* bottom_data2 = bottom[1]->cpu_data();
  	Dtype scale = Dtype(0.0);
      for (int n = 0; n < num_; ++n){
        caffe_sub(channels_, bottom_data1+n*channels_, bottom_data2+n*channels_, diff_feat_data);
        caffe_copy(1, top_diff+n, &scale);
    //const Dtype* diff_feat_data2 = diff_feat.cpu_data();
        caffe_axpy(channels_, scale*Dtype(2.0) / max_d, diff_feat_data, bottom_diff1+n*channels_);
        caffe_axpy(channels_, scale*Dtype(-2.0) / max_d, diff_feat_data, bottom_diff2+n*channels_);
    }
  }


  #ifdef CPU_ONLY
    STUB_GPU(BatchEuclideanVectorLayer);
  #endif
  INSTANTIATE_CLASS(BatchEuclideanVectorLayer);
  REGISTER_LAYER_CLASS(BatchEuclideanVector);
} // namespace caffe
