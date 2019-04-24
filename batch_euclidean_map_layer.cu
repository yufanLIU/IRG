#include <algorithm>
#include <vector>

#include "caffe/layers/batch_euclidean_map_layer.hpp"

namespace caffe {
  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::Forward_gpu(const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top){
  	const Dtype* bottom_data = bottom[0]->gpu_data();
    Dtype* top_data = top[0]->mutable_gpu_data();
    Dtype* diff_feat_data = diff_feat.mutable_gpu_data();

    max_d = Dtype(0.0);
    for (int n = 0; n < num_; ++n){
      for (int nn = 0; nn < num_; ++nn){
      	//diff_feat = x_n - x_nn
      	caffe_gpu_sub(channels_, bottom_data+n*channels_, bottom_data+nn*channels_, diff_feat_data);
      	//sim = diff_feat * diff_feat
      	Dtype distance = Dtype(0.0);
		caffe_gpu_dot(channels_, diff_feat_data, diff_feat_data, &distance);
		if (distance > max_d){
			max_d = distance;
		}
      	//top[n, nn] = sim
      	caffe_gpu_set(1, distance, top_data+n*num_+nn);
      }
    }
	caffe_gpu_scal(num_*num_, Dtype(1.0) / max_d, top_data);
  }

  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::Backward_gpu(const vector<Blob<Dtype>*>& top, const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  	if (!propagate_down[0]){return;}
  	const Dtype* top_diff = top[0]->gpu_diff();
  	Dtype* bottom_diff = bottom[0]->mutable_gpu_diff();
  	caffe_gpu_set(num_*channels_, Dtype(0.0), bottom_diff);
  	Dtype* diff_feat_data = diff_feat.mutable_gpu_data();
  	const Dtype* bottom_data = bottom[0]->gpu_data();
  	Dtype scale = Dtype(0.0);
  	for (int n = 0; n < num_; ++n){
      for (int nn = 0; nn < num_; ++nn){
      	caffe_gpu_sub(channels_, bottom_data+n*channels_, bottom_data+nn*channels_, diff_feat_data);
        caffe_copy(1, top_diff+n*num_+nn, &scale);
		//const Dtype* diff_feat_data2 = diff_feat.gpu_data();
        caffe_gpu_axpy(channels_, scale*Dtype(2.0) / max_d, diff_feat_data, bottom_diff+n*channels_);
      }
  	}
  }
  INSTANTIATE_LAYER_GPU_FUNCS(BatchEuclideanMapLayer);
}
