#include <algorithm>
#include <vector>

#include "caffe/layer.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/layers/batch_euclidean_map_layer.hpp"

namespace caffe {
  
  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::LayerSetUp(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  	num_ = bottom[0]->num();
  	channels_ = bottom[0]->channels();
  	CHECK_EQ(bottom[0]->height(), 1);
  	CHECK_EQ(bottom[0]->width(), 1);
  }

  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::Reshape(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
  	diff_feat.Reshape(1, bottom[0]->channels(), 1, 1);
  	top[0]->Reshape(bottom[0]->num(), bottom[0]->num(), 1, 1);
  }

  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top){
    const Dtype* bottom_data = bottom[0]->cpu_data();
    Dtype* top_data = top[0]->mutable_cpu_data();
    Dtype* diff_feat_data = diff_feat.mutable_cpu_data();
	max_d = Dtype(0.0);

    for (int n = 0; n < num_; ++n){
      for (int nn = 0; nn < num_; ++nn){
      	//diff_feat = x_n - x_nn
      	caffe_sub(channels_, bottom_data+n*channels_, bottom_data+nn*channels_, diff_feat_data);
      	//sim = diff_feat * diff_feat
      	Dtype distance = caffe_cpu_dot(channels_, diff_feat_data, diff_feat_data);
		if (distance > max_d){
			max_d = distance;
		}
      	//top[n, nn] = sim
      	caffe_set(1, distance, top_data+n*num_+nn);
      }
    }
	caffe_scal(num_*num_, Dtype(1.0) / max_d, top_data);
  }

  template <typename Dtype>
  void BatchEuclideanMapLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom){
  	if (!propagate_down[0]){return;}
  	const Dtype* top_diff = top[0]->cpu_diff();
  	Dtype* bottom_diff = bottom[0]->mutable_cpu_diff();
  	caffe_set(num_*channels_, Dtype(0.0), bottom_diff);
  	Dtype* diff_feat_data = diff_feat.mutable_cpu_data();
  	const Dtype* bottom_data = bottom[0]->cpu_data();
  	Dtype scale = Dtype(0.0);
  	for (int n = 0; n < num_; ++n){
      for (int nn = 0; nn < num_; ++nn){
      	caffe_sub(channels_, bottom_data+n*channels_, bottom_data+nn*channels_, diff_feat_data);
        caffe_copy(1, top_diff+n*num_+nn, &scale);
		//const Dtype* diff_feat_data2 = diff_feat.cpu_data();
        caffe_axpy(channels_, scale*Dtype(2.0) / max_d, diff_feat_data, bottom_diff+n*channels_);
      }
  	}
  }


  #ifdef CPU_ONLY
    STUB_GPU(BatchEuclideanMapLayer);
  #endif
  INSTANTIATE_CLASS(BatchEuclideanMapLayer);
  REGISTER_LAYER_CLASS(BatchEuclideanMap);
} // namespace caffe
