#include <vector>
#include <ostream>
#include "net.h"
#include "cpu.h"
#include <opencv2/opencv.hpp>
static void interp(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Interp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 2);// 
    pd.set(3, 288);// 
    pd.set(4, 288);// 
    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void scale(const ncnn::Mat& in, const float& scale, int scale_data_size, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Scale");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, scale_data_size);// scale_data_size
    pd.set(1, 0);// 

    op->load_param(pd);

    // set weights
    ncnn::Mat scales[1];
    scales[0].create(scale_data_size);// scale_data

    for (int i = 0; i < scale_data_size; i++)
    {
        scales[0][i] = scale;
    }

    op->load_model(ncnn::ModelBinFromMatArray(scales));

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void binary_op(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c,int op_type)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("BinaryOp");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, op_type);// op_type

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void concat(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& c,int axis)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Concat");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, axis);// axis

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    std::vector<ncnn::Mat> bottoms(2);
    bottoms[0] = a;
    bottoms[1] = b;

    std::vector<ncnn::Mat> tops(1);
    op->forward(bottoms, tops, opt);

    c = tops[0];

    op->destroy_pipeline(opt);

    delete op;
}
static void transpose(const ncnn::Mat& in, ncnn::Mat& out,const int& order_type)
{
    ncnn::Option opt;
    opt.num_threads = 2;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = true;

    ncnn::Layer* op = ncnn::create_layer("Permute");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, order_type);// order_type

    op->load_param(pd);

    op->create_pipeline(opt);

    ncnn::Mat in_packed = in;
    {
        // resolve dst_elempack
        int dims = in.dims;
        int elemcount = 0;
        if (dims == 1) elemcount = in.elempack * in.w;
        if (dims == 2) elemcount = in.elempack * in.h;
        if (dims == 3) elemcount = in.elempack * in.c;

        int dst_elempack = 1;
        if (op->support_packing)
        {
            if (elemcount % 8 == 0 && (ncnn::cpu_support_x86_avx2() || ncnn::cpu_support_x86_avx()))
                dst_elempack = 8;
            else if (elemcount % 4 == 0)
                dst_elempack = 4;
        }

        if (in.elempack != dst_elempack)
        {
            convert_packing(in, in_packed, dst_elempack, opt);
        }
    }

    // forward
    op->forward(in_packed, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void reduction(const ncnn::Mat& in, ncnn::Mat& out)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Reduction");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, 0);// sum
    pd.set(1, 0);// reduce_all
    pd.set(4, 1);//keepdims
    ncnn::Mat axes = ncnn::Mat(1);
    axes.fill(0);
    pd.set(3, axes);

    op->load_param(pd);

    op->create_pipeline(opt);

    // forward
    op->forward(in, out, opt);

    op->destroy_pipeline(opt);

    delete op;
}
static void threshold(ncnn::Mat& in, const float& threshold)
{
    ncnn::Option opt;
    opt.num_threads = 4;
    opt.use_fp16_storage = false;
    opt.use_packing_layout = false;

    ncnn::Layer* op = ncnn::create_layer("Threshold");

    // set param
    ncnn::ParamDict pd;
    pd.set(0, threshold);//

    op->load_param(pd);


    op->create_pipeline(opt);

    // forward
    op->forward_inplace(in, opt);

    op->destroy_pipeline(opt);

    delete op;
}


static float im2col_get_pixel(const float* im, int height, int width, int channels, int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width * (row + height * channel)];
}
//from https://github.com/pjreddie/darknet/blob/master/src/im2col.c
static ncnn::Mat im2col_cpu(const ncnn::Mat& data_im, int ksize, int stride, int pad)
{
    int c, h, w;
    int channels = data_im.c;
    int height = data_im.h;
    int width = data_im.w;

    int height_col = (height + 2 * pad - ksize) / stride + 1;
    int width_col = (width + 2 * pad - ksize) / stride + 1;;
    int channels_col = channels * ksize * ksize;

    ncnn::Mat data_col = ncnn::Mat(channels_col * height_col * width_col, 1, 1);
    data_col.fill(0.0f);

    for (c = 0; c < channels_col; c++) 
    {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) 
        {
            for (w = 0; w < width_col; ++w) 
            {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col.channel(0)[col_index] = im2col_get_pixel((const float*)data_im.data, height, width, channels, im_row, im_col, c_im, pad);
            }
        }
    }

    return data_col.reshape(height_col * width_col, channels_col);
}

static int position_embedding(ncnn::Mat& mask,int num_pos_feats,ncnn::Mat& pos)
{
    ncnn::Mat y_embed = ncnn::Mat(mask.w, mask.h, mask.c);
    ncnn::Mat x_embed = ncnn::Mat(mask.w, mask.h, mask.c);
    
    for (int i = 0; i < mask.c; i++)
    {
        for (int j = 0; j < mask.h; j++)
        {
            float* mask_data = mask.channel(i).row(j);
            float* x_embed_data = x_embed.channel(i).row(j);
            for (int k = 0; k < mask.w; k++)
            {
                for (int l = k; l >= 0; l--)
                    x_embed_data[k] += mask_data[l];
            }
        }
        float* mask_data = mask.channel(i);
        for (int j = 0; j < mask.w; j++)
        {
            for (int k = 0; k < mask.h; k++)
            {
                float* y_embed_data = y_embed.channel(i).row(k);
                for (int l = k; l >= 0; l--)
                    y_embed_data[j] += mask_data[l * mask.w];
            }
        }
    }
    for (int i = 0; i < y_embed.c; i++)
    {
        for (int j = 0; j < y_embed.h; j++)
        {
            for (int k = 0; k < y_embed.w; k++)
            {
                y_embed[j * y_embed.w + k] = y_embed[j * y_embed.w + k]*6.283185307179586/(y_embed.row(y_embed.h-1)[k] + 0.000001);
                
            }
        }
    }
    for (int i = 0; i < x_embed.c; i++)
    {
        for (int j = 0; j < x_embed.h; j++)
        {
            for (int k = 0; k < x_embed.w; k++)
            {
                x_embed[j * x_embed.w + k] = x_embed[j * x_embed.w + k] * 6.283185307179586 / (x_embed[j * x_embed.w + x_embed.w - 1] + 0.000001);
            }
        }
    }

   
    std::vector<float> dim_t;
    for (int i = 0; i < num_pos_feats; i++)
        dim_t.push_back(i);
    for (int i = 0; i < num_pos_feats; i++)
    {
        dim_t[i] = std::pow(10000.0, 2 * std::floor(dim_t[i] / 2) / num_pos_feats);
    }

    ncnn::Mat pos_x = ncnn::Mat(num_pos_feats, mask.w, mask.h);
    ncnn::Mat pos_y = ncnn::Mat(num_pos_feats, mask.w, mask.h);

    for (int i = 0; i < pos_x.c; i++)
    {
        float* pos_x_data = pos_x.channel(i);
        for (int j = 0; j < pos_x.h; j++)
        {
            for (int k = 0; k < pos_x.w; k++)
            {
                pos_x_data[j * pos_x.w + k] = x_embed[i * pos_x.h + j] / dim_t[k];
            }
        }
    }
    for (int i = 0; i < pos_y.c; i++)
    {
        float* pos_y_data = pos_y.channel(i);
        for (int j = 0; j < pos_y.h; j++)
        {
            for (int k = 0; k < pos_y.w; k++)
            {
                pos_y_data[j * pos_y.w + k] = y_embed[i * pos_y.h + j] / dim_t[k];
            }
        }
    }

   
    for (int i = 0; i < pos_x.c; i++)
    {
        float* data = pos_x.channel(i);
        for (int j = 0; j < pos_x.h; j++)
        {
            for (int k = 0; k < pos_x.w;)
            {
                data[j * pos_x.w + k] = std::sin(data[j * pos_x.w + k]);
                k += 2;
            }
                
            for (int k = 1; k < pos_x.w;)
            {
                data[j * pos_x.w + k] = std::cos(data[j * pos_x.w + k]);
                k += 2;
            }
                
        }
    }

    for (int i = 0; i < pos_y.c; i++)
    {
        float* data = pos_y.channel(i);
        for (int j = 0; j < pos_y.h; j++)
        {
            for (int k = 0; k < pos_y.w;)
            {
                data[j * pos_y.w + k] = std::sin(data[j * pos_y.w + k]);
                k += 2;
            }
                
            for (int k = 1; k < pos_y.w;)
            {
                data[j * pos_y.w + k] = std::cos(data[j * pos_y.w + k]);
                k += 2;
            }
                
        }
    }

    concat(pos_y, pos_x, pos,2);
    transpose(pos, pos, 4);
   
    return 0;

}

static void coords_grid(int h, int w,ncnn::Mat& coords)
{
    coords.create(w, h, 2);
    float* ptr0 = coords.channel(0);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            ptr0[i * w + j] = j;
        }
    }
    float* ptr1 = coords.channel(1);
    for (int i = 0; i < h; i++)
    {
        for (int j = 0; j < w; j++)
        {
            ptr1[i * w + j] = i;
        }
    }
}


static float within_bounds_2d(const ncnn::Mat& data, int x, int y, int c, int H, int W) 
{
    if (y >= 0 && y < H && x >= 0 && x < W)
        return data.channel(c)[y * W + x];
    else
        return 0;
}
//from https://github.com/open-mmlab/mmdeploy/blob/master/csrc/backend_ops/onnxruntime/grid_sample/grid_sample.cpp
static ncnn::Mat grid_sample(const ncnn::Mat& input,const ncnn::Mat& grid)
{
    int channel = input.c;
    int input_height = input.h;
    int input_width = input.w;
    int output_height = grid.c;
    int output_width = grid.h;

    ncnn::Mat out(input.w, input.h, input.c);
    out.fill(0.0f);

    for (int h = 0; h < output_height; h++)
    {
        for (int w = 0; w < output_width; w++)
        {
            float x = grid.channel(h)[w * 2 + 0];
            float y = grid.channel(h)[w * 2 + 1];

            float ix = (x + 1) * input_width * 0.5 - 0.5;
            float iy = (y + 1) * input_height * 0.5 - 0.5;

            int ix_nw = static_cast<int>(std::floor(ix));
            int iy_nw = static_cast<int>(std::floor(iy));
            int ix_ne = ix_nw + 1;
            int iy_ne = iy_nw;

            int ix_sw = ix_nw;
            int iy_sw = iy_nw + 1;
            
            int ix_se = ix_nw + 1;
            int iy_se = iy_nw + 1;

            float nw = (ix_se - ix) * (iy_se - iy);
            float ne = (ix - ix_sw) * (iy_sw - iy);
            float sw = (ix_ne - ix) * (iy - iy_ne);
            float se = (ix - ix_nw) * (iy - iy_nw);


            for (int c = 0; c < channel; c++) 
            {
                float nw_res = within_bounds_2d(input, ix_nw, iy_nw, c, input_height, input_width);
                float ne_res = within_bounds_2d(input, ix_ne, iy_ne, c, input_height, input_width);
                float sw_res = within_bounds_2d(input, ix_sw, iy_sw, c, input_height, input_width);
                float se_res = within_bounds_2d(input, ix_se, iy_se, c, input_height, input_width);
                out.channel(c)[h * input_width + w] = nw_res * nw + ne_res * ne + sw_res * sw + se_res * se;
            }
        }
    }
    return out;
}

static void to_ocv(const ncnn::Mat& result, cv::Mat& out)
{
    cv::Mat cv_result_32F = cv::Mat::zeros(cv::Size(result.w, result.h), CV_32FC3);
    for (int i = 0; i < result.h; i++)
    {
        for (int j = 0; j < result.w; j++)
        {
            cv_result_32F.at<cv::Vec3f>(i, j)[0] = result.channel(0)[i * result.w + j];
            cv_result_32F.at<cv::Vec3f>(i, j)[1] = result.channel(1)[i * result.w + j];
            cv_result_32F.at<cv::Vec3f>(i, j)[2] = result.channel(2)[i * result.w + j];
        }
    }

    cv::Mat cv_result_8U;
    cv_result_32F.convertTo(cv_result_8U, CV_8UC3, 255.0, 0);

    cv_result_8U.copyTo(out);

}

ncnn::Mat seg(cv::Mat& img)
{
    ncnn::Net seg_net;
    seg_net.load_param("./models/seg.param");
    seg_net.load_model("./models/seg.bin");

    
    ncnn::Mat seg_in = ncnn::Mat::from_pixels_resize(img.data, ncnn::Mat::PIXEL_BGR2RGB, img.cols, img.rows, 288, 288);
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    seg_in.substract_mean_normalize(0, norm_vals);
    ncnn::Extractor ex0 = seg_net.create_extractor();

    ex0.input("input", seg_in);
    ncnn::Mat seg_out;
    ex0.extract("out", seg_out);
    threshold(seg_out, 0.5f);

    binary_op(seg_in, seg_out, seg_in, 2);

    cv::Mat seg_result = cv::Mat(cv::Size(seg_out.w, seg_out.h), CV_32FC1, (float*)seg_out.data);
    cv::Mat seg_result_8u;
    seg_result.convertTo(seg_result_8u, CV_8UC1, 255.0, 0);
    cv::imwrite("seg_result.jpg", seg_result_8u);

    //interp(seg_in, seg_out);

    
    //cv::imshow("seg_result", seg_result_8u);
    //cv::waitKey();

    return seg_in;
}
cv::Mat warp_image(const ncnn::Mat& lbl, const cv::Mat& img)
{
    ncnn::Mat im_ori = ncnn::Mat::from_pixels(img.data, ncnn::Mat::PIXEL_BGR, img.cols, img.rows);
    const float norm_vals[3] = { 1 / 255.f, 1 / 255.f, 1 / 255.f };
    im_ori.substract_mean_normalize(0, norm_vals);
    ncnn::Mat out;
    out = grid_sample(im_ori, lbl);

    cv::Mat cv_out;
    to_ocv(out, cv_out);

    return cv_out;
}
ncnn::Mat geo(const ncnn::Mat& seg_out,const cv::Mat& img)
{
    ncnn::Mat coords0, coords1, coodslar;
    coords_grid(288, 288, coodslar);
    coords_grid(36, 36, coords0);
    coords_grid(36, 36, coords1);

    ncnn::Net decorer_net;
    decorer_net.opt.use_packing_layout = false;//there is some bug in packing layout
    decorer_net.load_param("./models/decoder.param");
    decorer_net.load_model("./models/decoder.bin");

    ncnn::Net fbnet_net;
    fbnet_net.load_param("./models/fbnet.param");
    fbnet_net.load_model("./models/fbnet.bin");

    ncnn::Net encoder_net;
    encoder_net.load_param("./models/encoder.param");
    encoder_net.load_model("./models/encoder.bin");

    ncnn::Net update_block;
    update_block.load_param("./models/update_block.param");
    update_block.load_model("./models/update_block.bin");

    ncnn::Mat posf;
    ncnn::Mat mask = ncnn::Mat(36, 36, 1);
    mask.fill(1.0f);
    position_embedding(mask, 128, posf);

    ncnn::Extractor ex1 = fbnet_net.create_extractor();
    ex1.input("input", seg_out);
    ncnn::Mat fmap1;
    ex1.extract("out", fmap1);

    //encoder
    ncnn::Extractor ex2 = encoder_net.create_extractor();
    ex2.input("imgf", fmap1);
    ex2.input("pos", posf);

    ncnn::Mat fmap2;
    ex2.extract("out", fmap2);
    //decoder
    ncnn::Extractor ex3 = decorer_net.create_extractor();
    ex3.input("imgf", fmap2);
    ex3.input("pos", posf);

    ncnn::Mat fmap3;
    ex3.extract("out", fmap3);

    ncnn::Extractor ex4 = update_block.create_extractor();
    ex4.input("imgf", fmap3);
    ex4.input("coords", coords1);

    ncnn::Mat fmask, coords1_out;
    ex4.extract("mask", fmask);
    ex4.extract("coords1", coords1_out);

    ncnn::Mat coords;
    binary_op(coords1_out, coords0, coords, 1);//sub
    scale(coords, 8.0, coords.c, coords);
    ncnn::Mat up_flow = im2col_cpu(coords, 3, 1, 1);

    ncnn::Mat up_flow1 = up_flow.reshape(1296, 1, 9, 2);

    ncnn::Mat up_flow11 = up_flow1.channel(0);
    ncnn::Mat up_flow12 = up_flow1.channel(1);
    ncnn::Mat fmask_up_flow11;
    binary_op(fmask, up_flow11, fmask_up_flow11, 2);//mul
    ncnn::Mat fmask_up_flow12;
    binary_op(fmask, up_flow12, fmask_up_flow12, 2);//mul

    ncnn::Mat fmask_up_flow11_sum, fmask_up_flow12_sum;
    reduction(fmask_up_flow11, fmask_up_flow11_sum);
    reduction(fmask_up_flow12, fmask_up_flow12_sum);

    fmask_up_flow11_sum = fmask_up_flow11_sum.reshape(1296, 64, 1, 1);
    ncnn::Mat fmask_up_flow11_sum_t;
    transpose(fmask_up_flow11_sum, fmask_up_flow11_sum_t, 2);
    fmask_up_flow11_sum_t = fmask_up_flow11_sum_t.reshape(36, 36, 64, 1);
    transpose(fmask_up_flow11_sum_t, fmask_up_flow11_sum_t, 6);
    fmask_up_flow11_sum_t = fmask_up_flow11_sum_t.reshape(36, 36, 8, 8);

    fmask_up_flow12_sum = fmask_up_flow12_sum.reshape(1296, 64, 1, 1);
    ncnn::Mat fmask_up_flow12_sum_t;
    transpose(fmask_up_flow12_sum, fmask_up_flow12_sum_t, 2);
    fmask_up_flow12_sum_t = fmask_up_flow12_sum_t.reshape(36, 36, 64, 1);
    transpose(fmask_up_flow12_sum_t, fmask_up_flow12_sum_t, 6);
    fmask_up_flow12_sum_t = fmask_up_flow12_sum_t.reshape(36, 36, 8, 8);

    transpose(fmask_up_flow11_sum_t, fmask_up_flow11_sum_t, 13);
    transpose(fmask_up_flow12_sum_t, fmask_up_flow12_sum_t, 13);

    fmask_up_flow11_sum_t = fmask_up_flow11_sum_t.reshape(288, 288, 1);
    fmask_up_flow12_sum_t = fmask_up_flow12_sum_t.reshape(288, 288, 1);
    concat(fmask_up_flow11_sum_t, fmask_up_flow12_sum_t, up_flow, 0);
    ncnn::Mat bm_up;
    binary_op(coodslar, up_flow, bm_up, 0);//add
    const float mean[2] = { 286.8f / 2, 286.8f / 2 };
    const float norm[2] = { 2 * 0.99 / 286.8f, 2 * 0.99 / 286.8f };
    bm_up.substract_mean_normalize(mean, norm);

    cv::Mat cv_bm0 = cv::Mat(cv::Size(bm_up.w, bm_up.h), CV_32FC1, bm_up.channel(0));
    cv::Mat cv_bm1 = cv::Mat(cv::Size(bm_up.w, bm_up.h), CV_32FC1, bm_up.channel(1));

    cv::resize(cv_bm0, cv_bm0, img.size(), 0, 0, 1);
    cv::resize(cv_bm1, cv_bm1, img.size(), 0, 0, 1);
    cv::blur(cv_bm0, cv_bm0, cv::Size(3, 3));
    cv::blur(cv_bm1, cv_bm1, cv::Size(3, 3));

    ncnn::Mat bm0 = ncnn::Mat(cv_bm0.cols * cv_bm0.rows, (void*)cv_bm0.data).reshape(cv_bm0.cols, cv_bm0.rows, 1);
    ncnn::Mat bm1 = ncnn::Mat(cv_bm0.cols * cv_bm0.rows, (void*)cv_bm1.data).reshape(cv_bm0.cols, cv_bm0.rows, 1);

    ncnn::Mat lbl;
    concat(bm0, bm1, lbl, 0);
    transpose(lbl, lbl, 3);

    return lbl;
}

int main(int argc, char** argv)
{
    if (argc != 2)
    {
        fprintf(stderr, "Usage: %s [imagepath]\n", argv[0]);
        return -1;
    }

    const char* imagepath = argv[1];

    cv::Mat img = cv::imread(imagepath, 1);
    if (img.empty())
    {
        fprintf(stderr, "cv::imread %s failed\n", imagepath);
        return -1;
    }

    ncnn::Mat seg_out = seg(img);
    ncnn::Mat lbl = geo(seg_out, img);
    cv::Mat warp_result = warp_image(lbl, img);
    
    cv::imwrite("warp_result.jpg", warp_result);
    //cv::imshow("warp_result", warp_result);
    //cv::waitKey();
    

    return 0;
}
