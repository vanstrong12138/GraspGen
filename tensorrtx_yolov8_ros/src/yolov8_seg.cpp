#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include "cuda_utils.h"
#include "logging.h"
#include "model.h"
#include "postprocess.h"
#include "preprocess.h"
#include "utils.h"

#include "ros/ros.h"
#include <sensor_msgs/Image.h>
#include <sensor_msgs/CameraInfo.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud2_iterator.h>  
#include <cv_bridge/cv_bridge.h>
#include "opencv2/opencv.hpp"
#include <image_transport/image_transport.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/common/centroid.h>
#include <pcl_conversions/pcl_conversions.h>  
#include <visualization_msgs/MarkerArray.h>
#include <AABBGraspPlanner.hpp>

Logger gLogger;
using namespace nvinfer1;
const int kOutputSize = kMaxNumOutputBbox * sizeof(Detection) / sizeof(float) + 1;
const static int kOutputSegSize = 32 * (kInputH / 4) * (kInputW / 4);

class YoloV8SegROSNode {
private:
    ros::NodeHandle nh_;
    ros::NodeHandle private_nh_;
    image_transport::ImageTransport it_;
    image_transport::Subscriber image_sub_;
    image_transport::Subscriber depth_sub_;
    ros::Subscriber camera_info_sub_;
    image_transport::Publisher result_pub_;
    ros::Publisher pc_pub_;
    ros::Publisher centroid_pub_;  // 用于发布质心MarkerArray

    std::string color_image_topic;
    std::string depth_image_topic;
    std::string depth_camera_info_topic;
    
    // YOLO related variables
    // 修改为 shared_ptr 并使用自定义删除器
    std::shared_ptr<IRuntime> runtime_;
    std::shared_ptr<ICudaEngine> engine_;
    std::shared_ptr<IExecutionContext> context_;
    cudaStream_t stream_;
    
    float* device_buffers_[3];
    float* output_buffer_host_;
    float* output_seg_buffer_host_;
    float* decode_ptr_host_;
    float* decode_ptr_device_;
    
    std::vector<cv::Mat> img_batch_;
    std::string cuda_post_process_;
    int model_bboxes_;
    std::string engine_path_;
    std::string labels_path_;
    bool debug_;
    bool publish_results_;
    
    std::unordered_map<int, std::string> labels_map_;

    // Depth to point cloud variables
    cv::Mat K_;
    bool camera_info_received_;
    std::string camera_frame_;
    float depth_scale_;
    cv::Mat last_depth_image_;

    grasp_planner::AABBGraspPlanner planner_;  // Add this line

public:
    YoloV8SegROSNode() : private_nh_("~"), it_(nh_), camera_info_received_(false) ,planner_(nh_){
        setlocale(LC_ALL, "");
        cudaSetDevice(kGpuId);
        
        // Load parameters
        private_nh_.param<std::string>("engine_path", engine_path_, "");
        private_nh_.param<std::string>("labels_path", labels_path_, "");
        private_nh_.param<std::string>("cuda_post_process", cuda_post_process_, "c");
        private_nh_.param<bool>("debug", debug_, true);
        private_nh_.param<bool>("publish_results", publish_results_, true);
        // private_nh_.param<std::string>("depth_camera_info_topic", depth_camera_info_topic, "camera/depth/camera_info");
        // private_nh_.param<std::string>("depth_image_topic", depth_image_topic, "camera/depth/image_raw");
        // private_nh_.param<std::string>("color_image_topic", color_image_topic, "camera/image");
        private_nh_.param<std::string>("depth_camera_info_topic", depth_camera_info_topic, "");
        private_nh_.param<std::string>("depth_image_topic", depth_image_topic, "");
        private_nh_.param<std::string>("color_image_topic", color_image_topic, "");
        
        // Depth parameters
        private_nh_.param<std::string>("camera_frame", camera_frame_, "camera_color_optical_frame");
        private_nh_.param<float>("depth_scale", depth_scale_, 1.0);
        
        if (engine_path_.empty()) {
            ROS_ERROR("Engine file path not provided!");
            ros::shutdown();
            return;
        }
        
        // Load class labels
        if (!labels_path_.empty()) {
            read_labels(labels_path_, labels_map_);
            if (labels_map_.size() != kNumClass) {
                ROS_WARN("Number of classes in label file (%zu) doesn't match kNumClass (%d)", 
                        labels_map_.size(), kNumClass);
            }
        } else {
            ROS_WARN("No label file provided, detection boxes will have no class names");
        }
        
        // Initialize YOLO model
        deserialize_engine(engine_path_, &runtime_, &engine_, &context_);
        CUDA_CHECK(cudaStreamCreate(&stream_));
        cuda_preprocess_init(kMaxInputImageSize);
        
        auto out_dims = engine_->getBindingDimensions(1);
        model_bboxes_ = out_dims.d[0];
        
        // Prepare buffers
        prepare_buffer(engine_.get(), &device_buffers_[0], &device_buffers_[1], &device_buffers_[2],
                      &output_buffer_host_, &output_seg_buffer_host_, &decode_ptr_host_, 
                      &decode_ptr_device_, cuda_post_process_);
        
        // Initialize ROS subscriber and publisher
        image_sub_ = it_.subscribe(color_image_topic, 1, &YoloV8SegROSNode::imageCallback, this);
        depth_sub_ = it_.subscribe(depth_image_topic, 1, &YoloV8SegROSNode::depthCallback, this);
        camera_info_sub_ = nh_.subscribe(depth_camera_info_topic, 1, &YoloV8SegROSNode::cameraInfoCallback, this);
        
        if (publish_results_) {
            result_pub_ = it_.advertise("detection_result", 1);
        }
        pc_pub_ = nh_.advertise<sensor_msgs::PointCloud2>("masked_pointcloud", 10);
        centroid_pub_ = nh_.advertise<visualization_msgs::MarkerArray>("object_centroids", 1);
        
        ROS_INFO("YOLOv8 Segmentation ROS Node initialized successfully");
    }
    
    ~YoloV8SegROSNode() {
        freeResources();
    }

    void freeResources() {
        // shared_ptr 会自动管理资源，只需释放 CUDA 相关资源
        cudaStreamDestroy(stream_);
        CUDA_CHECK(cudaFree(device_buffers_[0]));
        CUDA_CHECK(cudaFree(device_buffers_[1]));
        CUDA_CHECK(cudaFree(device_buffers_[2]));
        CUDA_CHECK(cudaFree(decode_ptr_device_));
        delete[] decode_ptr_host_;
        delete[] output_buffer_host_;
        delete[] output_seg_buffer_host_;
        cuda_preprocess_destroy();
        
        ROS_INFO("YOLOv8 Segmentation ROS Node resources released");
    }

    void cameraInfoCallback(const sensor_msgs::CameraInfoConstPtr& msg) {
        if (!camera_info_received_) {
            // 初始化相机内参矩阵
            K_ = cv::Mat(3, 3, CV_32F);
            for (int i = 0; i < 9; ++i)
                K_.at<float>(i/3, i%3) = msg->K[i];

            camera_info_received_ = true;
            ROS_INFO_STREAM("相机内参矩阵:\n" << K_);
        }
    }

    void depthCallback(const sensor_msgs::ImageConstPtr& msg) {
        try {
            cv_bridge::CvImageConstPtr cv_ptr;
            if (msg->encoding == "16UC1") {
                cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_16UC1);
                depth_scale_ = 1000.0f; // 毫米转米
            } else if (msg->encoding == "32FC1") {
                cv_ptr = cv_bridge::toCvShare(msg, sensor_msgs::image_encodings::TYPE_32FC1);
                depth_scale_ = 1.0f;    // 已经是米
            } else {
                ROS_ERROR("不支持的深度图格式: %s", msg->encoding.c_str());
                return;
            }

            last_depth_image_ = cv_ptr->image.clone();
        }
        catch (const std::exception& e) {
            ROS_ERROR("处理深度图异常: %s", e.what());
        }
    }

    void imageCallback(const sensor_msgs::ImageConstPtr &msg) {
        try {
            if (!camera_info_received_) {
                ROS_WARN_THROTTLE(1.0, "等待相机内参...");
                return;
            }

            if (last_depth_image_.empty()) {
                ROS_WARN_THROTTLE(1.0, "等待深度图...");
                return;
            }

            cv::Mat image = cv_bridge::toCvShare(msg, "bgr8")->image;
            cv::Mat rgb_image = cv_bridge::toCvShare(msg, "bgr8")->image;
            
            // Clear batch and add new image
            img_batch_.clear();
            img_batch_.push_back(image.clone());
            
            // Preprocess
            cuda_batch_preprocess(img_batch_, device_buffers_[0], kInputW, kInputH, stream_);
            
            // Run inference
            auto start = std::chrono::high_resolution_clock::now();
            infer(*context_.get(), stream_, (void**)device_buffers_, output_buffer_host_, 
                output_seg_buffer_host_, kBatchSize, decode_ptr_host_, 
                decode_ptr_device_, model_bboxes_, cuda_post_process_);
            
            std::vector<std::vector<Detection>> res_batch;
            std::vector<cv::Mat> result_images;
            
            if (cuda_post_process_ == "c") {
                // CPU post-processing
                batch_nms(res_batch, output_buffer_host_, img_batch_.size(), kOutputSize, kConfThresh, kNmsThresh);
                
                // Process each image in batch
                for (size_t b = 0; b < img_batch_.size(); b++) {
                    auto& res = res_batch[b];
                    cv::Mat result_img = img_batch_[b].clone();
                    
                    // Process segmentation masks
                    auto masks = process_mask(&output_seg_buffer_host_[b * kOutputSegSize], kOutputSegSize, res);

                    // Process masked point cloud
                    processMaskedPointCloud(res, masks, last_depth_image_, result_img, msg->header, labels_map_);
                    
                    // Draw results
                    draw_mask_bbox(result_img, res, masks, labels_map_);
                    result_images.push_back(result_img);
                    
                    if (debug_) {
                        cv::imshow("YOLOv8 Segmentation", result_img);
                        cv::waitKey(1);
                    }
                }
            } else if (cuda_post_process_ == "g") {
                ROS_WARN_ONCE("GPU post-processing for segmentation is not yet supported");
                // GPU post-processing would go here
            }
            
            auto end = std::chrono::high_resolution_clock::now();
            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
            
            if (debug_) {
                ROS_INFO("Processing time: %ld ms", duration);
            }
            
            // Publish results if enabled
            if (publish_results_ && !result_images.empty()) {
                cv_bridge::CvImage out_msg;
                out_msg.header = msg->header;
                out_msg.encoding = sensor_msgs::image_encodings::BGR8;
                out_msg.image = result_images[0];
                result_pub_.publish(out_msg.toImageMsg());
            }
        }
        catch (cv_bridge::Exception &e) {
            ROS_ERROR("Failed to convert ROS image to OpenCV image: %s", e.what());
        }
        catch (const std::exception& e) {
            ROS_ERROR("Exception in image callback: %s", e.what());
        }
    }

    void processMaskedPointCloud(const std::vector<Detection>& dets, const std::vector<cv::Mat>& masks, 
                            const cv::Mat& depth_image, const cv::Mat& rgb_image,
                            const std_msgs::Header& header, std::unordered_map<int, std::string>& labels_map) {
        if (dets.empty() || masks.empty() || rgb_image.empty()) return;

        cv::Size depth_size = depth_image.size();
        cv::Mat rgb_image_copy = rgb_image.clone();
        float bbox[4];

        // 准备MarkerArray (先清除之前的标记)
        visualization_msgs::MarkerArray centroid_markers;
        visualization_msgs::Marker clear_marker;
        clear_marker.action = visualization_msgs::Marker::DELETEALL;
        centroid_markers.markers.push_back(clear_marker);
        
        // 为每个物体准备点云
        std::vector<pcl::PointCloud<pcl::PointXYZRGB>::Ptr> object_clouds(dets.size());
        pcl::PointCloud<pcl::PointXYZRGB>::Ptr merged_cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
        for (auto& cloud : object_clouds) {
            cloud.reset(new pcl::PointCloud<pcl::PointXYZRGB>);
        }

        // 转换参数准备
        float fx = K_.at<float>(0, 0);
        float fy = K_.at<float>(1, 1);
        float cx = K_.at<float>(0, 2);
        float cy = K_.at<float>(1, 2);
        float inv_depth_scale = 1.0f / depth_scale_;

        // 处理每个物体
        #pragma omp parallel for schedule(dynamic)
        for (size_t i = 0; i < dets.size(); i++) {
            cv::Mat img_mask = scale_mask(masks[i], rgb_image_copy);
            std::copy(dets[i].bbox, dets[i].bbox + 4, bbox);
            cv::Rect r = get_rect(rgb_image_copy, bbox);
            
            // 创建当前物体的掩码
            cv::Mat object_mask = cv::Mat::zeros(depth_size, CV_8UC1);
            for (int x = r.x; x < r.x + r.width; x++) {
                for (int y = r.y; y < r.y + r.height; y++) {
                    if (img_mask.at<float>(y, x) > 0.6f) {
                        object_mask.at<uint8_t>(y, x) = 255;
                    }
                }
            }
            
            // 转换为点云
            #pragma omp parallel for
            for (int v = 0; v < depth_image.rows; ++v) {
                for (int u = 0; u < depth_image.cols; ++u) {
                    if (object_mask.at<uint8_t>(v, u) == 0) continue;
                    
                    float depth;
                    if (depth_image.type() == CV_16UC1) {
                        depth = depth_image.at<uint16_t>(v, u) * inv_depth_scale;
                    } else {
                        depth = depth_image.at<float>(v, u) * inv_depth_scale;
                    }
                    
                    if (depth <= 0) continue;
                    
                    pcl::PointXYZRGB point;
                    point.z = depth;
                    point.x = (u - cx) * point.z / fx;
                    point.y = (v - cy) * point.z / fy;
                    
                    // 设置RGB颜色
                    cv::Vec3b color = rgb_image.at<cv::Vec3b>(v, u);
                    point.r = color[2];
                    point.g = color[1];
                    point.b = color[0];
                    
                    if(labels_map[(int)dets[i].class_id]=="bottle" || labels_map[(int)dets[i].class_id]=="mouse" || 
                        labels_map[(int)dets[i].class_id]=="cell phone"){
                    object_clouds[i]->push_back(point);
                    }
                }
            }
            
            // 使用PCL计算中心点
            if (!object_clouds[i]->empty()) {
                Eigen::Vector4f centroid;
                pcl::compute3DCentroid(*object_clouds[i], centroid);
                // 创建质心Marker
                visualization_msgs::Marker marker;
                marker.header = header;
                marker.ns = "centroids";
                marker.id = i;
                marker.type = visualization_msgs::Marker::SPHERE;
                marker.action = visualization_msgs::Marker::ADD;
                marker.pose.position.x = centroid[0];
                marker.pose.position.y = centroid[1];
                marker.pose.position.z = centroid[2];
                marker.pose.orientation.w = 1.0;
                
                // 可视化设置
                marker.scale.x = marker.scale.y = marker.scale.z = 0.05; // 10cm直径
                marker.color.r = 1.0; // 红色
                marker.color.a = 1.0; // 不透明
                marker.lifetime = ros::Duration(0.5); // 0.5秒后自动消失

                // 添加文本标签
                visualization_msgs::Marker text_marker = marker;
                text_marker.id = i + 1000;
                text_marker.type = visualization_msgs::Marker::TEXT_VIEW_FACING;
                // text_marker.text = "C" + std::to_string(i); // C表示Centroid
                text_marker.text = labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf);
                text_marker.scale.z = 0.1; // 文本大小
                text_marker.color.r = text_marker.color.g = text_marker.color.b = 1.0; // 白色
                text_marker.pose.position.z += 0.15; // 抬高文本

                centroid_markers.markers.push_back(marker);
                centroid_markers.markers.push_back(text_marker);

                ROS_INFO("Object %zu centroid: (%.3f, %.3f, %.3f)", 
                        i, centroid[0], centroid[1], centroid[2]);
            }
            *merged_cloud += *object_clouds[i];
        }
        
        // 发布点云 (可选)
        if (!object_clouds.empty()) {
            sensor_msgs::PointCloud2 cloud_msg;
            #pragma omp parallel for
            for (int l = 0; l < object_clouds.size(); ++l){
                auto grasp_pose = planner_.computeGraspPose(object_clouds[l], 0.10);
            }
            // pcl::toROSMsg(*object_clouds[0], cloud_msg); // 发布第一个物体或合并所有
            pcl::toROSMsg(*merged_cloud, cloud_msg);
            cloud_msg.header = header;
            cloud_msg.header.frame_id = camera_frame_;
            pc_pub_.publish(cloud_msg);
        }
         // 发布质心MarkerArray
        if (centroid_markers.markers.size() > 1) { // 大于1是因为有DELETEALL
            centroid_pub_.publish(centroid_markers);
        }
    }
    
    void run() {
        ros::Rate r(30); // 30 Hz
        while(ros::ok()) {
            ros::spinOnce();
            r.sleep();
        }
    }

private:
    static cv::Rect get_downscale_rect(float bbox[4], float scale) {
        float left = bbox[0];
        float top = bbox[1];
        float right = bbox[0] + bbox[2];
        float bottom = bbox[1] + bbox[3];

        left = left < 0 ? 0 : left;
        top = top < 0 ? 0 : top;
        right = right > kInputW ? kInputW : right;
        bottom = bottom > kInputH ? kInputH : bottom;

        left /= scale;
        top /= scale;
        right /= scale;
        bottom /= scale;
        return cv::Rect(int(left), int(top), int(right - left), int(bottom - top));
    }

    std::vector<cv::Mat> process_mask(const float* proto, int proto_size, std::vector<Detection>& dets) {
    std::vector<cv::Mat> masks(dets.size());
    
    // 并行处理每个检测
    cv::parallel_for_(cv::Range(0, dets.size()), [&](const cv::Range& range) {
            for (int i = range.start; i < range.end; ++i) {
                cv::Mat mask_mat = cv::Mat::zeros(kInputH / 4, kInputW / 4, CV_32FC1);
                auto r = get_downscale_rect(dets[i].bbox, 4);

                // 优化内存访问模式
                for (int y = r.y; y < r.y + r.height; y++) {
                    float* pMask = mask_mat.ptr<float>(y);
                    for (int x = r.x; x < r.x + r.width; x++) {
                        float e = 0.0f;
                        const float* pProto = proto + x + y * mask_mat.cols;
                        for (int j = 0; j < 32; j++) {
                            e += dets[i].mask[j] * pProto[j * proto_size / 32];
                        }
                        pMask[x] = 1.0f / (1.0f + expf(-e));
                    }
                }
                
                cv::resize(mask_mat, masks[i], cv::Size(kInputW, kInputH));
            }
        });
        
        return masks;
    }

    void deserialize_engine(std::string& engine_name, 
                       std::shared_ptr<IRuntime>* runtime, 
                       std::shared_ptr<ICudaEngine>* engine,
                       std::shared_ptr<IExecutionContext>* context) {
        std::ifstream file(engine_name, std::ios::binary);
        if (!file.good()) {
            ROS_ERROR("Failed to read engine file: %s", engine_name.c_str());
            throw std::runtime_error("Failed to read engine file");
        }
        
        size_t size = 0;
        file.seekg(0, file.end);
        size = file.tellg();
        file.seekg(0, file.beg);
        std::vector<char> serialized_engine(size);
        file.read(serialized_engine.data(), size);
        file.close();

        // 创建 runtime 并设置删除器
        runtime->reset(createInferRuntime(gLogger), 
                    [](IRuntime* ptr) { if (ptr) ptr->destroy(); });
        if (!*runtime) {
            throw std::runtime_error("Failed to create InferRuntime");
        }
        
        // 创建 engine 并设置删除器
        engine->reset((*runtime)->deserializeCudaEngine(serialized_engine.data(), size),
                [](ICudaEngine* ptr) { if (ptr) ptr->destroy(); });
        if (!*engine) {
            throw std::runtime_error("Failed to deserialize CUDA engine");
        }
        
        // 创建 context 并设置删除器
        context->reset((*engine)->createExecutionContext(),
                    [](IExecutionContext* ptr) { if (ptr) ptr->destroy(); });
        if (!*context) {
            throw std::runtime_error("Failed to create execution context");
        }
    }

    void prepare_buffer(ICudaEngine* engine, float** input_buffer_device, float** output_buffer_device,
                       float** output_seg_buffer_device, float** output_buffer_host, float** output_seg_buffer_host,
                       float** decode_ptr_host, float** decode_ptr_device, std::string cuda_post_process) {
        assert(engine->getNbBindings() == 3);
        const int inputIndex = engine->getBindingIndex(kInputTensorName);
        const int outputIndex = engine->getBindingIndex(kOutputTensorName);
        const int outputIndex_seg = engine->getBindingIndex("proto");
        
        assert(inputIndex == 0);
        assert(outputIndex == 1);
        assert(outputIndex_seg == 2);
        
        // Create GPU buffers on device
        CUDA_CHECK(cudaMalloc((void**)input_buffer_device, kBatchSize * 3 * kInputH * kInputW * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)output_buffer_device, kBatchSize * kOutputSize * sizeof(float)));
        CUDA_CHECK(cudaMalloc((void**)output_seg_buffer_device, kBatchSize * kOutputSegSize * sizeof(float)));
        
        if (cuda_post_process == "c") {
            *output_buffer_host = new float[kBatchSize * kOutputSize];
            *output_seg_buffer_host = new float[kBatchSize * kOutputSegSize];
        } else if (cuda_post_process == "g") {
            if (kBatchSize > 1) {
                ROS_ERROR("GPU post processing not supported for multiple batches");
                throw std::runtime_error("GPU post processing not supported for multiple batches");
            }
            *decode_ptr_host = new float[1 + kMaxNumOutputBbox * bbox_element];
            CUDA_CHECK(cudaMalloc((void**)decode_ptr_device, sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element)));
        }
    }

    void infer(IExecutionContext& context, cudaStream_t& stream, void** buffers, float* output, float* output_seg,
               int batchsize, float* decode_ptr_host, float* decode_ptr_device, int model_bboxes,
               std::string cuda_post_process) {
        context.enqueue(batchsize, buffers, stream, nullptr);
        
        if (cuda_post_process == "c") {
            CUDA_CHECK(cudaMemcpyAsync(output, buffers[1], batchsize * kOutputSize * sizeof(float), 
                                     cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(output_seg, buffers[2], batchsize * kOutputSegSize * sizeof(float),
                                     cudaMemcpyDeviceToHost, stream));
        } else if (cuda_post_process == "g") {
            CUDA_CHECK(cudaMemsetAsync(decode_ptr_device, 0, 
                                     sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), stream));
            cuda_decode((float*)buffers[1], model_bboxes, kConfThresh, decode_ptr_device, kMaxNumOutputBbox, stream);
            cuda_nms(decode_ptr_device, kNmsThresh, kMaxNumOutputBbox, stream);
            CUDA_CHECK(cudaMemcpyAsync(decode_ptr_host, decode_ptr_device,
                                     sizeof(float) * (1 + kMaxNumOutputBbox * bbox_element), 
                                     cudaMemcpyDeviceToHost, stream));
        }

        CUDA_CHECK(cudaStreamSynchronize(stream));
    }

    cv::Mat scale_mask(cv::Mat mask, cv::Mat img) {
        int x, y, w, h;
        float r_w = kInputW / (img.cols * 1.0);
        float r_h = kInputH / (img.rows * 1.0);
        if (r_h > r_w) {
            w = kInputW;
            h = r_w * img.rows;
            x = 0;
            y = (kInputH - h) / 2;
        } else {
            w = r_h * img.cols;
            h = kInputH;
            x = (kInputW - w) / 2;
            y = 0;
        }
        cv::Rect r(x, y, w, h);
        cv::Mat res;
        cv::resize(mask(r), res, img.size());
        return res;
    }

    void draw_mask_bbox(cv::Mat& img, std::vector<Detection>& dets, std::vector<cv::Mat>& masks,
                    std::unordered_map<int, std::string>& labels_map) {
        static std::vector<uint32_t> colors = {0xFF3838, 0xFF9D97, 0xFF701F, 0xFFB21D, 0xCFD231, 0x48F90A, 0x92CC17,
                                              0x3DDB86, 0x1A9334, 0x00D4BB, 0x2C99A8, 0x00C2FF, 0x344593, 0x6473FF,
                                              0x0018EC, 0x8438FF, 0x520085, 0xCB38FF, 0xFF95C8, 0xFF37C7};
        #pragma omp parallel for
        for (size_t i = 0; i < dets.size(); i++) {
            cv::Mat img_mask = scale_mask(masks[i], img);
            auto color = colors[(int)dets[i].class_id % colors.size()];
            auto bgr = cv::Scalar(color & 0xFF, color >> 8 & 0xFF, color >> 16 & 0xFF);
            cv::Rect r = get_rect(img, dets[i].bbox);
            for (int x = r.x; x < r.x + r.width; x++) {
                for (int y = r.y; y < r.y + r.height; y++) {
                    float val = img_mask.at<float>(y, x);
                    if (val <= 0.5)
                        continue;
                    img.at<cv::Vec3b>(y, x)[0] = img.at<cv::Vec3b>(y, x)[0] / 2 + bgr[0] / 2;
                    img.at<cv::Vec3b>(y, x)[1] = img.at<cv::Vec3b>(y, x)[1] / 2 + bgr[1] / 2;
                    img.at<cv::Vec3b>(y, x)[2] = img.at<cv::Vec3b>(y, x)[2] / 2 + bgr[2] / 2;
                }
            }

            cv::rectangle(img, r, bgr, 2);

            // Get the size of the text
            cv::Size textSize =
                    cv::getTextSize(labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                                    cv::FONT_HERSHEY_PLAIN, 1.2, 2, NULL);
            // Set the top left corner of the rectangle
            cv::Point topLeft(r.x, r.y - textSize.height);

            // Set the bottom right corner of the rectangle
            cv::Point bottomRight(r.x + textSize.width, r.y + textSize.height);

            // Set the thickness of the rectangle lines
            int lineThickness = 2;

            // Draw the rectangle on the image
            cv::rectangle(img, topLeft, bottomRight, bgr, -1);

            cv::putText(img, labels_map[(int)dets[i].class_id] + " " + to_string_with_precision(dets[i].conf),
                        cv::Point(r.x, r.y + 4), cv::FONT_HERSHEY_PLAIN, 1.2, cv::Scalar::all(0xFF), 2);
        }
    }
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "yolov8_seg_node");
    try {
        YoloV8SegROSNode node;
        node.run();
    } catch (const std::exception& e) {
        ROS_ERROR("Exception in YoloV8SegROSNode: %s", e.what());
        return 1;
    }
    return 0;
}