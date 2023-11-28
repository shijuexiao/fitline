#pragma once
/*****************************************************************//**
 * \file   fitline.h
 * \brief 卡尺拟合线工具，亚像素

 * \author WYJ
 * \date 2023-8-7
 *********************************************************************/
 /* 调用示例：
 * int main()
 {
	 Fvision::cvfunc::fitline fiteline;
	 std::string str = R"(C:\Users\Administrator\Desktop\23.PNG)";
	 cv::Mat src, dst;
	 src = cv::imread(str);
	 cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	 Fvision::cvfunc::fitline::fitLineParams lineparams;
	 lineparams.p1 = cv::Point2d(351, 192);//直线起点
	 lineparams.p2 = cv::Point2d(299, 336);//直线终点
	 lineparams.edgethreshold = 30;//幅度阈值
	 lineparams.edge_polarity = 0;//边缘极性
	 lineparams.edge_type = 0;//边缘选择
	 lineparams.halfheight = 30;//仿射矩形半高度
	 lineparams.halfwidth = 3;//仿射矩形半宽度
	 lineparams.segNum = 15;//卡尺数量
	 lineparams.num_instances = 1;//拟合实例个数
	 auto start = std::chrono::steady_clock::now();
	 std::vector<Fvision::cvfunc::fitline::lineResult> outline;
	 Fvision::cvfunc::fitline::edgePointsRes edges;
	 fiteline.findLine(src, lineparams, outline, edges);
	 auto end = std::chrono::steady_clock::now();
	 double ellipse2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	 std::cout << "找线耗时:" << ellipse2 << "ms" << std::endl;
	 for (auto& line : outline)
	 {
		 cv::line(src, line.p1, line.p2, cv::Scalar(0, 255, 0), 1, 8, 0);
	 }
 }*/
#include<vector>
#include<opencv2/opencv.hpp>
#ifdef MYCVDLL
#define MYCVAPI __declspec(dllexport)
#else
#define MYCVAPI  __declspec(dllimport)
#endif
namespace Fvision {
	namespace cvfunc
	{
		class MYCVAPI fitline {
		public:
			/**
			 * 线拟合输出结果.
			 */
			struct lineResult
			{
				cv::Point2d p1{};//起始点
				cv::Point2d p2{};//终点
				cv::Point2d p3{};//直线上一点
				double angle{ 0 };//角度[-pi,pi]顺时针为正
				double score{ 0 };//得分
				double k{ 0 };//斜率
				double b{ 0 };//截距
				std::vector<cv::Point2d> edgePoints;//点集（有效点）
				std::vector<cv::Point2d> invalidedgePoints;//点集（无效点）
			};
			/**
			 * 边缘点集和对应幅度值.
			 */
			struct edgePointsRes
			{
				std::vector<cv::Point2d> edgePoints;//点集
				std::vector<float> amplitude;//幅度

			};
			/**
			 * 卡尺拟合线参数.
			 */
			struct fitLineParams
			{
				cv::Point2d p1{};//起始点
				cv::Point2d p2{};//终点
				int segNum{ 8 };//卡尺数量
				double sigma{ 1.0 };//平滑
				double halfheight{ 10.0 };//卡尺半高度
				double halfwidth{ 3.0 };//卡尺半宽度
				int edgethreshold{ 5 };//最小边缘幅度
				int edge_type{ 0 };//边缘类型，参考measure_select说明
				int edge_polarity{ 0 };//边缘极性，参考measure_transition说明
				double score = 0.7;//最小得分，分值=用于计算的边缘点数/卡尺数量
				int num_instances = 1;//找到的实例最大个数（最多两个），实际上限制为正极性、负极性两种
				int max_num_iterations = -1;//执行RANSAC算法的最大迭代次数，默认不限制
				double distance_threshold = 3.5;//使用随机搜索 算法 （RANSAC） 以拟合几何形状。如果点到几何形状的距离<distance_threshold，认为该点符合预期
			};
		private:
			/**
			 * 边缘极性.
			 */
			enum class measure_transition
			{
				all = 0,//所有极性
				positive,//正极性，从黑到白
				negative//负极性，从白到黑
			};
			/**
			 * 边缘选择.
			 */
			enum class measure_select
			{
				all = 0,//所有边缘
				first,//第一个边缘
				last,//最后一个边缘
				best//最强幅度的边缘
			};
			/**
			 * 边缘点集.
			 */
			struct edgePoints
			{
				std::vector<cv::Point2d> edgePoints_p;//正极性
				std::vector<float> amplitude_p;//正极性幅度
				std::vector<cv::Point2d> edgePoints_n;//负极性
				std::vector<float> amplitude_n;//负极性幅度
			};

		public:
			fitline();
			~fitline();
			/**
			* \functionName  findLine
			* \brief 找线工具.
			*
			* \param src：输入灰度图
			* \param lineparams：输入直线参数
			* \param outline：输出直线结果
			* \param edges：输出边缘点及边缘幅度
			*/
			void findLine(cv::Mat& src, fitLineParams& lineparams, std::vector<lineResult>& outline, edgePointsRes& edges);
			/**
			 * \functionName  drawLineCalipers
			 * \brief 绘制直线卡尺.
			 *
			 * \param src：输入图像
			 * \param lineparams：输入直线信息
			 */
			void drawLineCalipers(cv::Mat& src, fitLineParams& lineparams);
		private:
			class impl;
			std::unique_ptr<impl>impl_;
		};
	}
}