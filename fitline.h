#pragma once
/*****************************************************************//**
 * \file   fitline.h
 * \brief ��������߹��ߣ�������

 * \author WYJ
 * \date 2023-8-7
 *********************************************************************/
 /* ����ʾ����
 * int main()
 {
	 Fvision::cvfunc::fitline fiteline;
	 std::string str = R"(C:\Users\Administrator\Desktop\23.PNG)";
	 cv::Mat src, dst;
	 src = cv::imread(str);
	 cv::cvtColor(src, dst, cv::COLOR_BGR2GRAY);
	 Fvision::cvfunc::fitline::fitLineParams lineparams;
	 lineparams.p1 = cv::Point2d(351, 192);//ֱ�����
	 lineparams.p2 = cv::Point2d(299, 336);//ֱ���յ�
	 lineparams.edgethreshold = 30;//������ֵ
	 lineparams.edge_polarity = 0;//��Ե����
	 lineparams.edge_type = 0;//��Եѡ��
	 lineparams.halfheight = 30;//������ΰ�߶�
	 lineparams.halfwidth = 3;//������ΰ���
	 lineparams.segNum = 15;//��������
	 lineparams.num_instances = 1;//���ʵ������
	 auto start = std::chrono::steady_clock::now();
	 std::vector<Fvision::cvfunc::fitline::lineResult> outline;
	 Fvision::cvfunc::fitline::edgePointsRes edges;
	 fiteline.findLine(src, lineparams, outline, edges);
	 auto end = std::chrono::steady_clock::now();
	 double ellipse2 = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	 std::cout << "���ߺ�ʱ:" << ellipse2 << "ms" << std::endl;
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
			 * �����������.
			 */
			struct lineResult
			{
				cv::Point2d p1{};//��ʼ��
				cv::Point2d p2{};//�յ�
				cv::Point2d p3{};//ֱ����һ��
				double angle{ 0 };//�Ƕ�[-pi,pi]˳ʱ��Ϊ��
				double score{ 0 };//�÷�
				double k{ 0 };//б��
				double b{ 0 };//�ؾ�
				std::vector<cv::Point2d> edgePoints;//�㼯����Ч�㣩
				std::vector<cv::Point2d> invalidedgePoints;//�㼯����Ч�㣩
			};
			/**
			 * ��Ե�㼯�Ͷ�Ӧ����ֵ.
			 */
			struct edgePointsRes
			{
				std::vector<cv::Point2d> edgePoints;//�㼯
				std::vector<float> amplitude;//����

			};
			/**
			 * ��������߲���.
			 */
			struct fitLineParams
			{
				cv::Point2d p1{};//��ʼ��
				cv::Point2d p2{};//�յ�
				int segNum{ 8 };//��������
				double sigma{ 1.0 };//ƽ��
				double halfheight{ 10.0 };//���߰�߶�
				double halfwidth{ 3.0 };//���߰���
				int edgethreshold{ 5 };//��С��Ե����
				int edge_type{ 0 };//��Ե���ͣ��ο�measure_select˵��
				int edge_polarity{ 0 };//��Ե���ԣ��ο�measure_transition˵��
				double score = 0.7;//��С�÷֣���ֵ=���ڼ���ı�Ե����/��������
				int num_instances = 1;//�ҵ���ʵ���������������������ʵ��������Ϊ�����ԡ�����������
				int max_num_iterations = -1;//ִ��RANSAC�㷨��������������Ĭ�ϲ�����
				double distance_threshold = 3.5;//ʹ��������� �㷨 ��RANSAC�� ����ϼ�����״������㵽������״�ľ���<distance_threshold����Ϊ�õ����Ԥ��
			};
		private:
			/**
			 * ��Ե����.
			 */
			enum class measure_transition
			{
				all = 0,//���м���
				positive,//�����ԣ��Ӻڵ���
				negative//�����ԣ��Ӱ׵���
			};
			/**
			 * ��Եѡ��.
			 */
			enum class measure_select
			{
				all = 0,//���б�Ե
				first,//��һ����Ե
				last,//���һ����Ե
				best//��ǿ���ȵı�Ե
			};
			/**
			 * ��Ե�㼯.
			 */
			struct edgePoints
			{
				std::vector<cv::Point2d> edgePoints_p;//������
				std::vector<float> amplitude_p;//�����Է���
				std::vector<cv::Point2d> edgePoints_n;//������
				std::vector<float> amplitude_n;//�����Է���
			};

		public:
			fitline();
			~fitline();
			/**
			* \functionName  findLine
			* \brief ���߹���.
			*
			* \param src������Ҷ�ͼ
			* \param lineparams������ֱ�߲���
			* \param outline�����ֱ�߽��
			* \param edges�������Ե�㼰��Ե����
			*/
			void findLine(cv::Mat& src, fitLineParams& lineparams, std::vector<lineResult>& outline, edgePointsRes& edges);
			/**
			 * \functionName  drawLineCalipers
			 * \brief ����ֱ�߿���.
			 *
			 * \param src������ͼ��
			 * \param lineparams������ֱ����Ϣ
			 */
			void drawLineCalipers(cv::Mat& src, fitLineParams& lineparams);
		private:
			class impl;
			std::unique_ptr<impl>impl_;
		};
	}
}