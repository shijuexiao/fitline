#include"fitline.h"

namespace Fvision {
	namespace cvfunc
	{
		class fitline::impl
		{
		public:
			void findLine(cv::Mat& src, fitLineParams& lineparams, std::vector<lineResult>& outline, edgePointsRes& edges)
			{
				if (src.empty() || lineparams.p1 == lineparams.p2 || lineparams.halfwidth < 1 || lineparams.halfheight < 1 || lineparams.sigma > 0.5 * lineparams.halfheight || lineparams.sigma < 0.4)
				{
					std::cout << "以上参数错误!" << std::endl;
					std::runtime_error err("以上参数错误!");
					return;
				}
				outline.clear();
				edges.amplitude.clear();
				edges.edgePoints.clear();
				cv::Mat gray, srctem;
				double distance;
				distance = cv::pow((lineparams.p1.x - lineparams.p2.x), 2) + pow((lineparams.p1.y - lineparams.p2.y), 2);
				distance = sqrt(distance);
				double disttem = distance;
				double deltw = MAX(lineparams.halfheight, lineparams.halfwidth);
				double dist = disttem + deltw;
				double w = lineparams.halfheight * 2 + deltw;
				double linangle = angle_lx(lineparams.p1, lineparams.p2);
				cv::Point2d center = (lineparams.p1 + lineparams.p2) / 2.;
				//旋转矩形0°为Y正方向（垂直向上）
				cv::RotatedRect rect(center, cv::Size(w, dist), linangle + 90);
				//映射后的矩形
				cv::Point2f transRoi[4];
				transRoi[0] = cv::Point2f(0, 0);
				transRoi[1] = cv::Point2f(0, dist);
				transRoi[2] = cv::Point2f(w, dist);
				transRoi[3] = cv::Point2f(w, 0);
				std::vector<float>values_p;//符合条件的正极性幅度值
				std::vector<double>indexs_p;//符合条件的正极性索引
				std::vector<float>values_n;//符合条件的负极性幅度值
				std::vector<double>indexs_n;//符合条件的负极性索引
					//仿射矩形映射到指定矩形
				cv::Point2f vertex[4];
				rect.points(vertex);
				cv::Mat transMatrix;
				cv::Mat invMatrix;
#if 0
				transMatrix = cv::getPerspectiveTransform(vertex, transRoi);
				cv::invertAffineTransform(transMatrix, invMatrix);
				cv::warpPerspective(src, srctem, transMatrix, cv::Size(w, dist), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
#else
				transMatrix = cv::getAffineTransform(vertex, transRoi);
				cv::invertAffineTransform(transMatrix, invMatrix);
				cv::warpAffine(src, srctem, transMatrix, cv::Size(w, dist), cv::INTER_LINEAR, cv::BORDER_REPLICATE);
#endif // 0
				if (srctem.channels() > 1)
				{
					cv::cvtColor(srctem, gray, cv::COLOR_BGR2GRAY);
				}
				else
				{
					gray = srctem;
				}

				cv::Mat transPointsInv;
				cv::invertAffineTransform(invMatrix, transPointsInv);
				invMatrix.at<double>(0, 2) += 0.5;
				invMatrix.at<double>(1, 2) += 0.5;
				std::vector<cv::Point2d>linepoints, linepoints_tem;
				linepoints.push_back(lineparams.p1);
				linepoints.push_back(lineparams.p2);
				cv::transform(linepoints, linepoints_tem, transPointsInv);
				lineparams.p1 = linepoints_tem[0];
				lineparams.p2 = linepoints_tem[1];
				std::vector<cv::Rect>rect2;
				createRect2(lineparams, rect2);
				//使用完后赋值回来
				lineparams.p1 = linepoints[0];
				lineparams.p2 = linepoints[1];
				edgePoints edgestem;
				getEdgePoints(gray, lineparams, rect2, edgestem);
				if (edgestem.edgePoints_p.size() > 0)
				{
					cv::transform(edgestem.edgePoints_p, edgestem.edgePoints_p, invMatrix);
				}
				if (edgestem.edgePoints_n.size() > 0)
				{
					cv::transform(edgestem.edgePoints_n, edgestem.edgePoints_n, invMatrix);
				}
				std::vector<cv::Point2d> realPoints_p;
				std::vector<cv::Point2d> realPoints_n;
				std::vector<cv::Point2d> invalidrealPoints_p;
				std::vector<cv::Point2d> invalidrealPoints_n;
				RansacLineFiler(edgestem.edgePoints_p, realPoints_p, invalidrealPoints_p, lineparams.segNum, lineparams.score, lineparams.max_num_iterations, lineparams.distance_threshold);
				RansacLineFiler(edgestem.edgePoints_n, realPoints_n, invalidrealPoints_n, lineparams.segNum, lineparams.score, lineparams.max_num_iterations, lineparams.distance_threshold);

				edges.edgePoints.insert(edges.edgePoints.begin(), edgestem.edgePoints_p.begin(), edgestem.edgePoints_p.end());
				edges.edgePoints.insert(edges.edgePoints.end(), edgestem.edgePoints_n.begin(), edgestem.edgePoints_n.end());
				edges.amplitude.insert(edges.amplitude.begin(), edgestem.amplitude_p.begin(), edgestem.amplitude_p.end());
				edges.amplitude.insert(edges.amplitude.end(), edgestem.amplitude_n.begin(), edgestem.amplitude_n.end());

				measure_transition transition = static_cast<measure_transition>(lineparams.edge_polarity);
				switch (transition)
				{
				case measure_transition::all:
					getlineResults(lineparams, realPoints_p, invalidrealPoints_p, outline);
					getlineResults(lineparams, realPoints_n, invalidrealPoints_n, outline);
					break;
				case measure_transition::positive:
					getlineResults(lineparams, realPoints_p, invalidrealPoints_p, outline);
					break;
				case measure_transition::negative:
					getlineResults(lineparams, realPoints_n, invalidrealPoints_n, outline);
					break;
				default:
					break;
				}
				if (lineparams.num_instances == 1 && outline.size() > 1)
				{
					outline.resize(1);
				}
			}
			void drawLineCalipers(cv::Mat& src, fitLineParams& lineparams)
			{
				cv::Mat dst;
				if (src.channels() < 3)
				{
					cv::cvtColor(src, dst, cv::COLOR_GRAY2BGR);
				}
				else
				{
					dst = src;
				}
				//绘制卡尺
				cv::Point2d p1 = lineparams.p1;
				cv::Point2d p2 = lineparams.p2;
				int rectnum = lineparams.segNum;
				int rectw = std::ceil(lineparams.halfwidth * 2);
				int recth = std::ceil(lineparams.halfheight * 2);
				double deltx = p1.x - p2.x;
				double delty = p1.y - p2.y;
				double avgx = (1. * deltx / rectnum);
				double avgy = (1. * delty / rectnum);
				double linangle = angle_lx(p1, p2);
				for (int i = 1; i < rectnum + 1; i++)
				{
					cv::Point2d center(p1.x - avgx * (i - 0.5), p1.y - avgy * (i - 0.5));
					//旋转矩形0°为Y正方向（垂直向上）
					cv::RotatedRect rect(center, cv::Size(rectw, recth), linangle + 180);
					//绘制卡尺
					cv::Point2f vertex[4];
					rect.points(vertex);
					for (int i = 0; i < 4; i++)
					{
						//cv::putText(src, std::to_string(i), vertex[i], 0, 0.3, cv::Scalar(0, 0, 255));
						//cv::circle(src, vertex[i], 1, cv::Scalar(0, 0, 255));
						cv::line(dst, vertex[i], vertex[(i + 1) % 4], cv::Scalar(255, 100, 200), 1);
					}
					double radian = ((linangle + 90) * CV_PI) / 180;
					cv::Point2d endpoint(center.x + recth * cos(radian), center.y + recth * sin(radian));
					cv::arrowedLine(dst, center, endpoint, cv::Scalar(0, 0, 255));
					cv::arrowedLine(dst, p1, p2, cv::Scalar(255, 0, 0));
				}
			}
		private:
			void getlineResults(fitLineParams& lineparams, std::vector<cv::Point2d>& points, std::vector<cv::Point2d>& invalidedgePoints, std::vector<lineResult>& result)
			{
				int num = points.size();
				if (num < 2)
				{
					return;
				}
				lineResult temResult;
				cv::Vec4f line_para;
				cv::fitLine(points, line_para, cv::DistanceTypes::DIST_L2, 0, 0.01, 0.01);
				//获取点斜式的点和斜率
				cv::Point2d point0;
				point0.x = line_para[2];
				point0.y = line_para[3];
				double k = line_para[1] / line_para[0];
				double b = point0.y - k * point0.x;
				//计算人为设置的直线的端点(y = k(x - x0) + y0)
				cv::Point2d point1, point2;
				if (abs(k) < 0.1)
				{
					point1.y = 1.0 * (lineparams.p1.x - point0.x) * k + point0.y;
					point1.x = 1. * lineparams.p1.x;
					point2.y = 1.0 * (lineparams.p2.x - point0.x) * k + point0.y;
					point2.x = 1. * lineparams.p2.x;

				}
				else
				{
					point1.x = 1.0 * (lineparams.p1.y - point0.y) / k + point0.x;
					point1.y = 1. * lineparams.p1.y;
					point2.x = 1.0 * (lineparams.p2.y - point0.y) / k + point0.x;
					point2.y = 1. * lineparams.p2.y;
				}

				double angle = angle_lx(point1, point2);
				temResult.p1 = point1;
				temResult.p2 = point2;
				temResult.angle = angle;
				temResult.p3 = point0;
				temResult.k = k;
				temResult.b = b;
				double scr = 1. * num / lineparams.segNum;
				temResult.score = MIN(1., scr);
				temResult.edgePoints.insert(temResult.edgePoints.begin(), points.begin(), points.end());
				temResult.invalidedgePoints.insert(temResult.invalidedgePoints.begin(), invalidedgePoints.begin(), invalidedgePoints.end());
				result.push_back(temResult);
			}
			double getsubpix(std::vector<cv::Point2d>& points, float* amptem)
			{
				//抛物线顶点式：y=a(x-b)^2+c
				if (points.size() < 3)
				{
					return -1;
				}
				cv::Mat matrix = cv::Mat_<float>(3, 3);
				matrix.at<float>(0, 0) = cv::pow(points[0].x, 2);
				matrix.at<float>(1, 0) = points[0].x;
				matrix.at<float>(2, 0) = 1;
				matrix.at<float>(0, 1) = cv::pow(points[1].x, 2);
				matrix.at<float>(1, 1) = points[1].x;
				matrix.at<float>(2, 1) = 1;
				matrix.at<float>(0, 2) = cv::pow(points[2].x, 2);
				matrix.at<float>(1, 2) = points[2].x;
				matrix.at<float>(2, 2) = 1;
				cv::Mat invMatrix;
				cv::invert(matrix, invMatrix, 1);
				cv::Mat matrix_y = cv::Mat_<float>(1, 3);
				matrix_y.at<float>(0, 0) = points[0].y;
				matrix_y.at<float>(0, 1) = points[1].y;
				matrix_y.at<float>(0, 2) = points[2].y;
				cv::Mat result = cv::Mat_<float>(1, 3);
				result = matrix_y * invMatrix;
				double a = result.at<float>(0, 0);
				double b = -0.5 * result.at<float>(0, 1) / a;
				float c = result.at<float>(0, 2) - a * b * b;
				*amptem = c;
				return b;
			}
			void calsubpix(cv::Mat& sobelimg1, std::vector<double>& indexs, std::vector<float>& values)
			{
				int num = indexs.size();
				if (num < 1)
				{
					return;
				}
				for (int i = 0; i < num; i++)
				{
					std::vector<cv::Point2d> points;
					double sobelindex = indexs[i];
					float val1 = sobelimg1.at<float>(0, sobelindex - 1);
					float val2 = sobelimg1.at<float>(0, sobelindex);
					float val3 = sobelimg1.at<float>(0, sobelindex + 1);
					points.emplace_back(cv::Point2d(sobelindex - 1., val1));
					points.emplace_back(cv::Point2d(sobelindex, val2));
					points.emplace_back(cv::Point2d(sobelindex + 1., val3));
					float tem;
					if (val1 == val2 && val1 == val3)
					{
						//拐点
						indexs[i] = sobelindex;
						values[i] = val1;
					}
					else
					{
						indexs[i] = getsubpix(points, &tem);
						values[i] = tem;
					}
				}
			}
			void getEdgePoints(cv::Mat& gray, fitLineParams& lineparams, std::vector<cv::Rect>& rect2, edgePoints& edges)
			{
				measure_select select = static_cast<measure_select>(lineparams.edge_type);

				int rect2_num = rect2.size();
				//一阶核
				cv::Mat matrix = (cv::Mat_<float>(1, 3) << -0.5, 0, 0.5);
				//sigma和ksize关系公式sigma = 0.3*((ksize-1)*0.5 - 1) + 0.8
				int ksize = 1;//核大小
				ksize = (lineparams.sigma - 0.35) / 0.15;
				if (ksize % 2 == 0)
				{
					ksize -= 1;
				}
				if (ksize <= 0)
				{
					ksize = 1;
				}
				//速度待优化
				for (int j = 0; j < rect2_num; ++j)
				{
					//映射后的矩形
					cv::Point2f transRoi[3];
					transRoi[0] = cv::Point2f(0, 0);
					transRoi[1] = cv::Point2f(rect2[j].width, 0);
					transRoi[2] = cv::Point2f(rect2[j].width, rect2[j].height);
					cv::Point2f Roi[3];
					Roi[0] = rect2[j].tl();
					Roi[1] = rect2[j].tl() + cv::Point(rect2[j].width, 0);
					Roi[2] = rect2[j].br();
					cv::Mat invMatrix = cv::getAffineTransform(transRoi, Roi);
					std::vector<float>values_p;//符合条件的正极性幅度值
					std::vector<double>indexs_p;//符合条件的正极性索引
					std::vector<float>values_n;//符合条件的负极性幅度值
					std::vector<double>indexs_n;//符合条件的负极性索引
					cv::Mat temimg;
					if (0 <= rect2[j].x && 0 <= rect2[j].width && rect2[j].x + rect2[j].width <= gray.cols && 0 <= rect2[j].y && 0 <= rect2[j].height && rect2[j].y + rect2[j].height <= gray.rows)
					{
						temimg = gray(rect2[j]);
					}
					else
					{
						continue;
					}
					//垂直投影
					int rows = temimg.rows;
					int cols = temimg.cols;
					cv::Mat img;
					cv::reduce(temimg, img, 0, cv::REDUCE_AVG);

					//对垂直投影图执行平滑操作
					cv::Mat gaussdst;
					cv::GaussianBlur(img, gaussdst, cv::Size(ksize, 1), lineparams.sigma);
					//对x方向求一阶导
					cv::Mat sobelimg1, sobelimg2;

					cv::filter2D(gaussdst, sobelimg1, CV_32F, matrix);
					sobelimg1.setTo(0, cv::abs(sobelimg1) < lineparams.edgethreshold);

					cv::filter2D(sobelimg1, sobelimg2, CV_32F, matrix);
#pragma region 每个卡尺判断正负极性和最强边缘
					int cols1 = sobelimg2.cols;
					std::vector<float>temvalues_p;//符合条件的正极性幅度值
					std::vector<double>temindexs_p;//符合条件的正极性索引
					std::vector<float>temvalues_n;//符合条件的负极性幅度值
					std::vector<double>temindexs_n;//符合条件的负极性索引
					int maxindexs[2];
					int minindexs[2];
					cv::minMaxIdx(sobelimg1, 0, 0, minindexs, maxindexs);
					double maxindex = maxindexs[1]; //最大幅度值索引
					double minindex = minindexs[1]; //最小幅度值索引
					float maxvalue = sobelimg1.at<float>(0, maxindex);//最大幅度值
					float minvalue = sobelimg1.at<float>(0, minindex);//最小幅度值

					for (int i = 0; i < cols1 - 1; i++)
					{
						float x1 = sobelimg2.at<float>(0, i);
						float x2 = sobelimg2.at<float>(0, i + 1);
						float x13 = x1 * x2;
						float tem = sobelimg1.at<float>(0, i);
						float tem2 = sobelimg1.at<float>(0, i + 1);
						double index = i;
						if (x13 < 0)
						{
							if (abs(tem) < abs(tem2))
							{
								tem = tem2;
								index = i + 1;
							}
							if (x1 < 0)
							{
								//负极性边缘
								temvalues_n.emplace_back(tem);
								temindexs_n.emplace_back(index);
							}
							else
							{
								temvalues_p.emplace_back(tem);
								temindexs_p.emplace_back(index);
							}
						}
						else if (x1 == 0 && tem != 0 && i > 0 && i < cols1)
						{
							//拐点
							float x4 = sobelimg2.at<float>(0, i - 1);
							if (x4 * x2 < 0)
							{
								if (x4 < 0)
								{
									//负极性边缘
									temvalues_n.emplace_back(tem);
									temindexs_n.emplace_back(index);
								}
								else
								{
									temvalues_p.emplace_back(tem);
									temindexs_p.emplace_back(index);
								}
							}
						}
					}
					if (lineparams.edge_polarity == 0 || lineparams.edge_polarity == 1)
					{
						calsubpix(sobelimg1, temindexs_p, temvalues_p);
						if (temindexs_p.size() > 0)
						{
							maxindex = *std::max_element(temindexs_p.begin(), temindexs_p.end());
							maxvalue = *std::max_element(temvalues_p.begin(), temvalues_p.end());
						}
					}
					if (lineparams.edge_polarity == 0 || lineparams.edge_polarity == 2)
					{
						calsubpix(sobelimg1, temindexs_n, temvalues_n);
						if (temindexs_n.size() > 0)
						{
							minindex = *std::min_element(temindexs_n.begin(), temindexs_n.end());
							minvalue = *std::min_element(temvalues_n.begin(), temvalues_n.end());
						}
					}
#pragma endregion
#pragma region 极性跟边缘位置选择
					int num_p = temvalues_p.size();
					int num_n = temvalues_n.size();
					auto fucall = [&] {
						if (num_p > 0)
						{
							values_p.insert(values_p.end(), temvalues_p.begin(), temvalues_p.end());
							indexs_p.insert(indexs_p.end(), temindexs_p.begin(), temindexs_p.end());
						}
						if (num_n > 0)
						{
							values_n.insert(values_n.end(), temvalues_n.begin(), temvalues_n.end());
							indexs_n.insert(indexs_n.end(), temindexs_n.begin(), temindexs_n.end());
						}
					};

					auto fucfirst = [&] {
						if (num_p > 0)
						{
							values_p.emplace_back(temvalues_p[0]);
							indexs_p.emplace_back(temindexs_p[0]);
						}
						if (num_n > 0)
						{
							values_n.emplace_back(temvalues_n[0]);
							indexs_n.emplace_back(temindexs_n[0]);
						}
					};
					int p_size = temindexs_p.size();
					int n_size = temindexs_n.size();
					auto fuclast = [&] {
						if (num_p > 0)
						{
							values_p.emplace_back(temvalues_p[num_p - 1]);
							indexs_p.emplace_back(temindexs_p[p_size - 1]);
						}
						if (num_n > 0)
						{
							values_n.emplace_back(temvalues_n[num_n - 1]);
							indexs_n.emplace_back(temindexs_n[n_size - 1]);
						}
					};
					auto fucbest = [&] {
						if (maxvalue > 0)
						{
							values_p.emplace_back(maxvalue);
							indexs_p.emplace_back(maxindex);
						}
						if (minvalue < 0)
						{
							values_n.emplace_back(minvalue);
							indexs_n.emplace_back(minindex);
						}
					};
					switch (select)
					{
					case measure_select::all:
						fucall();
						break;
					case measure_select::first:
						fucfirst();
						break;
					case measure_select::last:
						fuclast();
						break;
					case measure_select::best:
						fucbest();
						break;
					default:
						break;
					}
					double y = std::floor(lineparams.halfwidth);//y坐标
					if (indexs_p.size() > 0)
					{
						std::vector<cv::Point2d>points_p;
						for (int i = 0; i < indexs_p.size(); i++)
						{
							cv::Point2d temP(indexs_p[i], y);
							points_p.emplace_back(temP);
						}
						std::vector<cv::Point2d>tempoints_p;
						cv::transform(points_p, tempoints_p, invMatrix);
						edges.edgePoints_p.insert(edges.edgePoints_p.end(), tempoints_p.begin(), tempoints_p.end());
					}
					if (indexs_n.size() > 0)
					{
						std::vector<cv::Point2d>points_n;
						for (int i = 0; i < indexs_n.size(); i++)
						{
							cv::Point2d temN(indexs_n[i], y);
							points_n.emplace_back(temN);
						}
						std::vector<cv::Point2d>tempoints_n;
						cv::transform(points_n, tempoints_n, invMatrix);
						edges.edgePoints_n.insert(edges.edgePoints_n.end(), tempoints_n.begin(), tempoints_n.end());
					}
					edges.amplitude_p.insert(edges.amplitude_p.end(), values_p.begin(), values_p.end());
					edges.amplitude_n.insert(edges.amplitude_n.end(), values_n.begin(), values_n.end());
#pragma endregion
				}
			}
			void createRect2(fitLineParams& lineparams, std::vector<cv::Rect>& rect2)
			{
				cv::Point2d p1 = lineparams.p1;
				cv::Point2d p2 = lineparams.p2;
				int rectnum = lineparams.segNum;
				int rectw = std::floor(lineparams.halfwidth * 2) - 1;
				int recth = std::floor(lineparams.halfheight * 2) - 1;
				double deltx = p1.x - p2.x;
				double delty = p1.y - p2.y;
				double avgx = (1. * deltx / rectnum);
				double avgy = (1. * delty / rectnum);
				//double linangle = angle_lx(p1, p2);
				for (int i = 1; i < rectnum + 1; i++)
				{
					cv::Point center(p1.x - avgx * (i - 0.5), p1.y - avgy * (i - 0.5));
					//旋转矩形0°为Y正方向（垂直向上）
					int x = std::max(center.x - recth / 2, 0);
					int y = std::max(center.y - rectw / 2, 0);
					cv::Rect rect(x, y, recth, rectw);
					rect2.emplace_back(rect);
				}
			}
			void RansacLineFiler(const std::vector<cv::Point2d>& points, std::vector<cv::Point2d>& outPoints, std::vector<cv::Point2d>& invalidedgePoints, int segnum, double segscore, int max_num_iterations, double distance_threshold)
			{
				int n = points.size();
				if (n < 2)
				{
					return;
				}
				cv::RNG random;
				double bestScore = -1.;
				std::vector<cv::Point2d>vpdTemp;
				std::vector<cv::Point2d>vpdTempInvalid;//无效点
				int iterations;//迭代次数
				if (max_num_iterations != -1)
				{
					iterations = max_num_iterations;
				}
				else
				{
					iterations = log(1 - 0.9) / (log(1 - (1.00 / n))) * 10;
				}
				for (int k = 0; k < iterations; k++)
				{
					int i1 = 0, i2 = 0;
					while (i1 == i2)
					{
						i1 = random(n);
						i2 = random(n);
					}
					const cv::Point2d& p1 = points[i1];
					const cv::Point2d& p2 = points[i2];
					double score = 0;
					vpdTemp.clear();
					vpdTempInvalid.clear();

					double a = p2.y - p1.y;
					double b = p1.x - p2.x;
					double c = p2.x * p1.y - p1.x * p2.y;
					for (int i = 0; i < n; i += 2)
					{
						double distance = fabs(a * points[i].x + b * points[i].y + c) / sqrt(pow(a, 2) + pow(b, 2));//点到直线的距离
						if (distance < distance_threshold)
						{
							vpdTemp.push_back(points[i]);
							score += 1;
						}
						else
						{
							vpdTempInvalid.push_back(points[i]);
						}
						if (i == n - 1)
						{
							continue;
						}
						double distance1 = fabs(a * points[(size_t)(i + 1)].x + b * points[(size_t)(i + 1)].y + c) / sqrt(pow(a, 2) + pow(b, 2));//点到直线的距离
						if (distance1 < distance_threshold)
						{
							vpdTemp.push_back(points[(size_t)(i + 1)]);
							score += 1;
						}
						else
						{
							vpdTempInvalid.push_back(points[(size_t)(i + 1)]);
						}
					}
					if (score > bestScore)
					{
						bestScore = score;
						double scoreTemp = 1. * vpdTemp.size() / segnum;
						if (scoreTemp > segscore)
						{
							outPoints = vpdTemp;
							invalidedgePoints = vpdTempInvalid;
						}
						if (k >= iterations)
						{
							break;
						}
						if (max_num_iterations == -1)
						{
							//自适应迭代次数
							iterations = log(1 - 0.99) / (log(1 - (pow(scoreTemp, 2))));
						}
					}
				}
			}
			double angle_lx(cv::Point2d p1, cv::Point2d p2)
			{
				if (p1 == p2)
				{
					//同一个点
					return 0;
				}
				cv::Point2d vector = p2 - p1;
				if (vector.x == 0)
				{
					if (vector.y > 0)
					{
						return 90;
					}
					else
					{
						return -90;
					}
				}
				double angle = (acos(pow(vector.x, 2) / (vector.x * sqrt(pow(vector.x, 2) + pow(vector.y, 2))))) * (180 / CV_PI);
				if (p1.y > p2.y)
				{
					angle = -angle;
				}
				return  angle;
			}
		};
	}
}

namespace Fvision {
	namespace cvfunc
	{
		fitline::fitline() :impl_{ std::make_unique<impl>() }
		{
		}
		fitline::~fitline() = default;

		void fitline::findLine(cv::Mat& src, fitLineParams& lineparams, std::vector<lineResult>& outline, edgePointsRes& edges)
		{
			impl_->findLine(src, lineparams, outline, edges);
		}
		void fitline::drawLineCalipers(cv::Mat& src, fitLineParams& lineparams)
		{
			impl_->drawLineCalipers(src, lineparams);
		}
	}
}