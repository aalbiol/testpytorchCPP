

#include <torch/script.h>

#include <opencv2/opencv.hpp>
#define cimg_plugin1 "cvMat.h"
#define cimg_use_opencv

#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "CImg.h"

using namespace cimg_library;

// float* recortePlanos2blob(const std::vector<short*>& planos, int plane_width, int plane_height,
//                           int top, int bottom, int left, int right, int blob_width, int blob_height) {
//    int nplanos = planos.size();
//    float* newblob = new float[nplanos * blob_width * blob_height];
//    int crop_width = right - left + 1;
//    int crop_height = bottom - top + 1;
//    cv::Size crop_size;
//    crop_size.width = crop_width;
//    crop_size.height = crop_height;
//    std::cout << "crop_size=" << crop_size << "\n";
//    cv::Mat cropped_plane(crop_size, CV_32FC1);
//    cv::Mat resized_plane;

//    float* pcropped_plane = (float*)cropped_plane.data;
//    std::cerr << "Convirtiendo a blob\n";
//    for (int c = 0; c < nplanos; c++) {
//       short* plano = planos[c];
//       cv::Mat cvplano = cv::Mat(plane_height, plane_width, CV_16UC1, const_cast<short*>(plano));  // Sin copia
//       cv::Rect roi(left, top, crop_width, crop_height);
//       cv::Mat cropped_plane = cvplano(roi);
//       cv::Mat recorte_float;
//       cropped_plane.convertTo(recorte_float, CV_32FC1);
//       cv::resize(recorte_float, resized_plane, cv::Size(blob_width, blob_height));

//       float* pdest = newblob + c * blob_width * blob_height;
//       memcpy((float*)pdest, (float*)resized_plane.data, blob_width * blob_height * sizeof(float));
//    }
//    return newblob;
// }




/* ***********************************************************************************************************
void recortePlanos2tensor(const std::vector<short*>& planos, int plane_width, int plane_height,
                          int top, int bottom, int left, int right, int blob_width, int blob_height,
                          torch::Tensor& tensor)

Si el recorte es del tamaño final deseado NO se llama a resize y el resultado de convertir a float se hace directamente sobre la memoria
del tensor de salida para evitar copias 
****************************************************************************************************** */
void recortePlanos2tensor(const std::vector<short*>& planos, int plane_width, int plane_height,
                          int top, int bottom, int left, int right, int blob_width, int blob_height,
                          torch::Tensor& tensor) {
   int nplanos = planos.size();
   tensor = torch::empty({1, nplanos, blob_width, blob_height});
   float* newblob = tensor.data_ptr<float>();

   int crop_width = right - left + 1;
   int crop_height = bottom - top + 1;

   cv::Size crop_size;
   crop_size.width = crop_width;
   crop_size.height = crop_height;
   
   if (blob_width != crop_width || blob_height != crop_height) {
      std::cout << "Sin resize\n";
      
      for (int c = 0; c < nplanos; c++) {
         short* plano = planos[c];
         cv::Mat cvplano = cv::Mat(plane_height, plane_width, CV_16UC1, const_cast<short*>(plano));  // Sin copia
         cv::Rect roi(left, top, crop_width, crop_height);
         cv::Mat cropped_plane = cvplano(roi);
         cv::Mat recorte_float;
         cropped_plane.convertTo(recorte_float, CV_32FC1);
         float* pdest = newblob + c * blob_width * blob_height;
         cv::Mat resized_plane = cv::Mat(blob_height, blob_width, CV_32F, const_cast<float*>(pdest));
         cv::resize(recorte_float, resized_plane, cv::Size(blob_width, blob_height));
      }
   } else {

std::cout << "Sin resize\n";
      
      for (int c = 0; c < nplanos; c++) {
         short* plano = planos[c];
         cv::Mat cvplano = cv::Mat(plane_height, plane_width, CV_16UC1, const_cast<short*>(plano));  // Sin copia
         cv::Rect roi(left, top, crop_width, crop_height);
         cv::Mat cropped_plane = cvplano(roi); //short
   
         float* pdest = newblob + c * blob_width * blob_height;
         cv::Mat recorte_float = cv::Mat(blob_height, blob_width, CV_32F, const_cast<float*>(pdest));
         cropped_plane.convertTo(recorte_float, CV_32FC1);         
      }
   }
}

//=================================================================================================
int main(int argc, const char* argv[]) {
   std::string cimgfilename = argv[1];

   cimg_library::CImgList<short> listimg(cimgfilename.c_str());

   // Voy a hacer un recorte de cada vista y lo voy a reescalar a 70x70

   int top = 11;
   int bottom = 50;
   int left = 10;
   int right = 55;

   int blob_width = 46;
   int blob_height = 41;
   // forward clist
   CImgList<float> listblob(listimg.size());
   for (int v = 0; v < listimg.size(); v++) {
      cimg_library::CImg<short> cimg = listimg[v];

      std::vector<short*> planos(4);
      int ncanales = planos.size();
      for (int p = 0; p < 4; p++)
         planos[p] = cimg.get_shared_channel(p).data();
      int plane_width = cimg.width();
      int plane_height = cimg.height();


      torch::Tensor tensor;
      int64 t0 = cv::getTickCount();
      recortePlanos2tensor(planos, plane_width, plane_height,
                           top, bottom, left, right,
                           blob_width, blob_height, tensor);
      int64 t1 = cv::getTickCount();

      int64 usecs = 1e6 * (t1 - t0) / cv::getTickFrequency();
       std::cout << "Tiempo  Recorte plano " << usecs << " usecs " << " \n";

      CImg<float> cimgblob(tensor.data_ptr<float>(), blob_width, blob_height, 1, ncanales, true);  // No hace copia de los datos
      listblob[v] = cimgblob;
   }
   listblob.display("Recortadas");
}
