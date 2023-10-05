#include <torch/script.h>

#include <chrono>
#include <opencv2/opencv.hpp>
#include <thread>
#define cimg_plugin1 "cvMat.h"
#define cimg_use_opencv
// #define cimg_display 0
#include <cmath>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "CImg.h"

using namespace cimg_library;

std::vector<std::string> labels = {"Blandos", "Cicatrices", "defectos_severos", "Deformes", "Deshidratado", "Flor_seca", "Lagrima", "Rabos", "Rojos", "Verdes"};

void usage(std::string& modelpath, std::string& cimgfilename) {
   std::cerr << "Usage: prediction " << std::endl;
   std::cout << "Debe existir " << modelpath << " y " << cimgfilename << " en la carpeta actual\n";
}

float sigmoid(float x) { return 1.0 / (1.0 + exp(-x)); }

void printMatsDimensions(const std::vector<cv::Mat>& mats, const std::string& name) {
   for (int i = 0; i < mats.size(); i++) {
      std::cout << name << "[" << i << "]:\n";
      std::cout << "rows=" << mats[i].rows << " cols=" << mats[i].cols << " channels=" << mats[i].channels() << "\n";
      std::cout << mats[i].type() << "\n";
      std::cout << "Dims:";
      for (int k = 0; k < mats[i].dims; k++) {
         std::cout << mats[i].size[k] << " ";
      }
      std::cout << "\n";
   }
}

void printMatDimensions(const cv::Mat& mat, const std::string& name) {
   std::cout << name << ":\n";
   std::cout << "rows=" << mat.rows << " cols=" << mat.cols << " channels=" << mat.channels() << "\n";
   std::cout << mat.type() << "\n";
   std::cout << "Dims:";
   for (int k = 0; k < mat.dims; k++) {
      std::cout << mat.size[k] << " ";
   }
   std::cout << "\n";
}

void printTensorDims(const torch::Tensor& tensor, const std::string& name) {
   std::cout << name << ":\n";
   std::cout << "dims=" << tensor.dim() << "\n";
   std::cout << "sizes=";
   for (int i = 0; i < tensor.dim(); i++) {
      std::cout << tensor.size(i) << " ";
   }
   std::cout << "\n";
}

void cimgList_2_MatArray(const cimg_library::CImgList<float>& clist, std::vector<cv::Mat>& mats, const cv::Size& im_sz, float scale_factor = 1.0) {
   /*
    Returns an array of RGB Mats of size im_sz from a CImgList of size clist.
   */
   int nvistas = clist.size();
   for (int mm = 0; mm < nvistas; mm++) {
      const cimg_library::CImg<float>& vista = clist[mm];
      cimg_library::CImg<float> v = vista.get_shared_channels(0, 3) * scale_factor;
      cv::Mat vistamat = v.get_MAT2();  // Returns RGB MAt
      cv::resize(vistamat, vistamat, im_sz);
      mats.push_back(vistamat);
   }
}
//------------------

void cimgList_2_Tensor(const cimg_library::CImgList<float>& clist, const cv::Size& im_sz, float scale_factor, torch::Tensor& batch_tensor) {
   int nvistas = clist.size();
   batch_tensor = torch::zeros({nvistas, 4, im_sz.height, im_sz.width});
   // std::vector< torch::Tensor> tensores (nvistas);

   for (int mm = 0; mm < nvistas; mm++) {
      const cimg_library::CImg<float>& vista = clist[mm];
      cimg_library::CImg<float> v = vista.get_shared_channels(0, 3) * scale_factor;
      cv::Mat vistamat = v.get_MAT2();  // Returns RGB MAt
      cv::resize(vistamat, vistamat, im_sz);

      torch::Tensor input_tensor = torch::from_blob(vistamat.data, {im_sz.height, im_sz.width, vistamat.channels()}, c10::kFloat);
      batch_tensor[mm] = input_tensor.permute({2, 0, 1});
   }
}

void prepara_cimglist(cimg_library::CImgList<float>& clist, const cv::Size& im_sz) {
   // Usando OpenCV escala todas las vistas al mismo tamaño y se queda con los 4 primeros canales
   // Reemplaza a la lista original

   int nvistas = clist.size();
   for (int mm = 0; mm < nvistas; mm++) {
      cimg_library::CImg<float>& vista = clist[mm];
      cimg_library::CImg<float> v = vista.get_shared_channels(0, 3);
      cv::Mat vistamat = v.get_MAT2();  // Returns RGB MAt
      cv::resize(vistamat, vistamat, im_sz);
      vista.assign(vistamat);
   }
}

void fast_cimgList_2_Tensor(const cimg_library::CImgList<float>& clist, torch::Tensor& batch_tensor) {
   // TODAS las vistas mismo tamaño
   int nvistas = clist.size();
   int width = clist[0].width();
   int height = clist[0].height();
   int canales = clist[0].spectrum();
   batch_tensor = torch::zeros({nvistas, canales, height, width});

   for (int mm = 0; mm < nvistas; mm++) {
      const cimg_library::CImg<float>& vista = clist[mm];
      batch_tensor[mm] = torch::from_blob((void*)vista.data(), {canales, height, width}, c10::kFloat);
   }
}

//------------------
void normalizar(torch::Tensor& tensor, float scale_factor, const std::vector<float>& mean, const std::vector<float>& std) {
   int ncanales = mean.size();
   for (int v = 0; v < tensor.size(0); v++) {
      for (int i = 0; i < ncanales; i++) {
         tensor[v][i] -= mean[i] / scale_factor;
         tensor[v][i] *= (scale_factor / std[i]);
      }
   }
}

//------------------
void forwardCList(const cimg_library::CImgList<float>& clist, const std::vector<float>& mean, const std::vector<float>& std, torch::jit::script::Module& model, torch::Device& device, float scale_factor, cv::Size im_sz, torch::Tensor& out_tensor) {
   torch::Tensor batch_tensor;
   // cimgList_2_Tensor(clist, im_sz, scale_factor,batch_tensor);
   fast_cimgList_2_Tensor(clist, batch_tensor);
   // Move to GPU
   batch_tensor = batch_tensor.to(device);

   // Normalize
   normalizar(batch_tensor, scale_factor, mean, std);

   torch::Tensor out_tensor_gpu = model.forward({batch_tensor}).toTensor();
   out_tensor = out_tensor_gpu.to(torch::kCPU);
}

//------------------
void forwardMat(const cv::Mat& mat, const std::vector<float>& mean, const std::vector<float>& std, torch::jit::script::Module& model, torch::Device& device, torch::Tensor& out_tensor) {
   torch::Tensor input_tensor = torch::from_blob(mat.data, {mat.rows, mat.cols, mat.channels()}, c10::kFloat);
   input_tensor = input_tensor.permute({2, 0, 1});
   for (int c = 0; c < mean.size(); c++) input_tensor[c] = input_tensor[c].sub_(mean[c]).div_(std[c]);

   input_tensor.unsqueeze_(0);
   torch::Tensor gpuTensor = input_tensor.to(device);
   out_tensor = model.forward({gpuTensor}).toTensor();
   out_tensor = out_tensor.to(torch::kCPU);

   return;
}

cimg_library::CImg<float> cimgMaxY(const cimg_library::CImg<float>& cimg) {
   cimg_library::CImg<float> maxy(cimg.width(), 1, 1, 1);
   cimg_forX(cimg, x) { maxy(x) = cimg.get_column(x).max(); }
   return maxy;
}

//------------------
void pathchLogitsTensor_2_DefectProbs(const torch::Tensor& out_tensor, cimg_library::CImg<float>& probs) {
   int ndims = out_tensor.dim();

   cimg_library::CImg<float> cblobresult;

   if (ndims == 4) {
      // Calcula el máximo espacial. Las dimensiones espaciales son las últimas dos.
      // La penúltima  dimensión es el tipo de defecto
      // La ultima dimensión es el número de vista
      cblobresult.assign(out_tensor.data_ptr<float>(), out_tensor.size(3), out_tensor.size(2), out_tensor.size(1), out_tensor.size(0));
      probs.assign(cblobresult.depth());
      cimg_forX(probs, z) {
         float maximo = -1e20;
         cimg_forXYC(cblobresult, x, y, c) {
            float val = cblobresult(x, y, z, c);
            if (val > maximo) maximo = val;
         }
      }
   } else if (ndims == 2) {
      // Calcula el maximo de las vistas
      //  La ultima dimensión es el número de vista
      cblobresult.assign(out_tensor.data_ptr<float>(), out_tensor.size(1), out_tensor.size(0));
      probs = cimgMaxY(cblobresult);
   } else {
      std::cerr << "Error: ndims=" << ndims << " no soportado\n";
   }

   cimg_foroff(probs, i) { probs[i] = sigmoid(probs[i]); }
}

void pringCImg2D(const cimg_library::CImg<float>& cimg, const std::string& name, int precision = 3) {
   std::cout << name << ":\n";
   std::cout << std::fixed;
   std::cout << std::setprecision(precision);
   cimg_forY(cimg, y) {
      cimg_forX(cimg, x) { std::cout << cimg(x, y) << " "; }
      std::cout << "\n";
   }
}

//=================================================================================================
int main(int argc, const char* argv[]) {
   std::string model_path = "arandanos.ts";
   std::string cimgfilename = "arandano.cimg";
   usage(model_path, cimgfilename);

   torch::Device device = torch::kCUDA;
   std::string device_name = "GPU";

   torch::jit::script::Module model;
   try {
      model = torch::jit::load(model_path, device);
   } catch (const c10::Error& e) {
      std::cerr << "error loading the model " << model_path << "\n";
      return -1;
   }
   std::cerr << "Modelo " << model_path << " cargado\n";
   model.eval();
   float max_value = 255.0;
   float scale_factor = 1.0 / max_value;

   std::vector<float> mean = {0.22398601472377777, 0.21038898825645447, 0.2316783219575882, 0.44684037566185};
   std::vector<float> std{0.1344703882932663, 0.12302909046411514, 0.13383440673351288, 0.25189703702926636};

   cv::Size im_sz(110, 110);

   std::cout << cimgfilename << "\n";
   cimg_library::CImgList<float> listimg(cimgfilename.c_str());

   std::cerr << "Imagen " << cimgfilename << " Cargada\n";

   CImg<float> probs;

   int64 t0 = cv::getTickCount();
   prepara_cimglist(listimg, im_sz);
   int64 t1 = cv::getTickCount();
   int64 usecs = 1e6 * (t1 - t0) / cv::getTickFrequency();
   std::cout << "Tiempo  preparar batch OpenCV " << usecs << " usecs "
             << " Batch size:" << listimg.size() << " \n";
   // forward clist

   std::vector<int64_t> delays;
   delays.push_back(0);
   delays.push_back(10);
   delays.push_back(20);
   delays.push_back(30);
   delays.push_back(30);
   delays.push_back(40);
   delays.push_back(50);
   delays.push_back(60);
   delays.push_back(70);
   delays.push_back(80);
   delays.push_back(90);
   delays.push_back(100);
   delays.push_back(120);
   delays.push_back(130);
   delays.push_back(140);
   delays.push_back(150);

   std::vector <int> tiempos(delays.size());

   for (int n = 0; n < delays.size(); n++) {
      int64_t delay = delays[n];
      for (int r = 0; r<10; r++) {
         torch::Tensor out_tensor;
          
         std::this_thread::sleep_for(std::chrono::milliseconds(delay));
         int64 t0 = cv::getTickCount();
         // torch::Tensor entrada = cimgList_2_Tensor(listimg, im_sz, scale_factor );
         forwardCList(listimg, mean, std, model, device, scale_factor, im_sz, out_tensor);
         int64 t1 = cv::getTickCount();

         int64 usecs = 1e6 * (t1 - t0) / cv::getTickFrequency();
         std::cout <<"r" << r << " Tiempo  " << device_name << " : " << usecs << " usecs "
                   << " Batch size:" << listimg.size() << "  Delay:" << delay << " \n";

         pathchLogitsTensor_2_DefectProbs(out_tensor, probs);
         pringCImg2D(probs, "probs");
         tiempos[n]=usecs;
      }
   }
   for (int n = 0; n < delays.size(); n++) {
      std::cout << "Delay:" << delays[n] << " ---> tiempo: " << tiempos[n] << "\n";
   }
}
