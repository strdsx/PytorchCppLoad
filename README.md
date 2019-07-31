# Description

## Tutorials : Loading A Pytorch Model In C++ [참고 : Pytorch Tutorials](https://pytorch.org/tutorials/advanced/cpp_export.html)

1. toScriptModule.py
    - <code> example = torch.rand(1,3,32,32).cuda()</code>
        - 나의 모델 Input Shape에 맟춰 example tensor를 생성
    - <code>traced_script_module = torch.jit.trace(model, example)</code>
        - example tensor를 사용하여 Script Module로 변환
    - <code>traced_script_module.save('script_module.pt')</code>
        - Script Module 저장
    - 저장된 Script Module을 C++에서 사용
1. example-app.cpp
    - 사이트 참고하여 build
        - libtorch 다운로드 [download libtorch](https://pytorch.org/)
        - 다운로드 완료 후 [Pytorch Tutorials](https://pytorch.org/tutorials/advanced/cpp_export.html) 참고하여 build
    - 나의 Model Input Tensor 형태에 맞춰 Normalization, resize 진행 (Python Pytorch Input Tensor와 동일해야함)
    - ResNet C++ Load
        - Mat --> Tensor : <code>torch::from_blob(mat_image.data, {1, INPUTSIZE, INPUTSIZE, 3}, torch::ScalarType::Float).to(torch::kCUDA)</code>
        - <code>img_tensor.permute({ 0, 3, 1, 2 })</code>
        - (신기하게 mat -> tensor 변환과정에 바로 Input Shape에 맞춰 <code>{1, 3, INPUTSIZE, INPUTSIZE}</code>와 같이 사용하였는데, Input 형태가 달라져서 모델이 정상적으로 예측하지 않음
        그래서 위와 같이 permutation함. 이유는 모름...)
    - RetinaNet C++ Load 진행중
        - Input Image는 normalization - resize & scaling - padding --> Tensor로 변경
        - 현재 코드는 resize & scaling, padding --> Tensor 형태에서 Normalization 진행
        - Output : Tuple(Tensor, Tensor, Tensor) --> 코드 참고
        - **RetinaNet의 경우 최종 Input rows의 형태가 padding, scaling... 등올 통해 달라 (input shape이 고정적이지 않음). <code>torch.jit.trace()</code>를 Script modul를 생성할 때 고정된 example_tensor가 들어가게 되므로, 해당 example_tensor와 동일한 형태의 image width, height를 갖는 이미지만 detection 가능함**


## ResNetDLL_Export
1. DLL 크게 2가지 클래스로 구성됨
    - <code> void Classifier::init()</code> --> Script Module Initialization
    - <code> int Classifier::classify(cv::Mat img)</code> --> Mat image를 받아 예측값을 반환
1. Tutorial과 동일하게 진행...
1. ResNetDLL.dll, ResNetDLL.lib 생성
    - 저장 경로 : ResNetDLL/build/Release/..


## ResNetDLL_Import
1. 생성된 ResNet.dll, ResNet.lib를 bin, lib 폴더에 저장 후 Visual Studio 에서 환경 경로 설정
1. libtorch 환경도 **DLL Exports**와 동일하게 경로 설정
1. Build가 성공하면 "DLL_Test/build/Release/"에 필요한 dll 파일 이동
1. 실행
    - <code>cd DLL_Test/build/Release</code>
    - <code>DLL_Test.exe [MODEL_PATH] [IMAGE_PATH]