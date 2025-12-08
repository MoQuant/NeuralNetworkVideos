#include <iostream>
#include <torch/torch.h>
#include <vector>

struct NeuralNetwork : torch::nn::Module {
    torch::nn::Linear layer1{nullptr};
    torch::nn::Linear layer2{nullptr};
    torch::nn::Linear layer3{nullptr};

    NeuralNetwork(int64_t input_size, int64_t output_size){
        layer1 = register_module("layer1", torch::nn::Linear(input_size, 8));
        layer2 = register_module("layer2", torch::nn::Linear(8, 6));
        layer3 = register_module("layer3", torch::nn::Linear(6, 4));
    }

    torch::Tensor forward(torch::Tensor x){
        x = torch::relu(layer1->forward(x));
        x = torch::relu(layer2->forward(x));
        x = layer3->forward(x);
        return x;
    }


};

int main()
{
    std::vector<double> inputs = {0.33, 0.26, 0.91, 0.84, 0.33, 0.19, 0.47, 0.74, 0.12, 0.99};
    std::vector<double> outputs = {0.22, 0.43, 0.11, 0.83};

    torch::Tensor X = torch::tensor(inputs);
    torch::Tensor y = torch::tensor(outputs);

    NeuralNetwork nnet((int64_t) inputs.size(), (int64_t) outputs.size());
    
    int epochs = 100;
    torch::optim::Adam optimizer(nnet.parameters(), torch::optim::AdamOptions(1e-3));
    torch::nn::MSELoss criterion;

    for(int epoch = 0; epoch < epochs; ++epoch){
        torch::Tensor nnet_output = nnet.forward(X);
        torch::Tensor loss = criterion(nnet_output, y);
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
        std::cout << "Epochs Left: " << epochs - epoch << " | Error: " << loss.item() << std::endl;
        
    }

    return 0;
}