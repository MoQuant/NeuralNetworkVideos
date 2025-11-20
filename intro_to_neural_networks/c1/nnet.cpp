#include <iostream>
#include <string>
#include <math.h>
#include <time.h>
#include <vector>
#include <map>

void printer(std::vector<std::vector<double>> x){
    for(auto & i : x){
        for(auto & j : i){
            std::cout << j << "\t";
        }
        std::cout << std::endl;
    }
}

void dimensions(std::vector<std::vector<double>> x){
    std::cout << "Matrix Dimensions: (" << x.size() << "," << x[0].size() << ")" << std::endl;
}

std::vector<std::vector<double>> mmult(std::vector<std::vector<double>> x, std::vector<std::vector<double>> y){
    std::vector<std::vector<double>> z;
    std::vector<double> temp;
    double total = 0;

    for(int i = 0; i < x.size(); ++i){
        temp.clear();
        for(int j = 0; j < y[0].size(); ++j){
            total = 0;
            for(int k = 0; k < y.size(); ++k){
                total += x[i][k]*y[k][j];
            }
            temp.push_back(total);
        }
        z.push_back(temp);
    }

    return z;
}

std::vector<std::vector<double>> transpose(std::vector<std::vector<double>> x){
    std::vector<std::vector<double>> y;
    std::vector<double> temp;
    for(int j = 0; j < x[0].size(); ++j){
        temp.clear();
        for(int i = 0; i < x.size(); ++i){
            temp.push_back(x[i][j]);
        }
        y.push_back(temp);
    }
    return y;
}

std::vector<std::vector<double>> buildvector(int m, int n, bool randomx){
    srand(time(NULL));
    
    auto dWT = [](){
        int num = 100;
        double dw = (rand() % (2*num + 1));
        return fmin(1.0, dw / 100.0);
    };

    std::vector<std::vector<double>> z;
    std::vector<double> temp;

    for(int i = 0; i < m; ++i){
        temp.clear();
        for(int j = 0; j < n; ++j){
            if(randomx == true){
                temp.push_back(dWT());
            } else {
                temp.push_back(0.0);
            }
        }
        z.push_back(temp);
    }

    return z;
}

std::vector<std::vector<double>> Sigmoid(std::vector<std::vector<double>> x, bool dv){
    std::vector<std::vector<double>> y;
    for(int i = 0; i < x.size(); ++i){
        double fx = 1.0 / (1.0 + exp(-x[i][0]));
        if(dv == true){
            y.push_back({fx*(1-fx)});
        } else {
            y.push_back({fx});
        }
    }
    return y;
}

std::vector<std::vector<double>> calculate_error(std::vector<std::vector<double>> y, std::vector<std::vector<double>> yhat){
    std::vector<std::vector<double>> error;
    for(int i = 0; i < y.size(); ++i){
        error.push_back({pow(y[i][0] - yhat[i][0], 2)});
    }
    return error;
}

std::vector<std::vector<double>> update(std::vector<std::vector<double>> weights, std::vector<std::vector<double>> grad){
    for(int i = 0; i < weights.size(); ++i){
        for(int j = 0; j < weights[0].size(); ++j){
            weights[i][j] -= transpose(grad)[0][j];
        }
    }
    return weights;
}

std::vector<std::vector<double>> calculate_delta(std::vector<std::vector<double>> error, std::vector<std::vector<double>> Layer){
    std::vector<std::vector<double>> the_delta, sig = Sigmoid(Layer, true);
    for(int i = 0; i < error.size(); ++i){
        the_delta.push_back({2.0*error[i][0]*sig[i][0]});
    }
    return the_delta;
}

std::vector<std::vector<double>> calculate_delta2(std::vector<std::vector<double>> error, std::vector<std::vector<double>> Layer){
    std::vector<std::vector<double>> the_delta, sig = Sigmoid(Layer, true);
    for(int i = 0; i < error.size(); ++i){
        the_delta.push_back({error[i][0]*sig[i][0]});
    }
    return the_delta;
}

int main()
{
    std::vector<int> axis, raxis;
    std::vector<std::vector<double>> x, y, error, delta;
    std::map<int, std::vector<std::vector<double>>> Weights, Layers, SLayers, gradient;
    
    x = {{0.25},{0.33},{0.11},{0.77},{0.42},{0.86}};
    y = {{0.05},{0.11},{0.07}};

    int epochs = 200;

    int m = x.size();
    int n = y.size();

    for(int i = m; i > n; --i){
        axis.push_back(i);
    }

    raxis = axis;
    std::reverse(raxis.begin(), raxis.end());

    for(auto & t : axis){
        Weights[t] = buildvector(t, t-1, true);
        Layers[t] = buildvector(t-1, 1, false);
        SLayers[t] = buildvector(t-1, 1, false);
    }

    for(int epoch = 1; epoch <= epochs; ++epoch){
        // Forward Propigaton
        for(int i = 0; i < axis.size(); ++i){
            int index = axis[i];
            if(i == 0){
                Layers[index] = transpose(mmult(transpose(x), Weights[index]));
            } else {
                Layers[index] = transpose(mmult(transpose(SLayers[index+1]), Weights[index]));
            }
            SLayers[index] = Sigmoid(Layers[index], false);
        }

        // Backpropigation
        for(int i = 0; i < raxis.size(); ++i){
            int index = raxis[i];
            if(i == 0){
                error = calculate_error(y, SLayers[index]);
                printer(transpose(error));
                delta = calculate_delta(error, Layers[index]);
            } else {
                error = mmult(Weights[index-1], delta);
                delta = calculate_delta2(error, Layers[index]);
            }
            gradient[index] = delta;
        }

        // Update Weights
        for(auto & i : axis){
            Weights[i] = update(Weights[i], gradient[i]);
        }

    }

    return 0;
}

