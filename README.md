# ELM-C++

**Extreme Learning Machine - Fast C++ Implementation**

A high-performance C++ library implementing Extreme Learning Machines (ELM) for classification and regression tasks, with command-line tools and MATLAB MEX interfaces.

## What are Extreme Learning Machines?

Extreme Learning Machines (ELM) are single-hidden-layer feedforward neural networks that offer extremely fast training compared to traditional neural networks. The key innovation is that input weights and hidden layer biases are randomly initialized and remain fixed, while only the output weights are computed analytically.

### Key Features of ELM

- **Ultra-fast training**: No iterative backpropagation needed
- **Analytical solution**: Output weights computed using Moore-Penrose pseudoinverse
- **Good generalization**: Often achieves comparable or better accuracy than SVMs
- **Minimal tuning**: Only two hyperparameters (hidden neurons and regularization)
- **Universal approximation**: Can approximate any continuous function

### How ELM Works

1. **Random initialization**: Input weights (W_in) and biases are randomly generated
2. **Hidden layer computation**: H = sigmoid(W_in * X + bias)
3. **Output weight computation**: W_out = (H^T * H + I/C)^(-1) * H^T * T
   - Where C is the regularization parameter
   - T is the target matrix (1-of-K encoding for classification)

This approach eliminates the need for gradient descent, making training orders of magnitude faster than traditional neural networks.

### Mathematical Foundation

The ELM training solves the following regularized least-squares problem:

```
minimize: ||HW_out - T||^2 + ||W_out||^2/C
```

This is solved analytically using Cholesky decomposition for numerical stability.

**Reference**: G.-B. Huang et al., "Extreme Learning Machine for Regression and Multiclass Classification," IEEE TSMC, vol. 42, no. 2, pp. 513-529, 2012.

## Features

✅ **C++ Library**: Header-only template implementation with Eigen3  
✅ **Command-line Tools**: `elm_train` and `elm_predict` for CSV data  
✅ **MATLAB MEX**: Interfaces for MATLAB users  
✅ **Classification & Regression**: Unified framework for both tasks  
✅ **Robust Implementation**: Input validation, error handling, dimension checking  
✅ **Flexible Label Handling**: Supports arbitrary label values (not just 1-based)  

## Dependencies

- **Eigen3** (version 3.3 or higher): Linear algebra library
- **CMake** (version 3.10 or higher): Build system
- **C++11** compatible compiler

## Installation

### Building from Source

```bash
# Clone the repository
git clone https://github.com/yourusername/ELM-C--.git
cd ELM-C--

# Create build directory
mkdir build && cd build

# Configure and build
cmake ..
cmake --build .

# Install (optional)
sudo cmake --install .
```

**Note for Windows users:** If Eigen3 is installed in a non-standard location, specify it during configuration:

```bash
cmake -DEIGEN3_INSTALL_DIR="C:/Program Files (x86)/Eigen3" ..
```

Or alternatively, set CMAKE_PREFIX_PATH:

```bash
cmake -DCMAKE_PREFIX_PATH="C:/Program Files (x86)/Eigen3" ..
```

The executables will be in `build/bin/`:
- `elm_train` - Training utility
- `elm_predict` - Prediction utility

### Building MEX Files (Optional)

If you want to build MATLAB MEX files:

```bash
cmake -DBUILD_MEX=ON ..
cmake --build .
```

## Usage

### Command-Line Tools

#### Training a Model

```bash
./elm_train -i train_data.csv -o model.elm -n 100 -c 1.0
```

**Arguments:**
- `-i <file>`: Input CSV training file (required)
- `-o <file>`: Output model file (default: model.elm)
- `-n <int>`: Number of hidden neurons (default: features/2)
- `-c <double>`: Regularization parameter C (default: 1.0)
- `-l <int>`: Label column index (default: -1, last column)
- `--no-header`: Skip header row

**CSV Format for Training:**
```csv
feature1,feature2,feature3,label
1.2,3.4,5.6,1
2.3,4.5,6.7,2
3.4,5.6,7.8,1
```

#### Making Predictions

```bash
./elm_predict -i test_data.csv -m model.elm -o predictions.csv
```

**Arguments:**
- `-i <file>`: Input CSV test file (required)
- `-m <file>`: Input model file (required)
- `-o <file>`: Output predictions file (default: predictions.csv)
- `-l <int>`: Label column to skip: -1=last column, >=0=specific column, -2=none (default: -2)
- `--no-header`: Skip header row
- `--scores`: Output all class scores (not just predicted label)

**CSV Format for Testing:**
```csv
feature1,feature2,feature3
1.2,3.4,5.6
2.3,4.5,6.7
```

**Note:** If your test file includes labels (e.g., same format as training data), use `-l -1` to skip the last column:
```bash
./elm_predict -i train_data.csv -m model.elm -o predictions.csv -l -1
```

### Using as a C++ Library

Include the header in your project:

```cpp
#include "elml.hpp"
#include <Eigen/Core>

using namespace Eigen;

int main() {
    // Prepare your data
    double X[] = {...};  // Features (dims x nsmp)
    double Y[] = {...};  // Labels (nsmp)
    
    int dims = 4;        // Number of features
    int nsmp = 100;      // Number of samples
    int nhn = 50;        // Number of hidden neurons
    double C = 1.0;      // Regularization parameter
    
    // Train model
    MatrixXd inW, bias, outW;
    int result = elmTrain(X, dims, nsmp, Y, nhn, C, inW, bias, outW);
    
    if (result == 0) {
        // Predict on new data
        double X_test[] = {...};
        int nsmp_test = 20;
        MatrixXd scores;
        
        elmPredict(X_test, dims, nsmp_test, scores, inW, bias, outW);
        
        // scores contains prediction scores (nclasses x nsmp_test)
    }
    
    return 0;
}
```

### MATLAB Interface

#### Training

```matlab
% Load data
load('iris.mat');

% Train ELM model
% X: features (dims x samples)
% Y: labels (samples x 1)
% nhn: number of hidden neurons
% C: regularization parameter
[inW, bias, outW] = mexElmTrain(X, Y, 50, 1.0);
```

#### Prediction

```matlab
% Predict on test data
scores = mexElmPredict(inW, bias, outW, X_test);

% Get predicted labels
[~, predicted_labels] = max(scores, [], 1);
```

## Hyperparameter Tuning

### Number of Hidden Neurons (nhn)

- **Too few**: Underfitting, poor performance
- **Too many**: Overfitting, slower computation
- **Rule of thumb**: Start with `nhn = features / 2` to `features * 2`
- **Recommendation**: Use cross-validation to find optimal value

### Regularization Parameter (C)

- **Small C**: More regularization (simpler model)
- **Large C**: Less regularization (more complex model)
- **Range**: Typically 10^-6 to 10^6
- **Default**: 1.0 is a good starting point
- **Recommendation**: Grid search over [0.001, 0.01, 0.1, 1, 10, 100, 1000]

## Performance Tips

1. **Normalize your data**: ELM benefits from feature scaling (mean=0, std=1)
2. **Start with default parameters**: Often work well
3. **Increase hidden neurons**: If underfitting
4. **Increase regularization**: If overfitting
5. **Try multiple random seeds**: ELM uses random initialization

## Examples

See the `mat/` directory for MATLAB example datasets:
- `iris.mat` - Classic Iris flower classification
- `breast.mat` - Breast cancer classification
- `data.mat` - Generic example dataset

## Project Structure

```
ELM-C--/
├── elml.hpp              # Main ELM library (header-only)
├── train.cpp             # Training command-line utility
├── predict.cpp           # Prediction command-line utility
├── mexElmTrain.cpp       # MATLAB MEX training interface
├── mexElmPredict.cpp     # MATLAB MEX prediction interface
├── CMakeLists.txt        # CMake build configuration
├── README.md             # This file
├── mat/                  # MATLAB example datasets
└── attic/                # Old/experimental code

```

## Known Issues & Limitations

- **Binary model format**: Not portable across different architectures
- **Memory**: Large datasets may require significant RAM
- **Activation function**: Currently only sigmoid; could add ReLU, tanh, etc.
- **Regression**: Supported but primarily tested for classification

## Contributing

Contributions are welcome! Please feel free to submit pull requests or open issues for bugs and feature requests.

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

You are free to use, modify, and distribute this software in accordance with the Apache 2.0 license terms.

## Authors

- Original implementation: Vladislavs D.
- Enhancements and tools: Community contributors

## References

1. Huang, G.-B., Zhu, Q.-Y., & Siew, C.-K. (2006). "Extreme learning machine: Theory and applications." Neurocomputing, 70(1-3), 489-501.

2. Huang, G.-B., Zhou, H., Ding, X., & Zhang, R. (2012). "Extreme Learning Machine for Regression and Multiclass Classification." IEEE Transactions on Systems, Man, and Cybernetics, Part B, 42(2), 513-529.

3. Ding, S., Xu, X., & Nie, R. (2014). "Extreme learning machine and its applications." Neural Computing and Applications, 25(3-4), 549-556.

