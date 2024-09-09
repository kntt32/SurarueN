use crate::matrix::Matrix;

use std::fmt::{Display, Formatter, Error};



#[derive(Clone)]
pub struct Neuron {
    target: Vec<Target>,
    layer: Vec<Layer>
}


#[derive(Clone)]
struct Target {
    input: Vec<f64>,
    output: Vec<f64>
}


#[derive(PartialEq)]
enum RunMode {
    Run,
    Learn
}


#[derive(Clone)]
struct Layer {
    neurons: usize,

    u: Matrix,
    y: Matrix,

    weight: Matrix,
    bias: Matrix,

    delta: Matrix,

    u_diff: Matrix,
    weight_diff: Matrix,
    bias_diff: Matrix,

    activation: fn(&mut Matrix, &Matrix),
    activation_diff: fn(&mut Matrix, &Matrix)
}


fn activation_null(_y: &mut Matrix, _u: &Matrix) {
    panic!("activation_null called");
}


fn activation_diff_null(_y: &mut Matrix, _u: &Matrix) {
    panic!("activation_diff_null called");
}



impl Neuron {
    pub fn new(inputs: usize) -> Neuron {
        let mut neuron = Neuron { target: Vec::new(), layer: Vec::new() };

        neuron.layer.push(
            Layer {
                neurons: inputs,

                y: Matrix::new(1, inputs),
                u: Matrix::new(0, 0),

                weight: Matrix::new(0, 0),
                bias: Matrix::new(0, 0),

                delta: Matrix::new(0, 0),

                u_diff: Matrix::new(0, 0),
                weight_diff: Matrix::new(0, 0),
                bias_diff: Matrix::new(0, 0),

                activation: activation_null,
                activation_diff: activation_diff_null
            }
            );

        neuron
    }


    pub fn push_layer(&mut self, neurons: usize, activation: fn(&mut Matrix, &Matrix), activation_diff: fn(&mut Matrix, &Matrix)) -> &mut Self {
        let neurons_of_prev_layer = self.layer[self.layer.len() - 1].neurons;

        self.layer.push(
            Layer {
                neurons: neurons,

                u: Matrix::new(1, neurons),
                y: Matrix::new(1, neurons),
                
                weight: Matrix::rand(neurons_of_prev_layer, neurons),
                bias: Matrix::rand(1, neurons),

                delta: Matrix::new(1, neurons),

                u_diff: Matrix::new(1, neurons),
                weight_diff: Matrix::new(neurons_of_prev_layer, neurons),
                bias_diff: Matrix::new(1, neurons),

                activation: activation,
                activation_diff: activation_diff
            });

        self
    }


    pub fn run(&mut self, input: &[f64]) -> &mut Self {
        self.run_(input, RunMode::Run);
        self
    }


    fn run_(&mut self, input: &[f64], mode: RunMode) {
        if input.len() != self.layer[0].neurons {
            panic!("invalid input");
        }

        let layer_count = self.layer.len();

        self.layer[0].y = Matrix::from(1, self.layer[0].neurons, input);

        for i in 1 .. layer_count {
            let (left_slice, right_slice) = self.layer.split_at_mut(i);
            let target_layer = &mut right_slice[0];
            let prev_layer = &left_slice[i-1];

            target_layer.u.fill(0.0);

            target_layer.u.dot(&(target_layer.weight), &prev_layer.y);
            target_layer.u += &target_layer.bias;

            (target_layer.activation)(&mut target_layer.y, &target_layer.u);

            if mode == RunMode::Learn {
                (target_layer.activation_diff)(&mut target_layer.u_diff, &target_layer.u);
            }
        }
    }


    pub fn result(&self) -> Vec<f64> {
        self.layer[self.layer.len()-1].y.to_vec()
    }
}


impl Neuron {
    pub fn push_target(&mut self, input: &[f64], output: &[f64]) -> &mut Self {
        let input_vec = Vec::from(input);
        let output_vec = Vec::from(output);

        self.target.push( Target { input: input_vec, output: output_vec } );

        self
    }

    pub fn learn(&mut self, epoch: usize, learning_rate: f64) {
        for _ in 0 .. epoch {
            self.learn_(learning_rate);
        }
    }

    fn learn_(&mut self, learning_rate: f64) {
        let mut reset_diff = || {
            for i in 1 .. self.layer.len() {
                let target_layer = &mut self.layer[i];

                for y in 0 .. target_layer.weight_diff.height() {
                    for x in 0 .. target_layer.weight_diff.width() {
                        target_layer.weight_diff[y][x] = 0.0;
                    }
                }

                for y in 0 .. target_layer.bias_diff.height() {
                    target_layer.bias_diff[y][0] = 0.0;
                }
            }
        };

        reset_diff();

        for target_index in 0 .. self.target.len() {
            self.run_(&self.target[target_index].input.clone()[..], RunMode::Learn);

            let mut calc_delta = || {
                let layer_len = self.layer.len();

                {
                    let target_layer = &mut self.layer[layer_len-1];
                    for i in 0 .. target_layer.neurons {
                        target_layer.delta[i][0] = target_layer.y[i][0] - self.target[target_index].output[i];
                    }
                }

                for layer_index in (1 .. layer_len-1).rev() {
                    let (left_slice, right_slice) = self.layer.split_at_mut(layer_index+1);
                    let next_layer = &right_slice[0];
                    let target_layer = &mut left_slice[layer_index];
                    
                    for i in 0 .. target_layer.neurons {
                        let mut sum = 0.0;
                        for k in 0 .. next_layer.neurons {
                            sum += next_layer.delta[k][0]*next_layer.weight[k][i];
                        }
                        sum *= target_layer.u_diff[i][0];

                        target_layer.delta[i][0] = sum;

                    }
                }
            };

            calc_delta();

            let mut add_diff = || {
                for layer_index in 1 .. self.layer.len() {
                    let (left_slice, right_slice) = self.layer.split_at_mut(layer_index);
                    let prev_layer = &left_slice[layer_index-1];
                    let target_layer = &mut right_slice[0];

                    for y in 0 .. target_layer.neurons {
                        for x in 0 .. prev_layer.neurons {
                            target_layer.weight_diff[y][x] += target_layer.delta[y][0]*prev_layer.y[x][0];
                        }
                    }

                    for y in 0 .. target_layer.neurons {
                        target_layer.bias_diff[y][0] += target_layer.delta[y][0];
                    }
                }
            };

            add_diff();
        }

        let mut change_parameter = || {
            for layer_index in 1 .. self.layer.len() {
                let target_layer = &mut self.layer[layer_index];

                for y in 0 .. target_layer.weight.height() {
                    for x in 0 .. target_layer.weight.width() {
                        target_layer.weight[y][x] -= target_layer.weight_diff[y][x]*learning_rate;
                    }
                }

                for y in 0 .. target_layer.bias.height() {
                    target_layer.bias[y][0] -= target_layer.bias_diff[y][0]*learning_rate;
                }
            }
        };

        change_parameter();
    }
}


impl Neuron {
    pub fn relu(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = if 0.0 <= u[i][0] { u[i][0] } else { 0.0 };
        }
    }

    pub fn relu_diff(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = if 0.0 <= u[i][0] { 1.0 } else { 0.0 };
        }
    }

    pub fn leaky_relu(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = if 0.0 <= u[i][0] { u[i][0] } else { 0.01*u[i][0] };
        }
    }

    pub fn leaky_relu_diff(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = if 0.0 <= u[i][0] { 1.0 } else { 0.01 };
        }
    }

    pub fn identity(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = u[i][0];
        }
    }

    pub fn identity_diff(y: &mut Matrix, u: &Matrix) {
        for i in 0 .. u.height() {
            y[i][0] = 1.0;
        }
    }
}


impl Display for Neuron { 
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        for i in 0 .. self.layer.len() {
            writeln!(f, "{}:", i)?;
            writeln!(f, "    neurons: {}", self.layer[i].neurons)?;
            writeln!(f, "    weight: \n{}", self.layer[i].weight)?;
            writeln!(f, "    bias: \n{}", self.layer[i].bias)?;
        }

        Ok(())
    }
}
