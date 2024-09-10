mod matrix;
mod neuron;
mod rand;
use neuron::Neuron;

fn main() {
    let mut vec = vec![0,1,2,3,4,5,6,7,8,9];
    rand::shuffle(&mut vec);
    println!("{:?}", vec);

    let mut neuron = Neuron::new(2);
    neuron.push_layer(30, Neuron::leaky_relu, Neuron::leaky_relu_diff);
    neuron.push_layer(30, Neuron::leaky_relu, Neuron::leaky_relu_diff);
    neuron.push_layer(1, Neuron::identity, Neuron::identity_diff);

    const max: usize = 10;
    for y in 0 .. max {
        for x in 0 .. max {
            neuron.push_target(&[y as f64, x as f64], &[(x*x+y) as f64]);
        }
    }

    neuron.learn(1000, 0.00001, 50);

    for y in 0 .. max {
        for x in 0 .. max {
            neuron.run(&[y as f64, x as f64]);
            println!("({}, {}): {:?}", y, x, neuron.result());
        }
    }
}
