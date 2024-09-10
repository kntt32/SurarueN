mod matrix;
mod neuron;
mod rand;
use neuron::Neuron;

fn main() {
    let mut vec = vec![0,1,2,3,4,5,6,7,8,9];
    rand::shuffle(&mut vec);
    println!("{:?}", vec);

    let mut neuron = Neuron::new(2);
    neuron.push_layer(20, Neuron::relu, Neuron::relu_diff);
    neuron.push_layer(1, Neuron::identity, Neuron::identity_diff);

    neuron.push_target(&[0.0, 0.0], &[0.0]);
    neuron.push_target(&[0.0, 1.0], &[1.0]);
    neuron.push_target(&[1.0, 0.0], &[1.0]);
    neuron.push_target(&[1.0, 1.0], &[1.0]);

    neuron.learn(500, 0.01);

    for y in 0 .. 2 {
        for x in 0 .. 2 {
            neuron.run(&[y as f64, x as f64]);
            println!("({}, {}): {:?}", y, x, neuron.result());
        }
    }
}
