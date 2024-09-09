use std::fmt::{Display, Formatter, Error}; 
use std::ops::*;
use crate::rand::rand;

pub struct Matrix {
    width: usize,
    height: usize,
    vec: Vec<f64>
}

impl Matrix {
    pub fn new(width: usize, height: usize) -> Matrix {
        let mut vec = Vec::new();
        for _ in 0 .. width*height {
            vec.push(0.0);
        }
        Matrix { width: width, height: height, vec: vec }
    }
    
    pub fn from(width: usize, height: usize, matrix: &[f64]) -> Matrix {
        let mut new_matrix = Matrix::new(width, height);
        for y in 0 .. height {
            for x in 0 .. width {
                new_matrix[y][x] = matrix[x + y*width];
            }
        }
        
        new_matrix
    }

    pub fn fill(&mut self, n: f64) -> &mut Self {
        for y in 0 .. self.height {
            for x in 0 .. self.width {
                self[y][x] = n;
            }
        }
        self
    }
    
    pub fn rand(width: usize, height: usize) -> Matrix {
        let mut matrix = Matrix::new(width, height);

        for y in 0 .. height {
            for x in 0 .. width {
                matrix[y][x] = (rand() as f64 / 0xffff_ffff_ffff_ffffu64 as f64)*2.0-1.0;
            }
        }

        matrix
    }
    
    pub fn dot(&mut self, lhs: &Self, rhs: &Self) -> &mut Self {
        if lhs.width != rhs.height {
            panic!("invalid input");
        }

        if (self.width, self.height) != (rhs.width, lhs.height) {
            panic!("invalid input");
        }
        
        for y in 0 .. self.height {
            for x in 0 .. self.width {
                for i in 0 .. lhs.width {
                    self[y][x] += lhs[y][i] * rhs[i][x];
                }
            }
        }
        
        self
    }

    pub fn to_vec(&self) -> Vec<f64> {
        let mut vec = Vec::<f64>::new();

        for y in 0 .. self.height {
            for x in &self[y] {
                vec.push(*x);
            }
        }

        vec
    }

    pub fn map<T: Fn(&mut f64)>(&mut self, clsr: T) -> &mut Self {
        for y in 0 .. self.height {
            for x in  0 .. self.width {
                clsr(&mut self[y][x]);
            }
        }

        self
    }

    pub fn width(&self) -> usize {
        self.width
    }

    pub fn height(&self) -> usize {
        self.height
    }
}

impl Clone for Matrix {
    fn clone(&self) -> Self {
        Matrix { width: self.width, height: self.height, vec: self.vec.clone() }
    }
}

impl Index<usize> for Matrix {
    type Output = [f64];
    
    fn index(&self, index: usize) -> &Self::Output {
        if self.height <= index {
            panic!("out of range");
        }

        &self.vec[index*self.width .. (index+1)*self.width]
    }
}

impl IndexMut<usize> for Matrix {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        if self.height <= index {
            panic!("out of range");
        }

        &mut self.vec[index*self.width .. (index+1)*self.width]
    }
}

impl Add<&Matrix> for Matrix {
    type Output = Matrix;
    fn add(mut self, rhs: &Matrix) -> Self::Output {
        self += rhs;
        self
    }
}

impl AddAssign<&Matrix> for Matrix {
    fn add_assign(&mut self, rhs: &Matrix) {
        if (self.width, self.height) == (rhs.width, rhs.height) {
            for y in 0 .. self.height {
                for x in 0 .. self.width as usize {
                    self[y][x] += rhs[y][x];
                }
            }
        }else {
            panic!("addition bitween different size");
        }
    }
}

impl Display for Matrix {
    fn fmt(&self, f: &mut Formatter<'_>) -> Result<(), Error> {
        write!(f, "{{")?;
        for y in 0 .. self.height {
            if y != 0 {
                write!(f, ",\n {{")?;
            }else {
                write!(f, "{{")?;
            }
            for x in 0 .. self.width {
                if x != 0 {
                    write!(f, ", ")?;
                }
                write!(f, "{}", self.vec[x + y*self.width])?;
            }
            write!(f, "}}")?;
        }
        write!(f, "}}")?;
        Ok(())
    }
}
