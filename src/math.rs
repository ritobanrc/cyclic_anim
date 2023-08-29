use nalgebra as na;
pub const DIM: usize = 2;

pub type Dim = na::Const<DIM>;

pub type Real = f64;
pub type RV = na::SVector<Real, DIM>;
//pub type IntVector = na::SVector<isize, DIM>;
pub type UV = na::SVector<usize, DIM>;

pub use std::f64::consts;
