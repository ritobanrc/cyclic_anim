#[allow(dead_code)]
mod math;

use crate::math::*;
use argmin::core::observers::{ObserverMode, SlogLogger};
use argmin::core::{CostFunction, Error, Executor};
use argmin::solver::neldermead::NelderMead;
use ndarray::Array1;
use rand::prelude::*;

type State = Array1<RV>;

struct GravitySim {
    pos0: State,
    pos1: State,
    pos0_weight: Real,
    pos1_weight: Real,
    strength: Real,
    dt: Real,
    timesteps: usize,
    particles: usize,
}

impl GravitySim {
    fn mul_mass(&self, s: State) -> State {
        s // for now, assume everything has mass 1
    }

    fn mass_inv_dot(&self, u: &State, v: &State) -> Real {
        u.iter().zip(v).map(|(a, b)| a.dot(b)).sum()
    }

    fn mass_dot(&self, u: &State, v: &State) -> Real {
        u.iter().zip(v).map(|(a, b)| a.dot(b)).sum()
    }

    fn compute_forces(&self, state: &Array1<RV>) -> State {
        let mut forces = State::from_elem([self.pos0.len()], RV::zeros());
        for pos_i in state {
            for (j, pos_j) in state.iter().enumerate() {
                let r = pos_i - pos_j;
                let m3 = r.magnitude().powi(3);
                forces[j] += self.strength * r / m3;
            }
        }
        forces
    }
}

impl CostFunction for GravitySim {
    type Param = Array1<Real>;
    type Output = Real;

    fn cost(&self, param: &Self::Param) -> Result<Real, Error> {
        let anim = param
            .clone()
            .into_shape([self.timesteps, self.particles, DIM])
            .expect("Unable to reshape array");
        let anim = anim
            .outer_iter()
            .map(|row| {
                row.outer_iter()
                    .map(|v| RV::new(v[0], v[1])) // TODO: not dimension independent (tbf this code is completely terrible)
                    .collect::<Array1<RV>>()
            })
            .collect::<Array1<Array1<RV>>>();

        let cycle = |i: usize| i.rem_euclid(self.timesteps);

        let mut physical_loss = 0.;
        for (i, state) in anim.iter().enumerate() {
            let one_over_dt_squared = 1. / (self.dt * self.dt);
            let accel =
                (&anim[cycle(i + 1)] - &anim[i] * 2. + &anim[cycle(i - 1)]) * one_over_dt_squared;
            let ma = self.mul_mass(accel);

            let forces = self.compute_forces(&state);

            let u = ma - forces;
            physical_loss += self.dt / 2. * self.mass_inv_dot(&u, &u);
        }

        let delta0 = &anim[0] - &self.pos0;
        let delta1 = &anim[1] - &self.pos1;
        let r0_term =
            1. / (2. * self.dt.powi(3) * self.pos0_weight) * self.mass_dot(&delta0, &delta0);
        let r1_term =
            1. / (2. * self.dt.powi(3) * self.pos1_weight) * self.mass_dot(&delta1, &delta1);

        Ok(physical_loss + r0_term + r1_term)
    }
}

fn main() {
    let mut rng = thread_rng();
    let particles = 5;
    let timesteps = 24;
    let dt = 0.1;

    let mut gen_random_state = || Array1::from_shape_fn([particles], |_| rng.gen::<RV>());

    let pos0 = gen_random_state();
    let vel0 = gen_random_state();
    let pos1 = &pos0 + vel0 * dt;
    let sim = GravitySim {
        pos0,
        pos1,
        pos0_weight: 1.,
        pos1_weight: 1.,
        strength: 0.1,
        dt: 0.1,
        timesteps,
        particles,
    };

    let mut gen_random_anim =
        || Array1::from_shape_fn([timesteps * particles * DIM], |_| rng.gen::<Real>());

    let dim = DIM * particles * timesteps + 1;
    let init_points: Vec<Array1<Real>> = (0..dim).map(|_| gen_random_anim()).collect();

    let opt = NelderMead::new(init_points);
    let _res = Executor::new(sim, opt)
        .configure(|state| state.max_iters(100))
        .add_observer(SlogLogger::term(), ObserverMode::Always)
        .run()
        .unwrap();
}
