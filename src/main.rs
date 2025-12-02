use std::time::Instant;

use nalgebra::{ArrayStorage, Const, Matrix, Vector, matrix};

// type alias's for a RxC matrix using stack storage
pub type ArrayMatrix<const R: usize, const C: usize> =
    Matrix<f64, Const<R>, Const<C>, ArrayStorage<f64, R, C>>;
// type alias's for a vector of dimension D using stack storage
pub type ArrayVector<const D: usize> = Vector<f64, Const<D>, ArrayStorage<f64, D, 1>>;

/// A function of the form f(x)=½xᵀQx-xᵀb+c
/// It has a derivative of f'(x)=Qx-b
pub fn quadratic<const D: usize>(
    q: ArrayMatrix<D, D>,
    b: ArrayVector<D>,
    c: f64,
) -> impl Fn(ArrayVector<D>) -> f64 {
    move |x| (0.5 * x.transpose() * q * x - x.transpose() * b)[0] + c
}

pub fn quadratic_gradient<const D: usize>(
    q: ArrayMatrix<D, D>,
    b: ArrayVector<D>,
) -> impl Fn(ArrayVector<D>) -> ArrayVector<D> {
    move |x| q * x - b
}

pub fn quadratic_hessian<const D: usize>(
    q: ArrayMatrix<D, D>,
) -> impl Fn(ArrayVector<D>) -> ArrayMatrix<D, D> {
    move |_| q
}

pub fn rosenbrock(a: f64, b: f64) -> impl Fn(ArrayVector<2>) -> f64 {
    // b(y-x²)² + (a-x)²
    move |x| b * (x.y - x.x.powi(2)).powi(2) + (a - x.x).powi(2)
}

pub fn rosenbrock_gradient(a: f64, b: f64) -> impl Fn(ArrayVector<2>) -> ArrayVector<2> {
    move |x| {
        matrix![
            // -4bx(y-x²) - 2(a-x)
            -4.0 * b * x.x * (x.y - x.x.powi(2)) - 2.0 * (a - x.x);
            // 2b(y-x²)
            2.0 * b * (x.y - x.x.powi(2));
        ]
    }
}

pub fn rosenbrock_hessian(_a: f64, b: f64) -> impl Fn(ArrayVector<2>) -> ArrayMatrix<2, 2> {
    move |x| {
        matrix![
            // 12bx²-4by+2, -4bx
            12.0 * b * x.x.powi(2) - 4.0 * b * x.y + 2.0, -4.0 * b * x.x;
            // -4bx, 2b
            -4.0 * b * x.x, 2.0 * b;
        ]
    }
}

// root finding algorithm
pub fn secant<F: Fn(f64) -> f64>(f: &F, mut a: f64, mut b: f64, n_max: usize, epsilon: f64) -> f64 {
    let mut f_a = f(a);
    let mut f_b = f(b);

    for _ in 0..n_max {
        if f_a.abs() > f_b.abs() {
            (a, b) = (b, a);
            (f_a, f_b) = (f_b, f_a);
        }

        let d = (b - a) / (f_b - f_a) * f_a;
        b = a;
        f_b = f_a;

        if d.abs() < epsilon || !d.is_finite() {
            return a;
        }

        a -= d;
        f_a = f(a);
    }

    a
}

pub fn steepest_descent<const D: usize>(
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
    x_0: ArrayVector<D>,
) -> ArrayVector<D> {
    const TOL: f64 = 1e-3;
    let mut x = x_0;
    let mut rel_err = 1.0;

    while rel_err > TOL {
        let df_x = df(x);

        let alpha = get_alpha_minimizer(-df_x, x, df);

        rel_err = (alpha * df_x).norm() / x.norm().max(1e-8);
        x -= alpha * df_x;
    }

    x
}

fn get_alpha_minimizer<const D: usize>(
    d: ArrayVector<D>,
    x: ArrayVector<D>,
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
) -> f64 {
    let d_phi = |a: f64| (d.transpose() * df(x + a * d))[0];
    let a_min = 0.0;
    let mut a_max = 1e-3;
    let d_phi_0 = d_phi(a_min);
    let mut d_phi_1 = d_phi(a_max);

    while d_phi_0 * d_phi_1 > 0.0 && a_max < 1e3 {
        a_max *= 2.0;
        d_phi_1 = d_phi(a_max);
    }

    secant(&d_phi, a_min, a_max, 50, 1e-6)
}

pub fn conjugate_gradient<const D: usize>(
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
    x_0: ArrayVector<D>,
) -> ArrayVector<D> {
    let mut x = x_0;
    let mut g = df(x);
    if g.norm() < 1e-6 {
        return x;
    }
    let mut d = -g;
    loop {
        let alpha = get_alpha_minimizer(d, x, df);
        if alpha < 1e-3 {
            return x;
        }

        x += alpha * d;
        let g_next = df(x);
        if g_next.norm() < 1e-6 {
            return x;
        }

        let beta = g_next.norm_squared() / g.norm_squared();
        g = g_next;

        d = -g_next + beta * d;

    }
}

fn newton_method<const D: usize>(
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
    hessian: &impl Fn(ArrayVector<D>) -> ArrayMatrix<D, D>,
    x_0: ArrayVector<D>,
) -> ArrayVector<D> {
    let mut x = x_0;

    let mut i = 0;

    loop {
        i += 1;
        let g = df(x);
        let h = hessian(x);

        // todo: try to use LU decomposition to solve instead of taking inverse
        // F(x)⁻¹gᵏ
        let hig = h.try_inverse().unwrap() * g;
        if hig.norm() < 1e-9 {
            break;
        }

        // x⁽ᵏ⁺¹⁾ = x⁽ᵏ⁾ - F(x)⁻¹gᵏ
        x -= hig;
    }

    x
}

fn bfgs<const D: usize>(
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
    x_0: ArrayVector<D>,
    h_0: ArrayMatrix<D, D>,
) -> ArrayVector<D> {
    let mut x = x_0;
    let mut h = h_0;
    let mut g = df(x);
    let mut i = 0;
    loop {
        if g.norm() < 1e-6 {
            return x;
        }
        i += 1;
        let d = -h * g;
        let alpha = get_alpha_minimizer(d, x, df);
        if alpha < 1e-6 {
            return x;
        }

        let x_1 = x + alpha * d;
        let g_1 = df(x_1);

        let dx = x_1 - x;
        let dg = g_1 - g;

        h += (1.0 + (dg.transpose() * h * dg)[0] / (dg.transpose() * dx)[0])
            * (dx * dx.transpose())
            / (dx.transpose() * dg)[0]
            - (h * dg * dx.transpose() + (h * dg * dx.transpose()).transpose())
                / (dg.transpose() * dx)[0];

        x = x_1;
        g = g_1;
    }
}

// #[test]
fn main() {

    {
        println!("----- QUADRATIC -----");

        let q = random_positive_semidefinite_matrix::<8>();
        println!("q: {q}");
        dbg!(q.eigenvalues());
        let b: ArrayVector<8> = rand::random();
        println!("b: {b}");
        let f = quadratic(q, b, 0.0);
        let df = quadratic_gradient(q, b);
        let d2f = quadratic_hessian(q);
        println!("{q}");

        let min_x = q.lu().solve(&b).unwrap();
        println!("min: {min_x}");
        println!("{}", f(min_x));
        test_fns(&f, &df, &d2f, ArrayVector::zeros(), ArrayMatrix::identity());
    }
    
    {
        let f = rosenbrock(1.0, 100.0);
        let df = rosenbrock_gradient(1.0, 100.0);
        let d2f = rosenbrock_hessian(1.0, 100.0);
        println!("----- ROSENBROCK -----");
        test_fns(&f, &df, &d2f, ArrayVector::zeros(), ArrayMatrix::identity());
    }
}

fn test_fns<const D: usize>(
    f: &impl Fn(ArrayVector<D>) -> f64,
    df: &impl Fn(ArrayVector<D>) -> ArrayVector<D>,
    d2f: &impl Fn(ArrayVector<D>) -> ArrayMatrix<D, D>,
    x_0: ArrayVector<D>,
    h_0: ArrayMatrix<D, D>,
) {
    {
        let now = Instant::now();
        let x = steepest_descent(df, x_0);
        let elapsed = Instant::now() - now;
        print!("steepest_descent: {x:.3}");
        println!("{:.3}", f(x));
        println!("{:.2?}\n", elapsed)
    }
    {
        let now = Instant::now();
        let x = conjugate_gradient(df, x_0);
        let elapsed = Instant::now() - now;
        print!("conjugate_gradient: {x:.3}");
        println!("{:.3}", f(x));
        println!("{:.2?}\n", elapsed)
    }
    {
        let now = Instant::now();
        let x = newton_method(df, d2f, x_0);
        let elapsed = Instant::now() - now;
        print!("newton_method: {x:.3}");
        println!("{:.3}", f(x));
        println!("{:.2?}\n", elapsed)
    }
    {
        let now = Instant::now();
        let x = bfgs(df, x_0, h_0);
        let elapsed = Instant::now() - now;
        print!("bfgs: {x:.3}");
        println!("{:.3}", f(x));
        println!("{:.2?}\n", elapsed)
    }
}

fn random_positive_semidefinite_matrix<const D: usize>() -> ArrayMatrix<D, D> {
    let m: ArrayMatrix<D, D> = rand::random();

    m * m.transpose()
}
