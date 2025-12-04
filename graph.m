rosenbrock_steepest_descent = readtable("rosenbrock_steepest_descent.csv")
rosenbrock_conjugate_gradient = readtable("rosenbrock_conjugate_gradient.csv")
rosenbrock_newton_method = readtable("rosenbrock_newton_method.csv")
rosenbrock_bfgs = readtable("rosenbrock_bfgs.csv")

figure; hold on;

plot(rosenbrock_steepest_descent.i,       rosenbrock_steepest_descent.f,       'DisplayName', 'Steepest Descent');
plot(rosenbrock_conjugate_gradient.i,     rosenbrock_conjugate_gradient.f,     'DisplayName', 'Conjugate Gradient');
plot(rosenbrock_newton_method.i,          rosenbrock_newton_method.f,          'DisplayName', 'Newton Method');
plot(rosenbrock_bfgs.i,                   rosenbrock_bfgs.f,                   'DisplayName', 'BFGS');

xlabel('k');
ylabel('f(x^{(k)})');
title('Rosenbrock Optimization');

xlim([0, 10])
ylim([0,1.5])

legend;
grid on;
hold off;

