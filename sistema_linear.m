function [sol_jacobi, iter_jacobi, sol_seidel, iter_seidel] = linear_system_solver(A, b, tol, max_iter)
    n = length(b);
    x = zeros(n, 1);
    x_old = zeros(n, 1);

    % Verificar se a matriz A é diagonalmente dominante para ambos os métodos
    if all(abs(diag(A)) > sum(abs(A), 2) - abs(diag(A)))
        disp('A matriz é diagonalmente dominante.');

        % Metodo de Gauss-Jacobi
        for k = 1:max_iter
            for i = 1:n
                sigma = A(i, 1:i-1) * x_old(1:i-1) + A(i, i+1:end) * x_old(i+1:end);
                x(i) = (b(i) - sigma) / A(i, i);
            end

            if all(abs(x - x_old) < tol)
                sol_jacobi = x;
                iter_jacobi = k;
                fprintf('\nSolução Gauss-Jacobi:\n');
                disp(sol_jacobi);
                fprintf('Iterações necessárias: %d\n', iter_jacobi);
                return;
            end

            x_old = x;
        end

        sol_jacobi = [];
        iter_jacobi = max_iter;
        fprintf('\nGauss-Jacobi não convergiu.\n');

        % Metodo de Gauss-Seidel
        for k = 1:max_iter
            for i = 1:n
                sigma = A(i, 1:i-1) * x(1:i-1) + A(i, i+1:end) * x(i+1:end);
                x(i) = (b(i) - sigma) / A(i, i);
            end

            if all(abs(x - x_old) < tol)
                sol_seidel = x;
                iter_seidel = k;
                fprintf('\nSolução Gauss-Seidel:\n');
                disp(sol_seidel);
                fprintf('Iterações necessárias: %d\n', iter_seidel);
                return;
            end
        end

        sol_seidel = [];
        iter_seidel = max_iter;
        fprintf('\nGauss-Seidel não convergiu.\n');

    % Caso a Matriz não seja dominante, não irá convergir
    else
        disp('A matriz não é diagonalmente dominante');
        sol_jacobi = [];
        iter_jacobi = 0;
        sol_seidel = [];
        iter_seidel = 0;
    end
end

